"""Market data access (Alpaca) with an on-disk cache.

Rules:
* Data comes from Alpaca (or from this cache of earlier Alpaca downloads).
* Missing data raises DataUnavailable with a precise message. Nothing is ever
  interpolated, synthesized, or "filled in" with made-up prices.
* Only complete past sessions are cached.
"""

from __future__ import annotations

import calendar as _calendar
import datetime as dt
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .timeutil import ET, et_timestamp, now_et, parse_hhmm

log = logging.getLogger(__name__)

BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]
_ALPACA_FIELDS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "trade_count"}
DAILY_COLUMNS = ["open", "high", "low", "close", "volume"]
ALPACA_OPTIONS_HISTORY_START = dt.date(2024, 2, 1)


class DataUnavailable(RuntimeError):
    """Required market data is missing. The bot never substitutes made-up prices."""


@dataclass(frozen=True)
class TradingDay:
    date: dt.date
    open_time: dt.time = dt.time(9, 30)
    close_time: dt.time = dt.time(16, 0)

    @property
    def open_ts(self) -> pd.Timestamp:
        return et_timestamp(self.date, self.open_time)

    @property
    def close_ts(self) -> pd.Timestamp:
        return et_timestamp(self.date, self.close_time)

    @property
    def is_early_close(self) -> bool:
        return self.close_time < dt.time(16, 0)

    @property
    def session_minutes(self) -> int:
        return int((self.close_ts - self.open_ts) / pd.Timedelta(minutes=1))


# --------------------------------------------------------------------------- #
# Frame helpers
# --------------------------------------------------------------------------- #
def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype=float) for c in BAR_COLUMNS},
        index=pd.DatetimeIndex([], tz=ET, name="timestamp"),
    )


def bars_from_alpaca(raw: list[dict] | None) -> pd.DataFrame:
    """Alpaca bar dicts -> DataFrame indexed by bar START time (ET)."""
    if not raw:
        return empty_bars()
    df = pd.DataFrame.from_records(raw)
    if "t" not in df.columns:
        raise DataUnavailable("Alpaca bar payload has no 't' (timestamp) field")
    index = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True)).tz_convert(ET)
    out = pd.DataFrame(index=index)
    for src, dst in _ALPACA_FIELDS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce").to_numpy(dtype=float) if src in df else np.nan
    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def daily_from_alpaca(raw: list[dict] | None) -> pd.DataFrame:
    """Alpaca 1Day bars -> DataFrame indexed by naive session date."""
    bars = bars_from_alpaca(raw)
    if bars.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype=float)
    idx = pd.DatetimeIndex(bars.index.tz_localize(None).normalize(), name="date")
    out = bars[DAILY_COLUMNS].copy()
    out.index = idx
    return out[~out.index.duplicated(keep="last")].sort_index()


def regular_session(bars: pd.DataFrame, day: TradingDay) -> pd.DataFrame:
    """Bars whose start time is within [open, close)."""
    if bars.empty:
        return bars
    return bars.loc[(bars.index >= day.open_ts) & (bars.index < day.close_ts)]


def _write_bars(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index_label="timestamp")
    tmp.replace(path)


def _read_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    idx = pd.DatetimeIndex(pd.to_datetime(df.pop("timestamp"), utc=True)).tz_convert(ET)
    df.index = idx
    df.index.name = "timestamp"
    for col in BAR_COLUMNS:
        if col not in df:
            df[col] = np.nan
    return df[BAR_COLUMNS].astype(float)


def _write_daily(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index_label="date", date_format="%Y-%m-%d")
    tmp.replace(path)


def _read_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("date")), name="date")
    return df[DAILY_COLUMNS].astype(float)


def strike_grid(near_price: float, increment: float, width: int = 60) -> list[float]:
    base = round(near_price / increment) * increment
    return [round(base + i * increment, 4) for i in range(-width, width + 1)]


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
class MarketDataProvider(ABC):
    """Everything the backtester and runner need. Implementations must raise
    DataUnavailable rather than return invented data."""

    @abstractmethod
    def trading_days(self, start: dt.date, end: dt.date) -> list[TradingDay]: ...

    @abstractmethod
    def minute_bars(self, day: TradingDay) -> pd.DataFrame:
        """Regular-session 1-minute bars for a past/complete session (index = bar start, ET)."""

    @abstractmethod
    def daily_bars(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Daily bars, index = naive session date."""

    @abstractmethod
    def option_strikes(self, expiry: dt.date, right: str, near_price: float) -> tuple[list[float], str]:
        """(sorted strikes listed for this expiry/right, source label)."""

    @abstractmethod
    def option_minute_bars(self, symbol: str, day: TradingDay) -> pd.DataFrame: ...


class AlpacaDataProvider(MarketDataProvider):
    def __init__(
        self,
        client,
        *,
        underlying: str,
        stock_feed: str,
        adjustment: str = "raw",
        cache_dir: str | Path = "data_cache",
        strike_grid_fallback: float | None = 1.0,
        offline: bool = False,
        today: dt.date | None = None,
    ) -> None:
        if client is None and not offline:
            raise ValueError("AlpacaDataProvider needs an AlpacaClient unless offline=True")
        self.client = client
        self.underlying = underlying
        self.stock_feed = stock_feed
        self.adjustment = adjustment
        self.cache_dir = Path(cache_dir)
        self.strike_grid_fallback = strike_grid_fallback
        self.offline = offline
        self._fixed_today = today
        self._calendar_mem: dict[int, list[TradingDay]] = {}

    # ---------------------------------------------------------------- utils
    def _today(self) -> dt.date:
        return self._fixed_today or now_et().date()

    def _require_online(self, what: str, path: Path) -> None:
        if self.offline:
            raise DataUnavailable(
                f"--offline: {what} is not in the cache ({path}). Run once with network access and "
                "Alpaca keys to download it."
            )

    # ------------------------------------------------------------- calendar
    def _calendar_year(self, year: int) -> list[TradingDay]:
        if year in self._calendar_mem:
            return self._calendar_mem[year]
        path = self.cache_dir / "calendar" / f"{year}.json"
        # Past years never change. The current/future year's calendar is refreshed once a day when online.
        fresh = path.exists() and (
            self.offline or year < self._today().year or (time.time() - path.stat().st_mtime) < 86400
        )
        if fresh:
            raw = json.loads(path.read_text())
        else:
            self._require_online(f"Alpaca trading calendar for {year}", path)
            raw = self.client.calendar(dt.date(year, 1, 1), dt.date(year, 12, 31))
            if not raw:
                raise DataUnavailable(f"Alpaca /v2/calendar returned no trading days for {year}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw))
        days = [
            TradingDay(dt.date.fromisoformat(r["date"]), parse_hhmm(r["open"]), parse_hhmm(r["close"]))
            for r in raw
        ]
        days.sort(key=lambda d: d.date)
        self._calendar_mem[year] = days
        return days

    def trading_days(self, start: dt.date, end: dt.date) -> list[TradingDay]:
        out: list[TradingDay] = []
        for year in range(start.year, end.year + 1):
            out.extend(d for d in self._calendar_year(year) if start <= d.date <= end)
        return out

    # ---------------------------------------------------------- stock bars
    def minute_bars(self, day: TradingDay) -> pd.DataFrame:
        path = (
            self.cache_dir / "stocks" / self.underlying / "1Min" / self.stock_feed / f"{day.date.isoformat()}.csv"
        )
        if path.exists():
            return _read_bars(path)
        self._require_online(f"1-minute {self.underlying} bars for {day.date}", path)
        raw = self.client.stock_bars(
            self.underlying, "1Min", day.open_ts, day.close_ts, feed=self.stock_feed, adjustment=self.adjustment
        )
        bars = regular_session(bars_from_alpaca(raw), day)
        if bars.empty:
            raise DataUnavailable(
                f"Alpaca returned no 1-minute {self.underlying} bars for {day.date} "
                f"{day.open_time:%H:%M}-{day.close_time:%H:%M} ET (feed={self.stock_feed})."
            )
        if day.date < self._today():
            _write_bars(bars, path)
        return bars

    def daily_bars(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames = []
        year, month = start.year, start.month
        while (year, month) <= (end.year, end.month):
            month_start = dt.date(year, month, 1)
            month_end = dt.date(year, month, _calendar.monthrange(year, month)[1])
            path = self.cache_dir / "stocks" / self.underlying / "1Day" / self.stock_feed / f"{year:04d}-{month:02d}.csv"
            # Complete months are read from cache. The current month is re-downloaded when online and read
            # from cache only when offline; atr_for_day() still verifies every needed date is present.
            if path.exists() and (month_end < self._today() or self.offline):
                frames.append(_read_daily(path))
            else:
                self._require_online(f"daily {self.underlying} bars for {year}-{month:02d}", path)
                raw = self.client.stock_bars(
                    self.underlying,
                    "1Day",
                    et_timestamp(month_start, "00:00"),
                    et_timestamp(month_end, "23:59"),
                    feed=self.stock_feed,
                    adjustment=self.adjustment,
                )
                df = daily_from_alpaca(raw)
                today_ts = pd.Timestamp(self._today())
                complete = df.loc[df.index < today_ts]  # never cache today's partial daily bar
                if not complete.empty:
                    _write_daily(complete, path)
                frames.append(df)
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=DAILY_COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype=float)
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    # -------------------------------------------------------------- options
    def option_strikes(self, expiry: dt.date, right: str, near_price: float) -> tuple[list[float], str]:
        if right not in ("call", "put"):
            raise ValueError(f"right must be call/put, got {right!r}")
        path = self.cache_dir / "options" / "contracts" / self.underlying / f"{expiry.isoformat()}_{right}.json"
        strikes: list[float] = []
        if path.exists():
            strikes = [float(x) for x in json.loads(path.read_text())]
        elif not self.offline:
            prefix = f"{self.underlying}{expiry:%y%m%d}{right[0].upper()}"
            contracts: list[dict] = []
            for status in ("active", "inactive"):
                contracts.extend(
                    self.client.option_contracts(
                        underlying=self.underlying, expiration_date=expiry, contract_type=right, status=status
                    )
                )
            strikes = sorted(
                {
                    float(c["strike_price"])
                    for c in contracts
                    if str(c.get("symbol", "")).startswith(prefix)  # standard (non-adjusted) contracts only
                    and str(c.get("expiration_date")) == expiry.isoformat()
                }
            )
            if strikes and expiry < self._today():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(strikes))
        if strikes:
            return strikes, "alpaca_contracts"
        if self.strike_grid_fallback:
            log.warning(
                "No %s %s contracts listed by Alpaca for expiry %s%s. Using ASSUMED $%.2f strike grid "
                "(data.strike_grid_fallback). The contract must still have real Alpaca trade bars.",
                self.underlying, right, expiry, " (offline, not cached)" if self.offline else "",
                self.strike_grid_fallback,
            )
            return strike_grid(near_price, self.strike_grid_fallback), f"grid_fallback_{self.strike_grid_fallback:g}"
        raise DataUnavailable(
            f"Alpaca /v2/options/contracts returned no {self.underlying} {right} contracts expiring {expiry} "
            "(tried status=active and inactive), and data.strike_grid_fallback is null."
        )

    def option_minute_bars(self, symbol: str, day: TradingDay) -> pd.DataFrame:
        path = self.cache_dir / "options" / "bars" / symbol / f"{day.date.isoformat()}.csv"
        if path.exists():
            return _read_bars(path)
        self._require_online(f"1-minute bars for option {symbol} on {day.date}", path)
        if day.date < ALPACA_OPTIONS_HISTORY_START:
            raise DataUnavailable(
                f"Alpaca historical options data starts {ALPACA_OPTIONS_HISTORY_START}; "
                f"{symbol} on {day.date} is earlier than that."
            )
        raw = self.client.option_bars([symbol], "1Min", day.open_ts, day.close_ts).get(symbol, [])
        bars = regular_session(bars_from_alpaca(raw), day)
        if bars.empty:
            raise DataUnavailable(
                f"Alpaca /v1beta1/options/bars returned NO 1-minute trade bars for {symbol} on {day.date}. "
                "Either the contract didn't trade or doesn't exist (check the strike), or your Alpaca plan "
                "doesn't include this options history. No price was substituted."
            )
        if day.date < self._today():
            _write_bars(bars, path)
        return bars


def is_finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
