"""SYNTHETIC fixtures for the unit tests ONLY.

These prices are made up so the tests can check the plumbing and rule logic
deterministically. Nothing in qqq0dte/ imports this module. The backtester and
the runner only ever use real Alpaca data (or its cache).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from qqq0dte.data import DataUnavailable, MarketDataProvider, TradingDay
from qqq0dte.options import occ_symbol


def weekdays_ending(end: dt.date, n: int) -> list[TradingDay]:
    out: list[TradingDay] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(TradingDay(d))
        d -= dt.timedelta(days=1)
    return out[::-1]


def bars_from_closes(day: TradingDay, closes, first_open: float | None = None, volume: float = 1000.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(day.open_ts, periods=len(closes), freq="1min", name="timestamp")
    opens = np.concatenate([[closes[0] if first_open is None else first_open], closes[:-1]])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 0.01,
            "low": np.minimum(opens, closes) - 0.01,
            "close": closes,
            "volume": float(volume),
            "vwap": (opens + closes) / 2,
            "trade_count": 10.0,
        },
        index=idx,
    )


def option_bars_from_underlying(und: pd.DataFrame, strike: float, right: str, extrinsic: float = 0.5) -> pd.DataFrame:
    def value(s):
        intrinsic = np.maximum(strike - s, 0.0) if right == "put" else np.maximum(s - strike, 0.0)
        return intrinsic + extrinsic

    o, c = value(und["open"].to_numpy()), value(und["close"].to_numpy())
    return pd.DataFrame(
        {"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 100.0, "vwap": (o + c) / 2, "trade_count": 5.0},
        index=und.index,
    )


# --------------------------------------------------------------------------- paths
def put_day_closes(base: float = 100.0) -> np.ndarray:
    """Opening range ~[base-0.06, base+0.06]; drifts flat, then breaks down 10:20-10:30 to base-1.0."""
    c = np.empty(390)
    c[0:15] = [base + 0.05 if i % 2 == 0 else base - 0.05 for i in range(15)]
    c[15:50] = base - 0.05
    c[50:60] = np.linspace(base - 0.15, base - 1.0, 10)  # 10:30 check sees base-1.0
    c[60:] = np.linspace(base - 1.02, base - 1.5, 330)
    return c


def call_day_closes(base: float = 100.0) -> np.ndarray:
    return 2 * base - put_day_closes(base)


def flat_day_closes(base: float = 100.0) -> np.ndarray:
    c = np.full(390, base)
    c[0:15] = [base + 0.05 if i % 2 == 0 else base - 0.05 for i in range(15)]
    return c


def trailing_put_day_closes(base: float = 100.0) -> np.ndarray:
    """Put at 10:30 (base-1.0), then falls to base-6 (5 ATR favorable at ATR=1) by ~12:00, then rebounds."""
    c = put_day_closes(base)
    c[60:150] = np.linspace(base - 1.05, base - 6.0, 90)  # bottom at bar 149 (11:59)
    c[150:210] = np.linspace(base - 5.95, base - 3.0, 60)  # rebound
    c[210:] = base - 3.0
    return c


# --------------------------------------------------------------------------- provider
class SyntheticProvider(MarketDataProvider):
    def __init__(self, days, minute, daily, options, strikes):
        self.days = days
        self.minute = minute
        self.daily = daily
        self.options = options
        self.strikes = strikes
        self.option_requests: list[str] = []

    def trading_days(self, start, end):
        return [d for d in self.days if start <= d.date <= end]

    def minute_bars(self, day):
        if day.date not in self.minute:
            raise DataUnavailable(f"synthetic: no minute bars for {day.date}")
        return self.minute[day.date]

    def daily_bars(self, start, end):
        idx = self.daily.index
        return self.daily.loc[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]

    def option_strikes(self, expiry, right, near_price):
        return list(self.strikes), "synthetic"

    def option_minute_bars(self, symbol, day):
        self.option_requests.append(symbol)
        try:
            return self.options[symbol][day.date]
        except KeyError:
            raise DataUnavailable(f"synthetic: no bars for {symbol} on {day.date}") from None


def build_scenario(
    target_closes,
    *,
    target_date: dt.date = dt.date(2026, 11, 20),
    n_prior: int = 25,
    base: float = 100.0,
    prior_move: float = 0.10,
    daily_range: float = 1.0,
    build_options: bool = True,
    extrinsic: float = 0.5,
):
    """Prior days move exactly `prior_move` from the open every minute (sigma = prior_move);
    daily bars have high-low = `daily_range` and an unchanged close (ATR = daily_range)."""
    days = weekdays_ending(target_date, n_prior + 1)
    target = days[-1]
    minute = {d.date: bars_from_closes(d, np.full(390, base + prior_move), first_open=base) for d in days[:-1]}
    minute[target.date] = bars_from_closes(target, target_closes, first_open=base)
    daily = pd.DataFrame(
        {"open": base, "high": base + daily_range / 2, "low": base - daily_range / 2, "close": base, "volume": 1e6},
        index=pd.DatetimeIndex([pd.Timestamp(d.date) for d in days], name="date"),
    )
    lo, hi = int(np.floor(np.min(target_closes))) - 3, int(np.ceil(np.max(target_closes))) + 3
    strikes = [float(k) for k in range(lo - 10, hi + 11)]
    options: dict[str, dict] = {}
    if build_options:
        und = minute[target.date]
        for k in range(lo, hi + 1):
            for right in ("call", "put"):
                sym = occ_symbol("QQQ", target.date, right, float(k))
                options[sym] = {target.date: option_bars_from_underlying(und, float(k), right, extrinsic)}
    return SyntheticProvider(days, minute, daily, options, strikes), target
