"""Indicators: opening range, noise band, session VWAP, 5-day ATR.

Convention: 1-minute bars are indexed by their START time. The bar stamped 10:29
covers 10:29:00-10:29:59, and its close is the price "at 10:30".
"""

from __future__ import annotations

import datetime as dt
import logging
import warnings

import numpy as np
import pandas as pd

from .data import DataUnavailable, TradingDay
from .timeutil import FULL_SESSION_MINUTES, ONE_MINUTE, hhmm

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Basic price access (lookahead-safe)
# --------------------------------------------------------------------------- #
def completed_bars(bars: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Bars that had fully closed by time t (bar start <= t - 1 minute)."""
    return bars.loc[bars.index <= t - ONE_MINUTE]


def check_bar_gap(last_bar_start: pd.Timestamp, t: pd.Timestamp, max_gap_minutes: int | None) -> None:
    """Fail loudly if the newest bar before t is older than max_gap_minutes (missing data, not a quiet tape)."""
    if max_gap_minutes is None:
        return
    age = (t - (last_bar_start + ONE_MINUTE)) / ONE_MINUTE
    if age > max_gap_minutes:
        raise DataUnavailable(
            f"newest 1-minute bar before {hhmm(t)} ended at {hhmm(last_bar_start + ONE_MINUTE)} "
            f"({age:.0f} min old > data.max_bar_gap_minutes={max_gap_minutes}); refusing to use a stale price"
        )


def price_at(bars: pd.DataFrame, t: pd.Timestamp, max_gap_minutes: int | None = None) -> float:
    """Last traded price at time t = close of the last bar that ended at or before t."""
    done = completed_bars(bars, t)
    if done.empty:
        raise DataUnavailable(f"no completed 1-minute bar at or before {hhmm(t)}")
    check_bar_gap(done.index[-1], t, max_gap_minutes)
    return float(done["close"].iloc[-1])


def minute_offset(open_ts: pd.Timestamp, t: pd.Timestamp) -> int:
    """Minutes elapsed since the 9:30 open (10:30 -> 60)."""
    return int(round((t - open_ts) / ONE_MINUTE))


# --------------------------------------------------------------------------- #
# Session open & opening range
# --------------------------------------------------------------------------- #
def session_open_price(bars: pd.DataFrame, open_ts: pd.Timestamp, source: str = "first_bar_open") -> float:
    if source != "first_bar_open":
        raise ValueError(f"Unknown session_open_price source {source!r}")
    if bars.empty or open_ts not in bars.index:
        raise DataUnavailable(f"no 1-minute bar at the {hhmm(open_ts)} open, so the session open price is undefined")
    return float(bars.loc[open_ts, "open"])


def opening_range(bars: pd.DataFrame, open_ts: pd.Timestamp, minutes: int = 15) -> tuple[float, float, int]:
    """(high, low, n_bars) over the 1-minute bars starting in [open, open + minutes)."""
    window = bars.loc[(bars.index >= open_ts) & (bars.index < open_ts + pd.Timedelta(minutes=minutes))]
    if window.empty:
        raise DataUnavailable(f"no 1-minute bars in the opening range starting {hhmm(open_ts)}")
    if len(window) < minutes:
        log.warning(
            "Opening range on %s has %d of %d one-minute bars (minutes with no trades on this feed?)",
            open_ts.date(), len(window), minutes,
        )
    return float(window["high"].max()), float(window["low"].min()), int(len(window))


# --------------------------------------------------------------------------- #
# VWAP
# --------------------------------------------------------------------------- #
def _vwap_price(bars: pd.DataFrame, price: str) -> pd.Series:
    if price == "hlc3":
        return (bars["high"] + bars["low"] + bars["close"]) / 3.0
    if price == "bar_vwap":
        if bars["vwap"].isna().any():
            raise DataUnavailable("vwap_price=bar_vwap but some bars have no 'vw' field from Alpaca")
        return bars["vwap"]
    if price == "close":
        return bars["close"]
    raise ValueError(f"Unknown vwap_price {price!r}")


def vwap_at(bars: pd.DataFrame, t: pd.Timestamp, price: str = "hlc3", session_open: pd.Timestamp | None = None) -> float:
    """Session VWAP from the open through the last bar completed at time t."""
    done = completed_bars(bars, t)
    if session_open is not None:
        done = done.loc[done.index >= session_open]
    if done.empty:
        raise DataUnavailable(f"no completed bars for VWAP at {hhmm(t)}")
    vol = done["volume"].astype(float)
    total = float(vol.sum())
    if total <= 0:
        raise DataUnavailable(f"zero session volume for VWAP at {hhmm(t)}")
    return float((_vwap_price(done, price) * vol).sum() / total)


# --------------------------------------------------------------------------- #
# ATR: THE single ATR function. Change the definition here and only here.
# --------------------------------------------------------------------------- #
def compute_atr(daily_bars: pd.DataFrame, as_of: dt.date, period: int = 5) -> float:
    """N-day Average True Range as known before the session of `as_of`.

    True Range_i = max(high_i - low_i, |high_i - close_{i-1}|, |low_i - close_{i-1}|)
    ATR          = simple (unweighted) mean of the last `period` True Ranges

    Only bars dated strictly before `as_of` are used, so today's incomplete bar is
    never included. Needs period+1 bars (the first TR needs the prior close).
    `daily_bars`: columns high/low/close, index = session date (naive timestamps).
    """
    prior = daily_bars.loc[daily_bars.index < pd.Timestamp(as_of)].sort_index().tail(period + 1)
    if len(prior) < period + 1:
        raise DataUnavailable(
            f"{period}-day ATR for {as_of} needs {period + 1} daily bars before that date, found {len(prior)}"
        )
    high, low, close = prior["high"], prior["low"], prior["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1).iloc[1:]
    atr = float(true_range.mean())
    if not np.isfinite(atr) or atr <= 0:
        raise DataUnavailable(f"ATR for {as_of} came out as {atr}; check the daily bars")
    return atr


def daily_ohlc_from_minutes(bars: pd.DataFrame) -> dict:
    """Regular-session daily OHLC built from 1-minute bars (atr_source=rth_from_minutes)."""
    if bars.empty:
        raise DataUnavailable("no minute bars to build a daily bar from")
    return {
        "open": float(bars["open"].iloc[0]),
        "high": float(bars["high"].max()),
        "low": float(bars["low"].min()),
        "close": float(bars["close"].iloc[-1]),
        "volume": float(bars["volume"].sum()),
    }


# --------------------------------------------------------------------------- #
# Noise band
# --------------------------------------------------------------------------- #
def minute_move_profile(bars: pd.DataFrame, day: TradingDay, open_price: float, units: str = "dollars") -> np.ndarray:
    """Move from the open for every minute of one session.

    Element k-1 (k = 1..390) = |price at (open + k minutes) - open|, where price at T is the
    close of the bar ending at T. Minutes with no trade carry the last close forward.
    Minutes after an early close are NaN. units='percent' gives |price/open - 1|.
    """
    n = day.session_minutes
    grid = pd.date_range(day.open_ts, periods=n, freq="1min")
    closes = bars["close"].reindex(grid).ffill()
    if units == "dollars":
        move = (closes - open_price).abs()
    elif units == "percent":
        move = (closes / open_price - 1.0).abs()
    else:
        raise ValueError(f"Unknown noise_move_units {units!r}")
    out = np.full(FULL_SESSION_MINUTES, np.nan)
    out[: min(n, FULL_SESSION_MINUTES)] = move.to_numpy(dtype=float)[:FULL_SESSION_MINUTES]
    return out


def noise_sigma(profiles: list[np.ndarray]) -> np.ndarray:
    """Mean absolute move per minute offset across the lookback sessions (NaNs ignored)."""
    if not profiles:
        raise DataUnavailable("noise band needs at least one prior session")
    stack = np.vstack(profiles)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stack, axis=0)


def noise_band(open_price: float, sigma_k: float, multiplier: float, units: str = "dollars") -> tuple[float, float]:
    """(upper, lower) = open +/- multiplier * sigma (dollars) or open * (1 +/- multiplier * sigma) (percent)."""
    if units == "dollars":
        width = multiplier * sigma_k
        return open_price + width, open_price - width
    if units == "percent":
        return open_price * (1 + multiplier * sigma_k), open_price * (1 - multiplier * sigma_k)
    raise ValueError(f"Unknown noise_move_units {units!r}")
