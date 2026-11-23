"""Signal engine: the three entry conditions, evaluated on every 10-minute close.

The backtester and the paper/live runner both use these exact functions.

BUY PUTS  if, on the check close: close < OR low  - 0.3*ATR  AND close < lower band AND close < VWAP
BUY CALLS if, on the check close: close > OR high + 0.3*ATR  AND close > upper band AND close > VWAP
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .config import Assumptions, StrategyRules
from .data import DataUnavailable, TradingDay
from .indicators import (
    completed_bars,
    compute_atr,
    daily_ohlc_from_minutes,
    minute_move_profile,
    minute_offset,
    noise_band,
    noise_sigma,
    opening_range,
    session_open_price,
    vwap_at,
)
from .timeutil import et_timestamp, hhmm


@dataclass(frozen=True)
class DaySetup:
    """Everything fixed for the day once the opening range is complete."""

    day: TradingDay
    open_ts: pd.Timestamp
    open_price: float
    or_high: float
    or_low: float
    or_bars: int
    atr: float
    noise_sigma: np.ndarray = field(repr=False)
    noise_days: tuple[dt.date, ...] = ()

    def describe(self, rules: StrategyRules) -> str:
        m = rules.atr_breakout_multiplier
        return (
            f"{self.day.date} open={self.open_price:.2f} OR[{self.or_low:.2f}, {self.or_high:.2f}] "
            f"({self.or_bars} bars) ATR{rules.atr_period_days}={self.atr:.3f} -> put trigger "
            f"{self.or_low - m * self.atr:.2f}, call trigger {self.or_high + m * self.atr:.2f}; "
            f"noise lookback {self.noise_days[0] if self.noise_days else '?'}..{self.noise_days[-1] if self.noise_days else '?'}"
        )


@dataclass(frozen=True)
class CheckResult:
    check_time: pd.Timestamp
    last_bar: pd.Timestamp
    close: float
    vwap: float
    minute_offset: int
    noise_avg_move: float
    upper_band: float
    lower_band: float
    or_high: float
    or_low: float
    atr: float
    put_trigger: float
    call_trigger: float
    put_below_or_atr: bool
    put_below_lower_band: bool
    put_below_vwap: bool
    call_above_or_atr: bool
    call_above_upper_band: bool
    call_above_vwap: bool

    @property
    def put_signal(self) -> bool:
        return self.put_below_or_atr and self.put_below_lower_band and self.put_below_vwap

    @property
    def call_signal(self) -> bool:
        return self.call_above_or_atr and self.call_above_upper_band and self.call_above_vwap

    @property
    def signal(self) -> str | None:
        if self.put_signal:
            return "put"
        if self.call_signal:
            return "call"
        return None

    def as_row(self) -> dict:
        return {
            "date": self.check_time.date().isoformat(),
            "check_time": hhmm(self.check_time),
            "last_bar": hhmm(self.last_bar),
            "close": round(self.close, 4),
            "vwap": round(self.vwap, 4),
            "or_high": round(self.or_high, 4),
            "or_low": round(self.or_low, 4),
            "atr": round(self.atr, 4),
            "put_trigger": round(self.put_trigger, 4),
            "call_trigger": round(self.call_trigger, 4),
            "minute_offset": self.minute_offset,
            "noise_avg_move": round(self.noise_avg_move, 6),
            "upper_band": round(self.upper_band, 4),
            "lower_band": round(self.lower_band, 4),
            "put_c1_close_lt_orlow_minus_atr": self.put_below_or_atr,
            "put_c2_close_lt_lower_band": self.put_below_lower_band,
            "put_c3_close_lt_vwap": self.put_below_vwap,
            "call_c1_close_gt_orhigh_plus_atr": self.call_above_or_atr,
            "call_c2_close_gt_upper_band": self.call_above_upper_band,
            "call_c3_close_gt_vwap": self.call_above_vwap,
            "signal": self.signal or "",
        }

    def log_line(self) -> str:
        def tf(x: bool) -> str:
            return "T" if x else "F"

        return (
            f"CHECK {self.check_time:%Y-%m-%d} {hhmm(self.check_time)} close={self.close:.2f} | "
            f"PUT: close<ORlow-0.3ATR({self.put_trigger:.2f})={tf(self.put_below_or_atr)} "
            f"close<lowerBand({self.lower_band:.2f})={tf(self.put_below_lower_band)} "
            f"close<VWAP({self.vwap:.2f})={tf(self.put_below_vwap)} | "
            f"CALL: close>ORhigh+0.3ATR({self.call_trigger:.2f})={tf(self.call_above_or_atr)} "
            f"close>upperBand({self.upper_band:.2f})={tf(self.call_above_upper_band)} "
            f"close>VWAP={tf(self.call_above_vwap)} | signal={self.signal.upper() if self.signal else 'none'}"
        )


# --------------------------------------------------------------------------- #
# Day setup
# --------------------------------------------------------------------------- #
def noise_sigma_from_sessions(
    sessions: list[tuple[TradingDay, pd.DataFrame]], assumptions: Assumptions
) -> np.ndarray:
    profiles = []
    for day, bars in sessions:
        open_price = session_open_price(bars, day.open_ts, assumptions.session_open_price)
        profiles.append(minute_move_profile(bars, day, open_price, assumptions.noise_move_units))
    return noise_sigma(profiles)


def atr_for_day(
    day: TradingDay,
    history: list[TradingDay],
    rules: StrategyRules,
    assumptions: Assumptions,
    *,
    daily_bars: pd.DataFrame | None = None,
    minute_bars_fn: Callable[[TradingDay], pd.DataFrame] | None = None,
) -> float:
    """5-day ATR for `day`, using the trading days immediately before it (strictly verified)."""
    period = rules.atr_period_days
    needed = [d for d in history if d.date < day.date][-(period + 1):]
    if len(needed) < period + 1:
        raise DataUnavailable(f"ATR needs {period + 1} trading days before {day.date}; calendar has {len(needed)}")
    want = [pd.Timestamp(d.date) for d in needed]
    if assumptions.atr_source == "alpaca_daily":
        if daily_bars is None:
            raise ValueError("atr_source=alpaca_daily requires daily_bars")
        missing = [w.date().isoformat() for w in want if w not in daily_bars.index]
        if missing:
            raise DataUnavailable(
                f"Alpaca daily bars missing for {', '.join(missing)} (needed for the {period}-day ATR on {day.date})"
            )
        frame = daily_bars.loc[want]
    elif assumptions.atr_source == "rth_from_minutes":
        if minute_bars_fn is None:
            raise ValueError("atr_source=rth_from_minutes requires minute_bars_fn")
        frame = pd.DataFrame(
            [daily_ohlc_from_minutes(minute_bars_fn(d)) for d in needed],
            index=pd.DatetimeIndex(want, name="date"),
        )
    else:
        raise ValueError(f"Unknown atr_source {assumptions.atr_source!r}")
    return compute_atr(frame, day.date, period)


def build_day_setup(
    day: TradingDay,
    today_bars: pd.DataFrame,
    prior_sessions: list[tuple[TradingDay, pd.DataFrame]],
    atr: float,
    rules: StrategyRules,
    assumptions: Assumptions,
) -> DaySetup:
    if len(prior_sessions) != rules.noise_lookback_days:
        raise DataUnavailable(
            f"noise band needs exactly {rules.noise_lookback_days} prior sessions, got {len(prior_sessions)}"
        )
    if any(d.date >= day.date for d, _ in prior_sessions):
        raise ValueError("noise-band lookback must only contain sessions before the trade date")
    open_ts = et_timestamp(day.date, rules.session_open_time)
    open_price = session_open_price(today_bars, open_ts, assumptions.session_open_price)
    or_high, or_low, or_n = opening_range(today_bars, open_ts, rules.opening_range_minutes)
    sigma = noise_sigma_from_sessions(prior_sessions, assumptions)
    if not (math.isfinite(atr) and atr > 0):
        raise DataUnavailable(f"ATR is {atr}")
    return DaySetup(
        day=day,
        open_ts=open_ts,
        open_price=open_price,
        or_high=or_high,
        or_low=or_low,
        or_bars=or_n,
        atr=atr,
        noise_sigma=sigma,
        noise_days=tuple(d.date for d, _ in prior_sessions),
    )


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_times(day: TradingDay, rules: StrategyRules, assumptions: Assumptions) -> list[pd.Timestamp]:
    start = et_timestamp(day.date, rules.entry_window_start)
    end = et_timestamp(day.date, rules.entry_window_end)
    step = pd.Timedelta(minutes=rules.check_interval_minutes)
    out = []
    t = start
    while t <= end:
        if t == end and not assumptions.entry_window_end_inclusive:
            break
        out.append(t)
        t += step
    return [x for x in out if x <= day.close_ts]


def evaluate_check(
    setup: DaySetup,
    today_bars: pd.DataFrame,
    t: pd.Timestamp,
    rules: StrategyRules,
    assumptions: Assumptions,
) -> CheckResult:
    """Evaluate both sides' three conditions using only bars completed by time t."""
    done = completed_bars(today_bars, t)
    done = done.loc[done.index >= setup.open_ts]
    if done.empty:
        raise DataUnavailable(f"no completed bars at the {hhmm(t)} check")
    close = float(done["close"].iloc[-1])
    vwap = vwap_at(done, t, assumptions.vwap_price, session_open=setup.open_ts)
    k = minute_offset(setup.open_ts, t)
    if not 1 <= k <= len(setup.noise_sigma):
        raise ValueError(f"check at {hhmm(t)} is outside the session")
    sigma = float(setup.noise_sigma[k - 1])
    if not math.isfinite(sigma):
        raise DataUnavailable(f"noise band undefined at minute {k} ({hhmm(t)}): no lookback session has data there")
    upper, lower = noise_band(setup.open_price, sigma, rules.noise_multiplier, assumptions.noise_move_units)
    m = rules.atr_breakout_multiplier
    put_trigger = setup.or_low - m * setup.atr
    call_trigger = setup.or_high + m * setup.atr
    return CheckResult(
        check_time=t,
        last_bar=done.index[-1],
        close=close,
        vwap=vwap,
        minute_offset=k,
        noise_avg_move=sigma,
        upper_band=upper,
        lower_band=lower,
        or_high=setup.or_high,
        or_low=setup.or_low,
        atr=setup.atr,
        put_trigger=put_trigger,
        call_trigger=call_trigger,
        put_below_or_atr=close < put_trigger,
        put_below_lower_band=close < lower,
        put_below_vwap=close < vwap,
        call_above_or_atr=close > call_trigger,
        call_above_upper_band=close > upper,
        call_above_vwap=close > vwap,
    )
