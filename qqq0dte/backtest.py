"""Backtester: historical 1-minute QQQ bars + 0DTE option trade bars from Alpaca.

Per day:
  1. skip half days (early_close_policy=skip)
  2. build the setup: open, opening range, 5-day ATR, 20-day noise profile
  3. evaluate checks 10:00, 10:10, ..., 14:00; take the FIRST signal only
  4. buy the 0DTE contract chosen by STRIKE_SELECTION, filled from real option trade bars
  5. manage: trailing stop on 1-minute closes; hard exit at 15:45
Missing data never gets filled with invented prices. The day is recorded as a
data_error with the exact reason.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .data import DataUnavailable, MarketDataProvider, TradingDay
from .exits import TrailingStop
from .indicators import price_at
from .options import occ_symbol, select_strike
from .signals import CheckResult, DaySetup, atr_for_day, build_day_setup, check_times, evaluate_check
from .timeutil import ONE_MINUTE, et_timestamp, hhmm

log = logging.getLogger(__name__)


@dataclass
class Trade:
    date: str
    side: str
    signal_time: str
    underlying_at_signal: float
    strike: float
    option_symbol: str
    strike_source: str
    contracts: int
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str
    cost: float
    proceeds: float
    fees: float
    pnl: float
    pnl_pct: float
    underlying_at_exit: float
    atr: float
    or_high: float
    or_low: float
    trigger_level: float
    band_at_signal: float
    vwap_at_signal: float
    fill_model: str
    entry_fill_bar: str
    exit_fill_bar: str
    trail_basis: str
    trail_arm_threshold: float
    trail_peak: float
    trail_peak_time: str
    trail_peak_underlying: float | None
    trail_armed: bool
    trail_armed_time: str
    trail_armed_value: float | None
    trail_armed_underlying: float | None
    trail_exit_value: float | None
    trail_retracement_from_peak: float | None

    def as_row(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class DayResult:
    date: dt.date
    status: str  # traded | no_signal | skipped | data_error
    reason: str = ""
    signal_side: str | None = None
    signal_time: str | None = None
    trade: Trade | None = None
    checks: list[CheckResult] = field(default_factory=list)
    setup: DaySetup | None = None


@dataclass
class BacktestResult:
    start: dt.date
    end: dt.date
    days: list[DayResult]

    @property
    def trades(self) -> list[Trade]:
        return [d.trade for d in self.days if d.trade is not None]


def option_fill(
    opt_bars: pd.DataFrame, t: pd.Timestamp, model: str, staleness_minutes: int, *, symbol: str, label: str
) -> tuple[float, pd.Timestamp]:
    """Fill price for an option order decided at time t, from real 1-minute trade bars."""
    stale = pd.Timedelta(minutes=staleness_minutes)
    if model == "next_bar_open":
        w = opt_bars.loc[(opt_bars.index >= t) & (opt_bars.index < t + stale)]
        if w.empty:
            raise DataUnavailable(
                f"{label}: no {symbol} trade bars in [{hhmm(t)}, {hhmm(t + stale)}) ET "
                f"(fill_model=next_bar_open, option_fill_max_staleness_minutes={staleness_minutes})"
            )
        return float(w["open"].iloc[0]), w.index[0]
    if model == "signal_bar_close":
        w = opt_bars.loc[(opt_bars.index <= t - ONE_MINUTE) & (opt_bars.index > t - ONE_MINUTE - stale)]
        if w.empty:
            raise DataUnavailable(
                f"{label}: no {symbol} trade bar ending in ({hhmm(t - stale)}, {hhmm(t)}] ET "
                f"(fill_model=signal_bar_close, option_fill_max_staleness_minutes={staleness_minutes})"
            )
        return float(w["close"].iloc[-1]), w.index[-1]
    raise ValueError(f"Unknown fill_model {model!r}")


def last_trade_price(opt_bars: pd.DataFrame, t: pd.Timestamp) -> float:
    done = opt_bars.loc[opt_bars.index <= t - ONE_MINUTE]
    return float(done["close"].iloc[-1]) if not done.empty else math.nan


class Backtester:
    def __init__(self, cfg: Config, provider: MarketDataProvider) -> None:
        self.cfg = cfg
        self.provider = provider
        self._bars_cache: dict[dt.date, pd.DataFrame] = {}

    # ------------------------------------------------------------------ run
    def run(self, start: dt.date, end: dt.date) -> BacktestResult:
        rules, a = self.cfg.strategy, self.cfg.assumptions
        if end < start:
            raise ValueError("end date is before start date")
        pad_days = max(rules.noise_lookback_days, rules.atr_period_days + 1)
        calendar = self.provider.trading_days(start - dt.timedelta(days=pad_days * 2 + 21), end)
        targets = [d for d in calendar if start <= d.date <= end]
        if not targets:
            raise DataUnavailable(f"No trading days between {start} and {end} in the Alpaca calendar")
        first_idx = calendar.index(targets[0])
        if first_idx < pad_days:
            raise DataUnavailable(
                f"Calendar has only {first_idx} trading days before {targets[0].date}; need {pad_days} for lookbacks"
            )
        daily = None
        if a.atr_source == "alpaca_daily":
            daily = self.provider.daily_bars(calendar[first_idx - pad_days].date, end)

        days: list[DayResult] = []
        for day in targets:
            history = calendar[: calendar.index(day)]
            result = self.run_day(day, history, daily)
            days.append(result)
            log.info(self._day_line(result))
        return BacktestResult(start=start, end=end, days=days)

    @staticmethod
    def _day_line(r: DayResult) -> str:
        if r.status == "traded" and r.trade:
            t = r.trade
            return (
                f"{t.date} {t.side.upper()} signal {t.signal_time} | {t.contracts}x {t.option_symbol} "
                f"entry {t.entry_price:.2f} @ {t.entry_time} -> exit {t.exit_price:.2f} @ {t.exit_time} "
                f"({t.exit_reason}) P&L {t.pnl:+,.2f} ({t.pnl_pct:+.1f}%)"
            )
        if r.status == "no_signal":
            return f"{r.date} no signal ({len(r.checks)} checks)"
        level = "SKIPPED" if r.status == "skipped" else "DATA ERROR"
        return f"{r.date} {level}: {r.reason}"

    # ------------------------------------------------------------- per day
    def _session_bars(self, day: TradingDay) -> pd.DataFrame:
        if day.date not in self._bars_cache:
            self._bars_cache[day.date] = self.provider.minute_bars(day)
        return self._bars_cache[day.date]

    def build_setup(
        self, day: TradingDay, history: list[TradingDay], daily: pd.DataFrame | None
    ) -> tuple[DaySetup, pd.DataFrame]:
        rules, a = self.cfg.strategy, self.cfg.assumptions
        lookback = [d for d in history if d.date < day.date][-rules.noise_lookback_days:]
        if len(lookback) < rules.noise_lookback_days:
            raise DataUnavailable(
                f"need {rules.noise_lookback_days} prior trading days for the noise band, have {len(lookback)}"
            )
        sessions = [(d, self._session_bars(d)) for d in lookback]
        atr = atr_for_day(day, history, rules, a, daily_bars=daily, minute_bars_fn=self._session_bars)
        today = self._session_bars(day)
        return build_day_setup(day, today, sessions, atr, rules, a), today

    def run_day(self, day: TradingDay, history: list[TradingDay], daily: pd.DataFrame | None) -> DayResult:
        rules, a = self.cfg.strategy, self.cfg.assumptions
        if day.is_early_close and a.early_close_policy == "skip":
            return DayResult(
                day.date, "skipped",
                f"early close at {day.close_time:%H:%M} ET; early_close_policy=skip (3:45 exit impossible)",
            )
        try:
            setup, bars = self.build_setup(day, history, daily)
        except DataUnavailable as exc:
            return DayResult(day.date, "data_error", f"setup: {exc}")

        checks: list[CheckResult] = []
        signal: CheckResult | None = None
        for t in check_times(day, rules, a):
            try:
                res = evaluate_check(setup, bars, t, rules, a)
            except DataUnavailable as exc:
                return DayResult(day.date, "data_error", f"check {hhmm(t)}: {exc}", checks=checks, setup=setup)
            checks.append(res)
            if res.signal:  # FIRST signal only
                signal = res
                break
        if signal is None:
            return DayResult(day.date, "no_signal", checks=checks, setup=setup)
        try:
            trade = self.simulate_trade(day, setup, bars, signal)
        except DataUnavailable as exc:
            return DayResult(
                day.date, "data_error",
                f"{signal.signal.upper()} signal at {hhmm(signal.check_time)} but trade could not be priced: {exc}",
                signal_side=signal.signal, signal_time=hhmm(signal.check_time), checks=checks, setup=setup,
            )
        return DayResult(
            day.date, "traded", signal_side=signal.signal, signal_time=hhmm(signal.check_time),
            trade=trade, checks=checks, setup=setup,
        )

    # --------------------------------------------------------------- trade
    def simulate_trade(self, day: TradingDay, setup: DaySetup, bars: pd.DataFrame, signal: CheckResult) -> Trade:
        rules, a = self.cfg.strategy, self.cfg.assumptions
        side = signal.signal
        assert side in ("call", "put")
        strikes, strike_source = self.provider.option_strikes(day.date, side, signal.close)
        strike = select_strike(
            signal.close, side, strikes, method=a.strike_selection, offset=a.strike_offset, tie_break=a.atm_tie_break
        )
        symbol = occ_symbol(rules.underlying, day.date, side, strike)
        opt = self.provider.option_minute_bars(symbol, day)

        t_entry = signal.check_time
        entry_px, entry_bar = option_fill(
            opt, t_entry, a.fill_model, a.option_fill_max_staleness_minutes, symbol=symbol, label="entry"
        )
        hard_exit = et_timestamp(day.date, rules.hard_exit_time)
        stop = TrailingStop(
            side=side,
            entry_time=t_entry,
            entry_underlying=signal.close,
            atr=setup.atr,
            entry_option_price=entry_px,
            arm_threshold=rules.trail_arm_threshold,
            giveback_fraction=rules.trail_giveback_fraction,
        )
        step = pd.Timedelta(minutes=a.trail_check_interval_minutes)
        exit_reason, t_exit = f"hard_exit_{rules.hard_exit_time}", hard_exit
        t = t_entry + step
        while t < hard_exit:
            mark = last_trade_price(opt, t) if stop.needs_option_price else None
            if stop.update(t, underlying=price_at(bars, t), option_price=mark):
                exit_reason, t_exit = "trailing_stop", t
                break
            t += step
        underlying_exit = price_at(bars, t_exit)
        if exit_reason != "trailing_stop":
            mark = last_trade_price(opt, t_exit) if stop.needs_option_price else None
            stop.mark_exit(t_exit, underlying=underlying_exit, option_price=mark)
        exit_px, exit_bar = option_fill(
            opt, t_exit, a.fill_model, a.option_fill_max_staleness_minutes, symbol=symbol, label="exit"
        )

        qty = a.contracts
        entry_eff = entry_px + a.slippage_per_share
        exit_eff = max(exit_px - a.slippage_per_share, 0.0)
        cost = entry_eff * 100 * qty
        proceeds = exit_eff * 100 * qty
        fees = a.commission_per_contract * qty * 2
        pnl = proceeds - cost - fees
        band = signal.lower_band if side == "put" else signal.upper_band
        trigger = signal.put_trigger if side == "put" else signal.call_trigger
        trade = Trade(
            date=day.date.isoformat(),
            side=side,
            signal_time=hhmm(t_entry),
            underlying_at_signal=round(signal.close, 4),
            strike=strike,
            option_symbol=symbol,
            strike_source=strike_source,
            contracts=qty,
            entry_time=hhmm(t_entry),
            entry_price=round(entry_px, 4),
            exit_time=hhmm(t_exit),
            exit_price=round(exit_px, 4),
            exit_reason=exit_reason,
            cost=round(cost, 2),
            proceeds=round(proceeds, 2),
            fees=round(fees, 2),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl / cost * 100, 2) if cost else math.nan,
            underlying_at_exit=round(underlying_exit, 4),
            atr=round(setup.atr, 4),
            or_high=round(setup.or_high, 4),
            or_low=round(setup.or_low, 4),
            trigger_level=round(trigger, 4),
            band_at_signal=round(band, 4),
            vwap_at_signal=round(signal.vwap, 4),
            fill_model=a.fill_model,
            entry_fill_bar=hhmm(entry_bar),
            exit_fill_bar=hhmm(exit_bar),
            trail_basis=stop.basis,
            trail_arm_threshold=rules.trail_arm_threshold,
            trail_peak=round(stop.peak, 4),
            trail_peak_time=hhmm(stop.peak_time),
            trail_peak_underlying=None if stop.peak_underlying is None else round(stop.peak_underlying, 4),
            trail_armed=stop.armed,
            trail_armed_time=hhmm(stop.armed_time) if stop.armed_time is not None else "",
            trail_armed_value=None if stop.armed_value is None else round(stop.armed_value, 4),
            trail_armed_underlying=None if stop.armed_underlying is None else round(stop.armed_underlying, 4),
            trail_exit_value=None if stop.exit_value is None else round(stop.exit_value, 4),
            trail_retracement_from_peak=None
            if stop.retracement_from_peak is None
            else round(stop.retracement_from_peak, 4),
        )
        log.info("%s trailing stop: %s", day.date, stop.describe())
        return trade
