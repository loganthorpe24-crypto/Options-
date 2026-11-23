"""Validation against the creator's Nov 20, 2026 example.

Creator's trade (from the video): at 10:30am buy 10x $632 QQQ puts expiring the
same day at $11.26 ($11,260), sell at 3:45pm at $25.76 ($25,760) for a $14,500
profit (+128.8%). His caption's shorthand "$11,260 in, $14,500 out" is read as
proceeds $25,760 / profit $14,500, per the video.

This module NEVER tunes anything to match. It runs the normal backtest with your
config, then reports what matched, what didn't, and which input data explain
the gap.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass

import pandas as pd

from .backtest import Backtester, DayResult, option_fill
from .config import Config
from .data import DataUnavailable, MarketDataProvider
from .indicators import price_at
from .options import occ_symbol
from .timeutil import et_timestamp, hhmm


@dataclass(frozen=True)
class CreatorExample:
    date: dt.date = dt.date(2026, 11, 20)
    side: str = "put"
    signal_time: str = "10:30"
    strike: float = 632.0
    contracts: int = 10
    entry_price: float = 11.26
    exit_time: str = "15:45"
    exit_price: float = 25.76
    cost: float = 11_260.0
    proceeds: float = 25_760.0
    profit: float = 14_500.0
    return_pct: float = 128.8


CREATOR_EXAMPLE = CreatorExample()

EXIT_OK, EXIT_MISMATCH, EXIT_DATA = 0, 1, 3


def _fmt(x, nd=2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def _checks_table(day: DayResult) -> list[str]:
    lines = [
        "  time   close     putTrig   lowerBand  VWAP      | P1 P2 P3 | callTrig  upperBand | C1 C2 C3 | signal",
    ]
    for c in day.checks:
        tf = lambda b: "T " if b else "F "  # noqa: E731
        lines.append(
            f"  {hhmm(c.check_time)}  {c.close:8.2f}  {c.put_trigger:8.2f}  {c.lower_band:9.2f}  {c.vwap:8.2f}  | "
            f"{tf(c.put_below_or_atr)} {tf(c.put_below_lower_band)} {tf(c.put_below_vwap)}| "
            f"{c.call_trigger:8.2f}  {c.upper_band:9.2f} | "
            f"{tf(c.call_above_or_atr)} {tf(c.call_above_upper_band)} {tf(c.call_above_vwap)}| {c.signal or ''}"
        )
    lines.append("  P1: close < OR low - 0.3*ATR   P2: close < lower band   P3: close < VWAP   (C1-C3 = call mirror)")
    return lines


def _explain_check(c) -> list[str]:
    """Distance of each put condition from flipping at one check."""
    return [
        f"    P1 close - (ORlow - 0.3*ATR) = {c.close - c.put_trigger:+.3f}  ({'pass' if c.put_below_or_atr else 'FAIL'})",
        f"    P2 close - lower band        = {c.close - c.lower_band:+.3f}  ({'pass' if c.put_below_lower_band else 'FAIL'})",
        f"    P3 close - VWAP              = {c.close - c.vwap:+.3f}  ({'pass' if c.put_below_vwap else 'FAIL'})",
    ]


def _creator_contract_diagnostic(cfg: Config, provider: MarketDataProvider, ex: CreatorExample) -> list[str]:
    lines = ["Creator's contract, priced from Alpaca data (diagnostic only):"]
    rules = cfg.strategy
    sym = occ_symbol(rules.underlying, ex.date, ex.side, ex.strike)
    try:
        day = provider.trading_days(ex.date, ex.date)[0]
        und = provider.minute_bars(day)
        t_in = et_timestamp(ex.date, ex.signal_time)
        t_out = et_timestamp(ex.date, ex.exit_time)
        q_in, q_out = price_at(und, t_in), price_at(und, t_out)
        lines.append(f"  QQQ at {ex.signal_time}: {q_in:.2f}   QQQ at {ex.exit_time}: {q_out:.2f}")
        intrinsic = (lambda q: max(ex.strike - q, 0.0)) if ex.side == "put" else (lambda q: max(q - ex.strike, 0.0))
        lines.append(
            f"  {sym} intrinsic: {intrinsic(q_in):.2f} at {ex.signal_time}, {intrinsic(q_out):.2f} at {ex.exit_time}"
        )
        opt = provider.option_minute_bars(sym, day)
        for model in ("next_bar_open", "signal_bar_close"):
            try:
                pin, bin_ = option_fill(opt, t_in, model, cfg.assumptions.option_fill_max_staleness_minutes, symbol=sym, label="entry")
                pout, bout = option_fill(opt, t_out, model, cfg.assumptions.option_fill_max_staleness_minutes, symbol=sym, label="exit")
                pnl = (pout - pin) * 100 * ex.contracts
                lines.append(
                    f"  fill_model={model:16s}: entry {pin:.2f} (bar {hhmm(bin_)}) vs creator {ex.entry_price:.2f}; "
                    f"exit {pout:.2f} (bar {hhmm(bout)}) vs creator {ex.exit_price:.2f}; "
                    f"{ex.contracts}x P&L {pnl:+,.2f} vs creator {ex.profit:+,.2f}"
                )
            except DataUnavailable as exc:
                lines.append(f"  fill_model={model}: {exc}")
    except (DataUnavailable, IndexError) as exc:
        lines.append(f"  unavailable: {exc}")
    return lines


def _sensitivity(cfg: Config, provider: MarketDataProvider, date: dt.date) -> list[str]:
    """Re-run the day under each alternative ASSUMPTION value. Diagnostic only; defaults are not changed."""
    variants = {
        "noise_move_units=percent": {"noise_move_units": "percent"},
        "vwap_price=bar_vwap": {"vwap_price": "bar_vwap"},
        "vwap_price=close": {"vwap_price": "close"},
        "atr_source=rth_from_minutes": {"atr_source": "rth_from_minutes"},
        "fill_model=signal_bar_close": {"fill_model": "signal_bar_close"},
    }
    lines = ["Sensitivity to the ASSUMPTIONS (diagnostic only, your config is NOT changed):"]
    for name, change in variants.items():
        if all(getattr(cfg.assumptions, k) == v for k, v in change.items()):
            continue
        alt = dataclasses.replace(cfg, assumptions=dataclasses.replace(cfg.assumptions, **change))
        try:
            d = Backtester(alt, provider).run(date, date).days[0]
        except DataUnavailable as exc:
            lines.append(f"  {name:32s} -> data unavailable: {exc}")
            continue
        if d.trade:
            t = d.trade
            lines.append(
                f"  {name:32s} -> {t.side.upper()} at {t.signal_time}, {t.option_symbol}, "
                f"{t.entry_price:.2f} -> {t.exit_price:.2f}, P&L {t.pnl:+,.2f}"
            )
        else:
            lines.append(f"  {name:32s} -> {d.status}{': ' + d.reason if d.reason else ''}")
    return lines


def run_validation(
    cfg: Config, provider: MarketDataProvider, *, example: CreatorExample = CREATOR_EXAMPLE, sensitivity: bool = True
) -> tuple[str, int, DayResult]:
    ex = example
    bt = Backtester(cfg, provider)
    result = bt.run(ex.date, ex.date)
    day = result.days[0]
    out: list[str] = [
        "=" * 100,
        f"VALIDATION vs creator example, {ex.date:%a %b %d, %Y}",
        f"  Creator: {ex.side.upper()} at {ex.signal_time}, {ex.contracts}x ${ex.strike:g} 0DTE, entry ${ex.entry_price:.2f} "
        f"(${ex.cost:,.0f}), exit {ex.exit_time} at ${ex.exit_price:.2f} (${ex.proceeds:,.0f}), "
        f"profit ${ex.profit:,.0f} (+{ex.return_pct}%)",
        "=" * 100,
    ]
    if day.setup is not None:
        out.append("Setup: " + day.setup.describe(cfg.strategy))
    if day.checks:
        out.append("10-minute checks (stops at the first signal):")
        out.extend(_checks_table(day))

    code = EXIT_OK
    if day.status == "traded" and day.trade:
        t = day.trade
        rows = [
            ("side", ex.side, t.side),
            ("signal time", ex.signal_time, t.signal_time),
            ("strike", ex.strike, t.strike),
            ("contracts", ex.contracts, t.contracts),
            ("entry price", ex.entry_price, t.entry_price),
            ("exit time", ex.exit_time, t.exit_time),
            ("exit price", ex.exit_price, t.exit_price),
            ("cost", ex.cost, t.cost),
            ("proceeds", ex.proceeds, t.proceeds),
            ("profit", ex.profit, t.pnl),
            ("return %", ex.return_pct, t.pnl_pct),
        ]
        out.append("Comparison (creator vs this backtest):")
        for name, a, b in rows:
            same = (abs(a - b) < 0.005) if isinstance(a, float) else (a == b)
            out.append(f"  {name:12s} {_fmt(a):>12s}  {_fmt(b):>12s}  {'MATCH' if same else 'differs'}")
        out.append(f"  exit reason: {t.exit_reason}; strike source: {t.strike_source}; QQQ at signal {t.underlying_at_signal}")
        out.append(
            f"  trailing stop: basis={t.trail_basis} peak={t.trail_peak} at {t.trail_peak_time} "
            f"armed={t.trail_armed} {t.trail_armed_time} retracement={t.trail_retracement_from_peak}"
        )
        if t.side != ex.side or t.signal_time != ex.signal_time:
            code = EXIT_MISMATCH
    else:
        code = EXIT_DATA if day.status == "data_error" else EXIT_MISMATCH
        out.append(f"RESULT: {day.status.upper()}{' - ' + day.reason if day.reason else ''}")

    signal_ok = day.signal_side == ex.side and day.signal_time == ex.signal_time
    verdict = "PASS" if signal_ok else "FAIL"
    out.append(
        f"SIGNAL CHECK: {verdict} (expected {ex.side.upper()} at {ex.signal_time}; "
        f"got {(day.signal_side or 'none').upper()} at {day.signal_time or '-'})"
    )
    if not signal_ok and day.checks:
        target = next((c for c in day.checks if hhmm(c.check_time) == ex.signal_time), None)
        if target is not None:
            out.append(f"Why the {ex.signal_time} check did not give a {ex.side.upper()}:")
            out.extend(_explain_check(target))
        elif day.signal_time:
            out.append(f"  The backtest's first signal came at {day.signal_time}, before {ex.signal_time} (first signal only).")
    if day.status == "traded" and day.trade and day.trade.strike != ex.strike:
        out.append(
            f"NOTE: STRIKE_SELECTION={cfg.assumptions.strike_selection!r} picked ${day.trade.strike:g} "
            f"(QQQ {day.trade.underlying_at_signal:.2f} at the signal). The creator bought ${ex.strike:g}. "
            "The creator never states his strike rule, so this gap is expected and is NOT tuned away."
        )
    out.extend(_creator_contract_diagnostic(cfg, provider, ex))
    if sensitivity:
        out.extend(_sensitivity(cfg, provider, ex.date))
    if not signal_ok:
        out.extend(
            [
                "Input data that can differ from what the creator saw:",
                f"  - stock feed: data.stock_feed={cfg.data.stock_feed!r} (SIP = all exchanges; IEX is about 2-3% of volume, "
                "so VWAP and the opening range differ)",
                f"  - daily bars for the ATR: atr_source={cfg.assumptions.atr_source!r}",
                f"  - noise-band units: noise_move_units={cfg.assumptions.noise_move_units!r}; "
                f"VWAP price: vwap_price={cfg.assumptions.vwap_price!r}",
                "  - his charting platform's bars (time-stamp convention, extended-hours inclusion in VWAP/ATR)",
                "  - whether he checked 10:30 on the 10:20-10:30 bar close (this code) or on a live tick",
            ]
        )
    out.append("=" * 100)
    return "\n".join(out), code, day
