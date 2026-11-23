import dataclasses
import datetime as dt

import numpy as np
import pytest

from qqq0dte.backtest import Backtester
from qqq0dte.config import load_config
from qqq0dte.indicators import price_at
from qqq0dte.options import occ_symbol
from qqq0dte.report import summarize, write_outputs
from qqq0dte.signals import check_times, evaluate_check
from qqq0dte.timeutil import et_timestamp, hhmm
from tests.synthetic import (
    build_scenario,
    call_day_closes,
    flat_day_closes,
    put_day_closes,
    trailing_put_day_closes,
)

CFG = load_config(None)
D = dt.date(2026, 11, 20)


def run(provider, cfg=CFG):
    return Backtester(cfg, provider).run(D, D).days[0]


def test_check_times_window_inclusive():
    from qqq0dte.data import TradingDay

    times = [hhmm(t) for t in check_times(TradingDay(D), CFG.strategy, CFG.assumptions)]
    assert times[0] == "10:00" and times[-1] == "14:00" and len(times) == 25
    excl = dataclasses.replace(CFG.assumptions, entry_window_end_inclusive=False)
    assert hhmm(check_times(TradingDay(D), CFG.strategy, excl)[-1]) == "13:50"


def test_put_signal_at_1030_and_hard_exit():
    provider, target = build_scenario(put_day_closes())
    day = run(provider)
    assert day.status == "traded"
    t = day.trade
    assert (t.side, t.signal_time) == ("put", "10:30")
    # checks 10:00, 10:10, 10:20 were evaluated and false, 10:30 fired, no later checks
    assert [hhmm(c.check_time) for c in day.checks] == ["10:00", "10:10", "10:20", "10:30"]
    assert [c.signal for c in day.checks] == [None, None, None, "put"]
    c = day.checks[-1]
    assert c.put_below_or_atr and c.put_below_lower_band and c.put_below_vwap
    # setup: ATR=1, OR low=99.94 -> trigger 99.64; noise avg 0.10 -> lower band 100 - 0.125
    assert c.atr == pytest.approx(1.0)
    assert c.put_trigger == pytest.approx(99.94 - 0.3)
    assert c.lower_band == pytest.approx(99.875)
    # ATM strike for QQQ 99.00 -> 99
    assert t.strike == 99.0 and t.option_symbol == occ_symbol("QQQ", D, "put", 99.0)
    assert t.exit_reason == "hard_exit_15:45" and t.exit_time == "15:45"
    # next_bar_open fills: entry at the 10:30 option bar open, exit at the 15:45 bar open
    und = provider.minute[D]
    s_in = und.loc[et_timestamp(D, "10:30"), "open"]
    s_out = und.loc[et_timestamp(D, "15:45"), "open"]
    assert t.entry_price == pytest.approx(max(99 - s_in, 0) + 0.5, abs=1e-4)
    assert t.exit_price == pytest.approx(max(99 - s_out, 0) + 0.5, abs=1e-4)
    assert t.pnl == pytest.approx((t.exit_price - t.entry_price) * 100 * 10, abs=0.1)
    assert t.contracts == 10
    # trailing stop never armed: QQQ only moved ~0.5 ATR in our favor
    assert t.trail_armed is False and t.trail_peak < 4


def test_call_signal_mirror():
    provider, _ = build_scenario(call_day_closes())
    t = run(provider).trade
    assert (t.side, t.signal_time, t.strike) == ("call", "10:30", 101.0)


def test_no_signal_day():
    provider, _ = build_scenario(flat_day_closes())
    day = run(provider)
    assert day.status == "no_signal" and day.trade is None and len(day.checks) == 25
    assert provider.option_requests == []


def test_first_signal_only_one_trade():
    # call breakout at 10:10, then a collapse that would satisfy the put rules later in the day
    c = flat_day_closes()
    c[30:40] = np.linspace(100.2, 101.0, 10)  # 10:10 check sees 101.0
    c[40:120] = 101.0
    c[120:] = 97.0
    provider, _ = build_scenario(c)
    day = run(provider)
    assert day.trade.side == "call" and day.trade.signal_time == "10:10"
    assert len(day.checks) == 2  # 10:00 and 10:10; nothing after the first signal


def test_trailing_stop_arms_and_fires_on_50pct_giveback():
    provider, _ = build_scenario(trailing_put_day_closes())
    t = run(provider).trade
    assert t.exit_reason == "trailing_stop"
    assert t.trail_armed and t.trail_peak >= 4
    assert t.trail_peak == pytest.approx(5.0, abs=0.01)  # entry 99.0, low 94.0, ATR 1.0
    # exit at the first minute the favorable move is <= 50% of peak
    assert t.trail_exit_value <= t.trail_peak * 0.5 + 1e-9
    assert t.trail_retracement_from_peak >= 0.5
    und = provider.minute[D]
    exit_ts = et_timestamp(D, t.exit_time)
    assert (99.0 - price_at(und, exit_ts)) / 1.0 == pytest.approx(t.trail_exit_value)
    prev = exit_ts - np.timedelta64(1, "m")
    assert (99.0 - price_at(und, prev)) > t.trail_peak * 0.5  # did not fire a minute earlier


def test_no_lookahead_in_check():
    provider, target = build_scenario(put_day_closes())
    bt = Backtester(CFG, provider)
    cal = provider.trading_days(D - dt.timedelta(days=60), D)
    daily = provider.daily_bars(cal[0].date, D)
    setup, bars = bt.build_setup(target, cal[:-1], daily)
    t = et_timestamp(D, "10:30")
    base = evaluate_check(setup, bars, t, CFG.strategy, CFG.assumptions)
    future = bars.copy()
    future.loc[future.index >= t, ["open", "high", "low", "close"]] = 1e6
    future.loc[future.index >= t, "volume"] = 1e9
    assert evaluate_check(setup, future, t, CFG.strategy, CFG.assumptions) == base


def test_missing_option_data_is_a_loud_data_error_not_a_fake_fill():
    provider, _ = build_scenario(put_day_closes(), build_options=False)
    day = run(provider)
    assert day.status == "data_error" and day.trade is None
    assert "PUT signal at 10:30" in day.reason and "no bars" in day.reason


def test_early_close_day_skipped():
    from qqq0dte.data import TradingDay

    provider, target = build_scenario(put_day_closes())
    provider.days[-1] = TradingDay(D, close_time=dt.time(13, 0))
    day = run(provider)
    assert day.status == "skipped" and "early close" in day.reason


def test_strike_selection_itm_and_signal_bar_close_fill():
    provider, _ = build_scenario(put_day_closes())
    cfg = dataclasses.replace(
        CFG, assumptions=dataclasses.replace(CFG.assumptions, strike_selection="itm", strike_offset=2, fill_model="signal_bar_close")
    )
    t = run(provider, cfg).trade
    assert t.strike == 101.0  # puts: ITM = higher strikes
    und = provider.minute[D]
    s_in = und.loc[et_timestamp(D, "10:29"), "close"]
    assert t.entry_price == pytest.approx(101 - s_in + 0.5, abs=1e-4)


def test_summary_and_outputs(tmp_path):
    provider, _ = build_scenario(put_day_closes())
    result = Backtester(CFG, provider).run(D, D)
    s = summarize(result)
    assert s["trades"] == 1 and s["days_with_no_trade"] == 0 and s["max_drawdown"] == 0.0
    paths = write_outputs(result, CFG, tmp_path)
    header = paths["trades"].read_text().splitlines()[0].split(",")
    assert header[:13] == [
        "date", "signal_time", "side", "strike", "option_symbol", "contracts", "entry_time",
        "entry_price", "exit_time", "exit_price", "exit_reason", "pnl", "pnl_pct",
    ]
    assert "trail_peak" in header and "trail_armed_time" in header and "trail_retracement_from_peak" in header
    checks = paths["checks"].read_text()
    assert "put_c1_close_lt_orlow_minus_atr" in checks
    assert "TRAIL_MEASUREMENT_BASIS" in paths["assumptions"].read_text()


def test_max_drawdown():
    from qqq0dte.backtest import BacktestResult, DayResult

    class T:  # minimal stand-in carrying only the fields summarize() reads
        def __init__(self, pnl):
            self.pnl, self.side, self.exit_reason, self.trail_armed = pnl, "put", "x", False

    days = [DayResult(D + dt.timedelta(days=i), "traded", trade=T(p)) for i, p in enumerate([100, -300, 50, -100, 400])]
    s = summarize(BacktestResult(D, D, days))
    assert s["max_drawdown"] == -350.0  # peak 100 -> trough -250
    assert s["total_pnl"] == 150.0 and s["wins"] == 3 and s["losses"] == 2


def test_stale_underlying_bars_are_a_data_error_not_a_silent_price():
    c = put_day_closes()
    provider, _ = build_scenario(c)
    bars = provider.minute[D]
    gap = (bars.index >= et_timestamp(D, "10:05")) & (bars.index < et_timestamp(D, "12:00"))
    provider.minute[D] = bars.loc[~gap]
    day = run(provider)
    assert day.status == "data_error" and "max_bar_gap_minutes" in day.reason and day.trade is None
