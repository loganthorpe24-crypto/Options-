import dataclasses
import datetime as dt
import math

import pandas as pd
import pytest

from qqq0dte import exits
from qqq0dte.config import CREATOR_RULES, Config, ConfigError, assumption_report, load_config
from qqq0dte.exits import TrailingStop, favorable_move
from qqq0dte.options import occ_symbol, parse_occ, select_strike


# ----------------------------------------------------------------- options
def test_occ_symbol_matches_creator_contract():
    assert occ_symbol("QQQ", dt.date(2026, 11, 20), "put", 632) == "QQQ261120P00632000"
    assert occ_symbol("QQQ", dt.date(2026, 11, 20), "call", 632.5) == "QQQ261120C00632500"
    assert parse_occ("QQQ261120P00632000") == ("QQQ", dt.date(2026, 11, 20), "put", 632.0)


def test_select_strike_atm_and_ties():
    strikes = [630, 631, 632, 633]
    assert select_strike(631.7, "put", strikes) == 632
    assert select_strike(631.5, "put", strikes, tie_break="lower") == 631
    assert select_strike(631.5, "put", strikes, tie_break="higher") == 632


def test_select_strike_itm_otm_direction():
    strikes = list(range(620, 641))
    assert select_strike(630.2, "put", strikes, method="itm", offset=2) == 632
    assert select_strike(630.2, "put", strikes, method="otm", offset=2) == 628
    assert select_strike(630.2, "call", strikes, method="itm", offset=2) == 628
    assert select_strike(630.2, "call", strikes, method="otm", offset=2) == 632


# ------------------------------------------------------------------- exits
def test_default_basis_is_underlying_over_atr5():
    assert exits.TRAIL_MEASUREMENT_BASIS == "UNDERLYING_MOVE_OVER_ATR5"
    assert favorable_move("UNDERLYING_MOVE_OVER_ATR5", side="put", entry_underlying=630, underlying=622, atr=8) == 1.0
    assert favorable_move("UNDERLYING_MOVE_OVER_ATR5", side="call", entry_underlying=630, underlying=622, atr=8) == -1.0


def _stop(**kw):
    t0 = pd.Timestamp("2026-11-20 10:30", tz="America/New_York")
    base = dict(side="put", entry_time=t0, entry_underlying=100.0, atr=1.0, entry_option_price=1.0, arm_threshold=4.0, giveback_fraction=0.5)
    base.update(kw)
    return TrailingStop(**base), t0


def test_trailing_stop_sequence():
    stop, t0 = _stop()
    m = lambda i: t0 + pd.Timedelta(minutes=i)  # noqa: E731
    assert not stop.update(m(1), underlying=98.0)  # +2 ATR, not armed
    assert not stop.armed
    assert not stop.update(m(2), underlying=96.0)  # +4 ATR -> arms
    assert stop.armed and stop.armed_time == m(2) and stop.armed_value == 4.0
    assert not stop.update(m(3), underlying=94.0)  # peak +6
    assert stop.peak == 6.0 and stop.stop_level == 3.0
    assert not stop.update(m(4), underlying=96.9)  # +3.1 > 3.0
    assert stop.update(m(5), underlying=97.0)  # +3.0 <= 3.0 -> exit
    assert stop.exit_value == 3.0 and stop.retracement_from_peak == 0.5
    assert stop.peak_time == m(3) and stop.peak_underlying == 94.0


def test_trailing_stop_never_arms_below_threshold_even_on_big_giveback():
    stop, t0 = _stop()
    assert not stop.update(t0 + pd.Timedelta(minutes=1), underlying=96.1)  # +3.9
    assert not stop.update(t0 + pd.Timedelta(minutes=2), underlying=100.5)  # gave back everything
    assert not stop.armed and not stop.triggered


def test_trailing_stop_state_roundtrip():
    stop, t0 = _stop()
    stop.update(t0 + pd.Timedelta(minutes=1), underlying=95.0)
    other, _ = _stop()
    other.restore(stop.to_state())
    assert (other.peak, other.armed, other.armed_time) == (stop.peak, stop.armed, stop.armed_time)
    bad = dict(stop.to_state(), basis="OPTION_PREMIUM_RETURN")
    with pytest.raises(ValueError):
        other.restore(bad)


def test_option_premium_basis_alternative():
    stop, t0 = _stop(basis="OPTION_PREMIUM_RETURN")
    assert stop.needs_option_price
    assert not stop.update(t0 + pd.Timedelta(minutes=1), underlying=0, option_price=math.nan)
    assert not stop.update(t0 + pd.Timedelta(minutes=2), underlying=0, option_price=5.0)  # +400% arms
    assert stop.armed
    assert stop.update(t0 + pd.Timedelta(minutes=3), underlying=0, option_price=3.0)  # +200% <= 50% of +400%


# ------------------------------------------------------------------ config
def test_config_yaml_matches_creator_rules_and_defaults():
    cfg = load_config("config.yaml")
    assert dataclasses.asdict(cfg.strategy) == CREATOR_RULES
    assert dataclasses.replace(cfg, source_path=None) == Config()
    assert cfg.assumptions.strike_selection == "atm" and cfg.assumptions.contracts == 10


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("assumptions:\n  strike_selectoin: atm\n")
    with pytest.raises(ConfigError, match="strike_selectoin"):
        load_config(p)


def test_config_rejects_bad_choice(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("assumptions:\n  strike_selection: deep_itm\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_config_has_no_key_or_live_switch():
    text = open("config.yaml").read().lower()
    for forbidden in ("secret_key:", "api_key:", "key_id:", "live: true", "live_trading:"):
        assert forbidden not in text


def test_assumption_report_flags_changed_creator_rules():
    cfg = Config(strategy=dataclasses.replace(Config().strategy, noise_multiplier=1.0))
    report = "\n".join(assumption_report(cfg))
    assert "CHANGED (creator: 1.25)" in report
    assert "STRIKE_SELECTION" in report and "TRAIL_MEASUREMENT_BASIS" in report
