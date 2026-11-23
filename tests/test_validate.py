"""Validation-report plumbing, exercised on a SYNTHETIC day dated Nov 20, 2026.

This does NOT reproduce the creator's trade. That needs real Alpaca data
(`python -m qqq0dte validate`). It only checks that the report compares fields
correctly and never tunes anything.
"""

import numpy as np

from qqq0dte.config import load_config
from qqq0dte.validate import CREATOR_EXAMPLE, EXIT_DATA, EXIT_MISMATCH, EXIT_OK, run_validation
from tests.synthetic import build_scenario, flat_day_closes, put_day_closes


def test_creator_example_numbers_are_internally_consistent():
    ex = CREATOR_EXAMPLE
    assert round(ex.entry_price * 100 * ex.contracts, 2) == ex.cost
    assert round(ex.exit_price * 100 * ex.contracts, 2) == ex.proceeds
    assert ex.proceeds - ex.cost == ex.profit
    assert round(ex.profit / ex.cost * 100, 1) == ex.return_pct


def test_validation_signal_pass_but_prices_differ_on_synthetic_day():
    provider, _ = build_scenario(put_day_closes(632.0), base=632.0)
    text, code, day = run_validation(load_config(None), provider)
    assert code == EXIT_OK
    assert "SIGNAL CHECK: PASS" in text
    assert "differs" in text  # synthetic option prices do not match the creator's, and must not be forced to
    assert "STRIKE_SELECTION='atm' picked $631" in text  # QQQ 631.00 at the signal, ATM is 631, not 632
    assert "Sensitivity" in text


def test_validation_reports_failure_reasons():
    provider, _ = build_scenario(flat_day_closes(632.0), base=632.0)
    text, code, day = run_validation(load_config(None), provider, sensitivity=False)
    assert code == EXIT_MISMATCH
    assert "SIGNAL CHECK: FAIL" in text
    assert "Why the 10:30 check did not give a PUT" in text and "FAIL" in text


def test_validation_data_error_exit_code():
    provider, _ = build_scenario(put_day_closes(632.0), base=632.0, build_options=False)
    text, code, day = run_validation(load_config(None), provider, sensitivity=False)
    assert code == EXIT_DATA and day.status == "data_error"
    assert "SIGNAL CHECK: PASS" in text  # the signal itself still fired at 10:30
    assert np.isfinite(day.checks[-1].close)
