import datetime as dt

import numpy as np
import pandas as pd
import pytest

from qqq0dte.data import DataUnavailable, TradingDay
from qqq0dte.indicators import (
    completed_bars,
    compute_atr,
    minute_move_profile,
    minute_offset,
    noise_band,
    noise_sigma,
    opening_range,
    price_at,
    session_open_price,
    vwap_at,
)
from qqq0dte.timeutil import et_timestamp
from tests.synthetic import bars_from_closes

DAY = TradingDay(dt.date(2026, 11, 20))


def test_price_at_uses_only_completed_bars():
    bars = bars_from_closes(DAY, [100, 101, 102, 103], first_open=100)
    # at 9:32 the bars 9:30 and 9:31 are complete; the 9:32 bar has not started closing
    assert price_at(bars, et_timestamp(DAY.date, "09:32")) == 101
    assert len(completed_bars(bars, et_timestamp(DAY.date, "09:32"))) == 2
    with pytest.raises(DataUnavailable):
        price_at(bars, et_timestamp(DAY.date, "09:30"))


def test_minute_offset():
    assert minute_offset(DAY.open_ts, et_timestamp(DAY.date, "10:30")) == 60


def test_session_open_and_opening_range():
    closes = [100.5, 99.0, 101.0] + [100.0] * 12 + [200.0]  # bar 15 (9:45) must be excluded
    bars = bars_from_closes(DAY, closes, first_open=100.0)
    assert session_open_price(bars, DAY.open_ts) == 100.0
    hi, lo, n = opening_range(bars, DAY.open_ts, 15)
    assert n == 15
    assert hi == pytest.approx(101.01)
    assert lo == pytest.approx(98.99)


def test_session_open_missing_930_bar_is_loud():
    bars = bars_from_closes(DAY, [100, 101], first_open=100).iloc[1:]
    with pytest.raises(DataUnavailable, match="open"):
        session_open_price(bars, DAY.open_ts)


def test_vwap_hlc3_and_close():
    idx = pd.date_range(DAY.open_ts, periods=2, freq="1min")
    bars = pd.DataFrame(
        {"open": [10, 12], "high": [12, 14], "low": [9, 11], "close": [11, 13], "volume": [100, 300], "vwap": [10.5, 12.5], "trade_count": 1},
        index=idx,
    )
    t = et_timestamp(DAY.date, "09:32")
    tp1, tp2 = (12 + 9 + 11) / 3, (14 + 11 + 13) / 3
    assert vwap_at(bars, t, "hlc3") == pytest.approx((tp1 * 100 + tp2 * 300) / 400)
    assert vwap_at(bars, t, "close") == pytest.approx((11 * 100 + 13 * 300) / 400)
    assert vwap_at(bars, t, "bar_vwap") == pytest.approx((10.5 * 100 + 12.5 * 300) / 400)
    # lookahead safety: at 9:31 only the first bar counts
    assert vwap_at(bars, et_timestamp(DAY.date, "09:31"), "close") == pytest.approx(11)


def test_compute_atr_simple_average_of_true_range_excluding_today():
    dates = pd.DatetimeIndex(pd.to_datetime(["2026-11-12", "2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]), name="date")
    df = pd.DataFrame(
        {
            "high": [10, 12, 11, 15, 14, 13, 999],
            "low": [9, 10, 9, 12, 13, 10, 0],
            "close": [9.5, 11, 10, 14, 13.5, 11, 500],
        },
        index=dates,
    )
    # TRs for 11-13..11-19 (prev close in parens):
    # 11-13: max(2, |12-9.5|=2.5, |10-9.5|=.5) = 2.5
    # 11-16: max(2, |11-11|=0, |9-11|=2) = 2
    # 11-17: max(3, |15-10|=5, |12-10|=2) = 5
    # 11-18: max(1, |14-14|=0, |13-14|=1) = 1
    # 11-19: max(3, |13-13.5|=.5, |10-13.5|=3.5) = 3.5
    assert compute_atr(df, dt.date(2026, 11, 20), 5) == pytest.approx((2.5 + 2 + 5 + 1 + 3.5) / 5)
    with pytest.raises(DataUnavailable):
        compute_atr(df, dt.date(2026, 11, 18), 5)


def test_noise_profile_dollars_percent_and_ffill():
    closes = np.full(390, 101.0)
    bars = bars_from_closes(DAY, closes, first_open=100.0).drop(index=[et_timestamp(DAY.date, "09:35")])
    prof = minute_move_profile(bars, DAY, 100.0, "dollars")
    assert prof.shape == (390,)
    assert np.allclose(prof, 1.0)  # the missing 9:35 bar is carried forward
    pct = minute_move_profile(bars, DAY, 100.0, "percent")
    assert np.allclose(pct, 0.01)


def test_noise_profile_early_close_is_nan_after_close():
    half = TradingDay(dt.date(2026, 11, 27), close_time=dt.time(13, 0))
    bars = bars_from_closes(half, np.full(210, 100.5), first_open=100.0)
    prof = minute_move_profile(bars, half, 100.0)
    assert np.allclose(prof[:210], 0.5) and np.isnan(prof[210:]).all()
    sigma = noise_sigma([prof, np.full(390, 1.5)])
    assert sigma[0] == pytest.approx(1.0) and sigma[300] == pytest.approx(1.5)


def test_noise_band_formula():
    assert noise_band(600.0, 2.0, 1.25, "dollars") == (602.5, 597.5)
    up, lo = noise_band(600.0, 0.01, 1.25, "percent")
    assert up == pytest.approx(607.5) and lo == pytest.approx(592.5)
