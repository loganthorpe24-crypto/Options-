"""AlpacaDataProvider end-to-end with a fake REST client that returns Alpaca-shaped JSON
(SYNTHETIC prices). Checks parsing, timezones, the CSV cache, and that an --offline
re-run reproduces the online result exactly."""

import datetime as dt

import pandas as pd

from qqq0dte.backtest import Backtester
from qqq0dte.config import load_config
from qqq0dte.data import AlpacaDataProvider
from tests.synthetic import build_scenario, put_day_closes

D = dt.date(2026, 11, 20)


def _raw(df):
    return [
        {"t": ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "o": r.open, "h": r.high, "l": r.low, "c": r.close, "v": r.volume, "n": 7, "vw": r.vwap}
        for ts, r in df.iterrows()
    ]


class FakeRest:
    def __init__(self, scenario):
        self.s = scenario
        self.calls = []

    def calendar(self, start, end):
        self.calls.append("calendar")
        return [{"date": d.date.isoformat(), "open": "09:30", "close": "16:00"} for d in self.s.days if start <= d.date <= end]

    def stock_bars(self, symbol, timeframe, start, end, *, feed, adjustment="raw"):
        self.calls.append(f"stock_bars:{timeframe}")
        assert symbol == "QQQ" and feed == "sip" and adjustment == "raw"
        if timeframe == "1Min":
            df = self.s.minute.get(start.date())
            return [] if df is None else _raw(df.loc[(df.index >= start) & (df.index <= end)])
        daily = self.s.daily.loc[(self.s.daily.index >= start.tz_localize(None)) & (self.s.daily.index <= end.tz_localize(None))]
        return [
            {"t": pd.Timestamp(ix).tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
             "o": r.open, "h": r.high, "l": r.low, "c": r.close, "v": r.volume, "n": 1, "vw": r.close}
            for ix, r in daily.iterrows()
        ]

    def option_contracts(self, *, underlying, expiration_date, contract_type=None, status=None, **kw):
        self.calls.append(f"contracts:{status}")
        if status != "inactive":
            return []
        r = "P" if contract_type == "put" else "C"
        return [
            {"symbol": f"QQQ{expiration_date:%y%m%d}{r}{int(k * 1000):08d}", "strike_price": f"{k:.2f}", "expiration_date": expiration_date.isoformat(), "type": contract_type}
            for k in self.s.strikes
        ] + [{"symbol": f"QQQ1{expiration_date:%y%m%d}{r}00099000", "strike_price": "99.5", "expiration_date": expiration_date.isoformat()}]

    def option_bars(self, symbols, timeframe, start, end):
        self.calls.append("option_bars")
        out = {}
        for sym in symbols:
            df = self.s.options.get(sym, {}).get(start.date())
            if df is not None:
                out[sym] = _raw(df)
        return out


def test_online_then_offline_identical(tmp_path):
    scenario, _ = build_scenario(put_day_closes())
    rest = FakeRest(scenario)
    cfg = load_config(None)
    kw = dict(underlying="QQQ", stock_feed="sip", cache_dir=tmp_path, today=dt.date(2026, 11, 23))

    online = Backtester(cfg, AlpacaDataProvider(rest, **kw)).run(D, D).days[0]
    assert online.status == "traded"
    t = online.trade
    assert (t.side, t.signal_time, t.strike, t.strike_source) == ("put", "10:30", 99.0, "alpaca_contracts")
    assert "option_bars" in rest.calls

    offline = Backtester(cfg, AlpacaDataProvider(None, offline=True, **kw)).run(D, D).days[0]
    assert offline.trade == online.trade
    assert [c.as_row() for c in offline.checks] == [c.as_row() for c in online.checks]
    assert (tmp_path / "calendar" / "2026.json").exists()
    assert (tmp_path / "options" / "bars" / "QQQ261120P00099000" / "2026-11-20.csv").exists()


def test_adjusted_contracts_are_ignored_and_bars_parse_to_et(tmp_path):
    scenario, target = build_scenario(put_day_closes())
    p = AlpacaDataProvider(FakeRest(scenario), underlying="QQQ", stock_feed="sip", cache_dir=tmp_path, today=dt.date(2026, 11, 23))
    strikes, src = p.option_strikes(D, "put", 99.0)
    assert 99.5 not in strikes and src == "alpaca_contracts"  # QQQ1 (adjusted) root excluded
    bars = p.minute_bars(target)
    assert str(bars.index.tz) == "America/New_York"
    assert bars.index[0] == pd.Timestamp("2026-11-20 09:30", tz="America/New_York") and len(bars) == 390
    daily = p.daily_bars(dt.date(2026, 11, 1), D)
    assert pd.Timestamp(D) in daily.index


def test_daily_cache_month_written_mid_month_is_refreshed_after_month_end(tmp_path):
    scenario, _ = build_scenario(put_day_closes(), target_date=dt.date(2026, 11, 30))
    rest = FakeRest(scenario)
    kw = dict(underlying="QQQ", stock_feed="sip", cache_dir=tmp_path)
    # On Mon Nov 30, the Nov file can only hold bars through Nov 27
    AlpacaDataProvider(rest, today=dt.date(2026, 11, 30), **kw).daily_bars(dt.date(2026, 11, 1), dt.date(2026, 11, 27))
    assert not (tmp_path / "stocks/QQQ/1Day/sip_raw/2026-11.csv").exists()
    # On Dec 1 the ATR needs Nov 30: the partial file must not be trusted
    d = AlpacaDataProvider(rest, today=dt.date(2026, 12, 1), **kw).daily_bars(dt.date(2026, 11, 1), dt.date(2026, 11, 30))
    assert pd.Timestamp("2026-11-30") in d.index
    assert (tmp_path / "stocks/QQQ/1Day/sip_raw/2026-11.csv").exists()


def test_nyse_calendar_and_grid_strike_sources_need_no_trading_api(tmp_path):
    import pytest

    pytest.importorskip("pandas_market_calendars")
    p = AlpacaDataProvider(None, underlying="QQQ", stock_feed="sip", cache_dir=tmp_path, offline=True,
                           calendar_source="pandas_market_calendars", strike_source="grid")
    days = {d.date: d for d in p.trading_days(dt.date(2025, 1, 6), dt.date(2025, 12, 31))}
    assert dt.date(2025, 1, 9) not in days  # one-off closure
    assert days[dt.date(2025, 11, 28)].close_time == dt.time(13, 0) and days[dt.date(2025, 11, 28)].is_early_close
    assert days[dt.date(2025, 11, 26)].close_time == dt.time(16, 0)
    strikes, src = p.option_strikes(D, "put", 631.4)
    assert src == "grid_1" and 631.0 in strikes and 632.0 in strikes
