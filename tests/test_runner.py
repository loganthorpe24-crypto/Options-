"""Runner tests: a fake Alpaca client + simulated clock replay a SYNTHETIC session."""

import dataclasses
import datetime as dt

import pandas as pd
import pytest

from qqq0dte.alpaca import PAPER_TRADING_BASE_URL
from qqq0dte.config import load_config
from qqq0dte.runner import Runner, StateStore
from qqq0dte.timeutil import ET, et_timestamp
from tests.synthetic import build_scenario, flat_day_closes, put_day_closes

D = dt.date(2026, 11, 20)


class FakeClock:
    def __init__(self, start):
        self.t = pd.Timestamp(start)

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += pd.Timedelta(seconds=seconds)

    def sleep_until(self, ts):
        if ts > self.t:
            self.t = pd.Timestamp(ts)


class FakeAlpaca:
    """Serves only bars that have COMPLETED by the fake clock's 'now'."""

    def __init__(self, provider, clock, live=False):
        self.provider, self.clock, self.live = provider, clock, live
        self.trading_base_url = PAPER_TRADING_BASE_URL
        self.orders = {}
        self.order_log = []
        self.position_qty = {}

    def account(self):
        return {"status": "ACTIVE", "options_trading_level": 3}

    def stock_latest_bar(self, symbol, feed):
        return {}

    def stock_bars(self, symbol, timeframe, start, end, feed, adjustment="raw"):
        bars = self.provider.minute[D]
        done = bars.loc[(bars.index <= end) & (bars.index + pd.Timedelta(minutes=1) <= self.clock.now())]
        return [
            {"t": ts.tz_convert("UTC").isoformat(), "o": r.open, "h": r.high, "l": r.low, "c": r.close, "v": r.volume, "vw": r.vwap, "n": 1}
            for ts, r in done.iterrows()
        ]

    def option_contracts(self, *, underlying, expiration_date, contract_type=None, status=None, strike_gte=None, strike_lte=None):
        r = "P" if contract_type == "put" else "C"
        return [
            {"symbol": f"QQQ{D:%y%m%d}{r}{int(k * 1000):08d}", "strike_price": str(k), "expiration_date": D.isoformat(), "tradable": True}
            for k in self.provider.strikes
            if strike_gte <= k <= strike_lte
        ]

    def _option_price(self, symbol):
        bars = self.provider.options[symbol][D]
        cur = bars.loc[bars.index <= self.clock.now().floor("min")]
        return float(cur["open"].iloc[-1])

    def option_latest_trades(self, symbols, feed):
        return {s: {"p": self._option_price(s)} for s in symbols}

    def submit_order(self, *, symbol, qty, side, order_type, time_in_force, client_order_id=None):
        assert self.live is False, "test client must never be live"
        oid = f"o{len(self.orders) + 1}"
        order = {"id": oid, "status": "filled", "filled_avg_price": str(self._option_price(symbol)), "filled_qty": str(qty), "symbol": symbol, "side": side, "client_order_id": client_order_id, "at": self.clock.now()}
        self.orders[oid] = order
        self.order_log.append(order)
        self.position_qty[symbol] = self.position_qty.get(symbol, 0) + (qty if side == "buy" else -qty)
        return order

    def get_order(self, oid):
        return self.orders[oid]

    def get_order_by_client_order_id(self, cid):
        return next((o for o in self.orders.values() if o["client_order_id"] == cid), None)

    def cancel_order(self, oid):
        pass

    def get_position(self, symbol):
        q = self.position_qty.get(symbol, 0)
        return {"qty": str(q)} if q else None

    def fail_stock_bars_between(self, start, end):
        """Simulate an Alpaca outage for QQQ bars between two clock times."""
        real = self.stock_bars

        def flaky(*args, **kw):
            if start <= self.clock.now() < end:
                from qqq0dte.alpaca import AlpacaError

                raise AlpacaError("simulated HTTP 503")
            return real(*args, **kw)

        self.stock_bars = flaky


def make(tmp_path, closes=None, start="09:20", dry_run=False):
    provider, _ = build_scenario(put_day_closes() if closes is None else closes)
    base = load_config(None)
    cfg = dataclasses.replace(base, runner=dataclasses.replace(base.runner, state_dir=str(tmp_path / "state"), log_dir=str(tmp_path / "logs")))
    clock = FakeClock(et_timestamp(D, start))
    client = FakeAlpaca(provider, clock)
    return Runner(cfg, client, provider, dry_run=dry_run, clock=clock), client, clock, cfg


def test_paper_session_buys_at_signal_and_sells_at_1545(tmp_path):
    runner, client, clock, cfg = make(tmp_path)
    runner.preflight()
    state = runner.run_session()
    assert [o["side"] for o in client.order_log] == ["buy", "sell"]
    buy, sell = client.order_log
    assert buy["symbol"] == "QQQ261120P00099000" and buy["client_order_id"] == "qqq0dte-2026-11-20-paper-entry"
    assert buy["at"].strftime("%H:%M") == "10:30"
    assert sell["at"].strftime("%H:%M") == "15:45"
    assert state.exit["reason"] == "hard_exit_15:45"
    assert state.exit["trail"]["armed"] is False
    checks = (tmp_path / "logs" / "checks_2026-11-20.csv").read_text().splitlines()
    assert len(checks) == 1 + 4  # header + 10:00, 10:10, 10:20, 10:30
    assert "put_c1_close_lt_orlow_minus_atr" in checks[0]
    trades = (tmp_path / "logs" / "trades.csv").read_text()
    assert "trail_peak" in trades and "hard_exit_15:45" in trades


def test_dry_run_sends_no_orders(tmp_path):
    runner, client, clock, cfg = make(tmp_path, dry_run=True)
    state = runner.run_session()
    assert client.order_log == []
    assert state.entry["simulated"] is True and state.exit["reason"] == "hard_exit_15:45"


def test_restart_after_trade_never_reenters(tmp_path):
    runner, client, clock, cfg = make(tmp_path)
    runner.run_session()
    n = len(client.order_log)
    clock.t = et_timestamp(D, "15:50")
    again = Runner(cfg, client, runner.provider, clock=clock)
    again.run_session()
    assert len(client.order_log) == n


def test_late_start_after_first_signal_takes_no_trade(tmp_path):
    runner, client, clock, cfg = make(tmp_path, start="11:00")
    state = runner.run_session()
    assert client.order_log == []
    assert state.first_signal["side"] == "put" and state.first_signal["replayed"] is True
    assert state.entry is None


def test_no_signal_day_no_orders(tmp_path):
    runner, client, clock, cfg = make(tmp_path, closes=flat_day_closes())
    state = runner.run_session()
    assert client.order_log == [] and "no signal" in state.notes


def test_runner_refuses_live_mismatch(tmp_path):
    runner, client, clock, cfg = make(tmp_path)
    with pytest.raises(RuntimeError, match="Safety"):
        Runner(cfg, client, runner.provider, live=True, clock=clock)
    with pytest.raises(ValueError):
        Runner(cfg, client, runner.provider, live=True, dry_run=True, clock=clock)


def test_state_store_roundtrip(tmp_path):
    store = StateStore(tmp_path, "paper")
    s = store.load(D)
    s.first_signal = {"side": "put"}
    store.save(s)
    assert store.load(D).first_signal == {"side": "put"}


def test_restart_mid_submit_resolves_existing_order_and_manages_it(tmp_path):
    runner, client, clock, cfg = make(tmp_path)
    clock.t = et_timestamp(D, "10:30:03")
    sym = "QQQ261120P00099000"
    order = client.submit_order(symbol=sym, qty=10, side="buy", order_type="market", time_in_force="day",
                                client_order_id="qqq0dte-2026-11-20-paper-entry")
    store = StateStore(cfg.runner.state_dir, "paper")
    st = store.load(D)
    st.first_signal = {"side": "put", "time": et_timestamp(D, "10:30").isoformat()}
    st.entry = {"status": "submitted", "side": "put", "symbol": sym, "strike": 99.0, "qty": 10,
                "signal_time": et_timestamp(D, "10:30").isoformat(), "entry_underlying": 99.0, "atr": 1.0,
                "fill_price": None, "order_id": None, "client_order_id": "qqq0dte-2026-11-20-paper-entry", "simulated": False}
    store.save(st)
    clock.t = et_timestamp(D, "11:00")
    state = Runner(cfg, client, runner.provider, clock=clock).run_session()
    assert [o["side"] for o in client.order_log] == ["buy", "sell"]  # no second buy
    assert state.entry["status"] == "filled" and state.entry["fill_price"] == float(order["filled_avg_price"])
    assert state.exit["reason"] == "hard_exit_15:45"


def test_restart_mid_submit_order_never_arrived_means_no_trade(tmp_path):
    runner, client, clock, cfg = make(tmp_path)
    store = StateStore(cfg.runner.state_dir, "paper")
    st = store.load(D)
    st.first_signal = {"side": "put"}
    st.entry = {"status": "submitted", "symbol": "QQQ261120P00099000", "client_order_id": "qqq0dte-2026-11-20-paper-entry", "simulated": False}
    store.save(st)
    clock.t = et_timestamp(D, "11:00")
    state = Runner(cfg, client, runner.provider, clock=clock).run_session()
    assert client.order_log == [] and state.entry["status"] == "not_filled:not_found"


def test_restart_mid_trade_replays_missed_minutes_for_trailing_stop(tmp_path):
    from tests.synthetic import trailing_put_day_closes

    runner, client, clock, cfg = make(tmp_path, closes=trailing_put_day_closes())
    sym = "QQQ261120P00099000"
    clock.t = et_timestamp(D, "10:30:03")
    client.submit_order(symbol=sym, qty=10, side="buy", order_type="market", time_in_force="day", client_order_id="qqq0dte-2026-11-20-paper-entry")
    store = StateStore(cfg.runner.state_dir, "paper")
    st = store.load(D)
    st.first_signal = {"side": "put"}
    st.entry = {"status": "filled", "side": "put", "symbol": sym, "strike": 99.0, "qty": 10,
                "signal_time": et_timestamp(D, "10:30").isoformat(), "entry_underlying": 99.0, "atr": 1.0,
                "fill_price": 0.5, "order_id": "o1", "client_order_id": "qqq0dte-2026-11-20-paper-entry", "simulated": False}
    store.save(st)
    # runner was down from 10:30 to 13:30; the 50% giveback happened at 12:50 (see backtest test)
    clock.t = et_timestamp(D, "13:30")
    state = Runner(cfg, client, runner.provider, clock=clock).run_session()
    assert state.exit["reason"] == "trailing_stop"
    assert pd.Timestamp(state.exit["time"]).strftime("%H:%M") == "12:50"
    assert state.exit["trail"]["armed"] is True and state.exit["trail"]["peak"] == pytest.approx(5.0)
    assert client.order_log[-1]["side"] == "sell" and client.order_log[-1]["at"].strftime("%H:%M") == "13:30"


def test_api_outage_mid_trade_still_exits_at_1545(tmp_path):
    runner, client, clock, cfg = make(tmp_path)
    client.fail_stock_bars_between(et_timestamp(D, "12:00"), et_timestamp(D, "16:00"))
    state = runner.run_session()
    assert [o["side"] for o in client.order_log] == ["buy", "sell"]
    assert client.order_log[-1]["at"].strftime("%H:%M") == "15:45"
    assert state.exit["reason"] == "hard_exit_15:45" and client.position_qty["QQQ261120P00099000"] == 0


def test_exit_order_errors_are_retried_without_double_selling(tmp_path):
    from qqq0dte.alpaca import AlpacaError

    runner, client, clock, cfg = make(tmp_path)
    real_submit, calls = client.submit_order, {"n": 0}

    def flaky_submit(**kw):
        if kw["side"] == "sell":
            calls["n"] += 1
            if calls["n"] == 1:
                raise AlpacaError("simulated timeout on the sell POST")
        return real_submit(**kw)

    client.submit_order = flaky_submit
    state = runner.run_session()
    sells = [o for o in client.order_log if o["side"] == "sell"]
    assert len(sells) == 1 and client.position_qty["QQQ261120P00099000"] == 0
    assert state.exit["qty"] == 10 and len(state.exit_orders) == 2


def test_paper_and_live_state_are_separate(tmp_path):
    runner, client, clock, cfg = make(tmp_path)
    live_store = StateStore(cfg.runner.state_dir, "live")
    st = live_store.load(D)
    st.first_signal = {"side": "put"}
    st.entry = {"status": "filled", "symbol": "QQQ261120P00099000", "qty": 10}
    live_store.save(st)
    state = runner.run_session()  # paper run is unaffected by the live file and trades normally
    assert state.mode == "paper" and state.exit is not None
    assert live_store.load(D).exit is None  # the live record is untouched
