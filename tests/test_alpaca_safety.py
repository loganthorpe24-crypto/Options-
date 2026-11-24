import json

import pandas as pd
import pytest

from qqq0dte.alpaca import (
    LIVE_TRADING_BASE_URL,
    PAPER_TRADING_BASE_URL,
    AlpacaClient,
    AlpacaSubscriptionError,
    LiveTradingNotAuthorized,
    MissingCredentialsError,
    load_credentials,
)
from qqq0dte.cli import build_parser
from qqq0dte.data import AlpacaDataProvider, DataUnavailable, TradingDay


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.content = json.dumps(body).encode()
        self.text = json.dumps(body)

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._body


class RecordingSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "params": params, "json": json})
        return self.responses.pop(0)


def test_credentials_only_from_env_and_missing_is_explicit():
    with pytest.raises(MissingCredentialsError) as exc:
        load_credentials({})
    assert "APCA_API_KEY_ID" in str(exc.value) and "APCA_API_SECRET_KEY" in str(exc.value)
    with pytest.raises(MissingCredentialsError) as exc:
        load_credentials({"APCA_API_KEY_ID": "x"})
    assert "APCA_API_SECRET_KEY" in str(exc.value) and "APCA_API_KEY_ID," not in str(exc.value)
    assert load_credentials({"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"}) == ("k", "s")


def test_default_client_is_paper():
    c = AlpacaClient("k", "s")
    assert c.live is False and c.trading_base_url == PAPER_TRADING_BASE_URL


def test_live_client_requires_explicit_confirmation():
    with pytest.raises(LiveTradingNotAuthorized):
        AlpacaClient("k", "s", live=True)
    assert AlpacaClient("k", "s", live=True, live_confirmed=True).trading_base_url == LIVE_TRADING_BASE_URL


def test_cli_run_is_paper_unless_live_flag():
    p = build_parser()
    assert p.parse_args(["run"]).live is False
    assert p.parse_args(["run", "--dry-run"]).live is False
    assert p.parse_args(["run", "--live"]).live is True


def test_orders_go_to_paper_host_with_env_keys_in_headers():
    sess = RecordingSession([FakeResponse(200, {"id": "1", "status": "new"})])
    c = AlpacaClient("KEY", "SECRET", session=sess)
    c.submit_order(symbol="QQQ261120P00632000", qty=10, side="buy")
    call = sess.calls[0]
    assert call["url"] == f"{PAPER_TRADING_BASE_URL}/v2/orders"
    assert call["headers"]["APCA-API-KEY-ID"] == "KEY"
    assert call["json"] == {"symbol": "QQQ261120P00632000", "qty": "10", "side": "buy", "type": "market", "time_in_force": "day"}


def test_subscription_error_is_explained():
    sess = RecordingSession([FakeResponse(403, {"message": "subscription does not permit querying recent SIP data"})])
    c = AlpacaClient("k", "s", session=sess, max_retries=0)
    with pytest.raises(AlpacaSubscriptionError, match="iex"):
        c.stock_latest_bar("QQQ", feed="sip")


def test_option_bars_pagination_and_empty_is_loud(tmp_path):
    t0 = pd.Timestamp("2026-11-20 14:30", tz="UTC")
    page1 = {"bars": {"QQQ261120P00632000": [{"t": "2026-11-20T15:30:00Z", "o": 11.2, "h": 11.3, "l": 11.1, "c": 11.26, "v": 5, "n": 2, "vw": 11.2}]}, "next_page_token": "abc"}
    page2 = {"bars": {"QQQ261120P00632000": [{"t": "2026-11-20T15:31:00Z", "o": 11.3, "h": 11.4, "l": 11.2, "c": 11.3, "v": 5, "n": 2, "vw": 11.3}]}, "next_page_token": None}
    sess = RecordingSession([FakeResponse(200, page1), FakeResponse(200, page2)])
    c = AlpacaClient("k", "s", session=sess)
    out = c.option_bars(["QQQ261120P00632000"], "1Min", t0, t0 + pd.Timedelta(hours=7))
    assert len(out["QQQ261120P00632000"]) == 2
    assert sess.calls[1]["params"]["page_token"] == "abc"

    empty = RecordingSession([FakeResponse(200, {"bars": {}, "next_page_token": None})])
    provider = AlpacaDataProvider(AlpacaClient("k", "s", session=empty), underlying="QQQ", stock_feed="sip", cache_dir=tmp_path)
    import datetime as dt

    with pytest.raises(DataUnavailable, match="NO 1-minute trade bars"):
        provider.option_minute_bars("QQQ261120P00632000", TradingDay(dt.date(2026, 11, 20)))


def test_offline_provider_never_invents_data(tmp_path):
    import datetime as dt

    provider = AlpacaDataProvider(None, underlying="QQQ", stock_feed="sip", cache_dir=tmp_path, offline=True)
    with pytest.raises(DataUnavailable, match="offline"):
        provider.minute_bars(TradingDay(dt.date(2026, 11, 20)))


def test_proxy_auth_mode_sends_no_key_headers(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("APCA_AUTH_MODE", "proxy")
    sess = RecordingSession([FakeResponse(200, {"is_open": False})])
    c = AlpacaClient.from_env(session=sess)
    c.clock()
    assert c.auth == "proxy" and c.trading_base_url == PAPER_TRADING_BASE_URL
    assert "APCA-API-KEY-ID" not in sess.calls[0]["headers"] and "APCA-API-SECRET-KEY" not in sess.calls[0]["headers"]
    monkeypatch.setenv("APCA_AUTH_MODE", "bogus")
    with pytest.raises(MissingCredentialsError):
        AlpacaClient.from_env()
