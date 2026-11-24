"""Thin REST client for Alpaca Market Data + Trading APIs (requests only, no SDK).

* Keys come ONLY from environment variables APCA_API_KEY_ID / APCA_API_SECRET_KEY.
* The trading endpoint is the PAPER host unless the client is constructed with
  live=True AND live_confirmed=True (only the CLI's --live flag does that).
* Every failure raises with a message that says exactly what is missing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable, Mapping

import pandas as pd
import requests

from .timeutil import to_utc_iso

log = logging.getLogger(__name__)

DATA_BASE_URL = "https://data.alpaca.markets"
PAPER_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_TRADING_BASE_URL = "https://api.alpaca.markets"

KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"
# Optional: APCA_AUTH_MODE=proxy means an outbound proxy you control adds the Alpaca credentials to every
# request (e.g. a cloud sandbox with saved network secrets). The process then holds no keys and sends none.
AUTH_MODE_ENV = "APCA_AUTH_MODE"


class AlpacaError(RuntimeError):
    """Any Alpaca API failure."""


class MissingCredentialsError(AlpacaError):
    pass


class AlpacaAuthError(AlpacaError):
    pass


class AlpacaSubscriptionError(AlpacaError):
    """Your Alpaca market-data plan doesn't cover the requested feed/data."""


class LiveTradingNotAuthorized(AlpacaError):
    pass


def load_credentials(env: Mapping[str, str] | None = None) -> tuple[str, str]:
    env = os.environ if env is None else env
    missing = [name for name in (KEY_ENV, SECRET_ENV) if not str(env.get(name, "")).strip()]
    if missing:
        raise MissingCredentialsError(
            "Missing Alpaca API credentials: environment variable(s) "
            + ", ".join(missing)
            + " not set.\nCreate PAPER trading keys at https://app.alpaca.markets "
            "(switch to the Paper account, then API Keys -> Generate) and export them:\n"
            f"  export {KEY_ENV}='your-key-id'\n  export {SECRET_ENV}='your-secret-key'\n"
            "Keys are never read from config.yaml or any file in this repo."
        )
    return str(env[KEY_ENV]).strip(), str(env[SECRET_ENV]).strip()


def _message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict):
            return str(body.get("message") or body)
        return str(body)
    except ValueError:
        return (resp.text or "").strip()[:500]


class AlpacaClient:
    def __init__(
        self,
        key_id: str | None,
        secret_key: str | None,
        *,
        live: bool = False,
        live_confirmed: bool = False,
        session: requests.Session | None = None,
        max_retries: int = 5,
        timeout: float = 30.0,
    ) -> None:
        if live and live_confirmed is not True:
            raise LiveTradingNotAuthorized(
                "Refusing to create a LIVE Alpaca trading client without explicit confirmation "
                "(the --live CLI flag)."
            )
        self.live = bool(live)
        self.trading_base_url = LIVE_TRADING_BASE_URL if self.live else PAPER_TRADING_BASE_URL
        self.data_base_url = DATA_BASE_URL
        self._headers = {"Accept": "application/json"}
        if key_id is None and secret_key is None:
            self.auth = "proxy"  # credentials are injected by an outbound proxy
        elif key_id and secret_key:
            self.auth = "headers"
            self._headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})
        else:
            raise MissingCredentialsError("Both the Alpaca key ID and the secret key are required")
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.timeout = timeout

    @classmethod
    def from_env(cls, *, live: bool = False, live_confirmed: bool = False, **kwargs: Any) -> "AlpacaClient":
        mode = os.environ.get(AUTH_MODE_ENV, "headers").strip().lower()
        if mode == "proxy":
            log.info("%s=proxy: Alpaca credentials are added by the outbound proxy; none are held here.", AUTH_MODE_ENV)
            return cls(None, None, live=live, live_confirmed=live_confirmed, **kwargs)
        if mode != "headers":
            raise MissingCredentialsError(f"{AUTH_MODE_ENV} must be 'headers' (default) or 'proxy', got {mode!r}")
        key, secret = load_credentials()
        return cls(key, secret, live=live, live_confirmed=live_confirmed, **kwargs)

    # ------------------------------------------------------------------ core
    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        allow_404: bool = False,
        retry: bool = True,
    ) -> Any:
        resp = None
        retries = self.max_retries if retry else 0
        for attempt in range(retries + 1):
            try:
                resp = self.session.request(
                    method, url, headers=self._headers, params=params, json=json_body, timeout=self.timeout
                )
            except requests.RequestException as exc:
                if attempt < retries:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise AlpacaError(
                    f"Could not reach {url}: {exc}. Check your internet connection, and that your "
                    "firewall/proxy allows data.alpaca.markets and paper-api.alpaca.markets."
                ) from exc
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < retries:
                    wait = min(2**attempt, 30)
                    log.warning("Alpaca %s %s -> HTTP %s, retrying in %ss", method, url, resp.status_code, wait)
                    time.sleep(wait)
                    continue
            break
        assert resp is not None
        if resp.status_code == 404 and allow_404:
            return None
        if resp.status_code == 401:
            raise AlpacaAuthError(
                f"Alpaca rejected the API keys (HTTP 401) for {url}: {_message(resp)}. "
                f"Check {KEY_ENV}/{SECRET_ENV}. Paper and live accounts have DIFFERENT keys."
            )
        if resp.status_code == 403:
            msg = _message(resp)
            if "subscription" in msg.lower() or "feed" in msg.lower():
                raise AlpacaSubscriptionError(
                    f"Alpaca market-data permission error for {url}: {msg}. Your Alpaca data plan doesn't "
                    "include this feed/data. Real-time SIP stock data and OPRA options data need a paid "
                    "plan. Free plans: set data.live_stock_feed: iex and data.option_feed: indicative."
                )
            raise AlpacaAuthError(f"Alpaca returned HTTP 403 for {url}: {msg}")
        if not resp.ok:
            raise AlpacaError(f"Alpaca {method} {url} failed: HTTP {resp.status_code}: {_message(resp)}")
        if not resp.content:
            return None
        return resp.json()

    def _paginate_symbol_map(self, url: str, params: dict, key: str) -> dict[str, list]:
        params = dict(params)
        out: dict[str, list] = {}
        while True:
            data = self._request("GET", url, params=params) or {}
            for sym, rows in (data.get(key) or {}).items():
                out.setdefault(sym, []).extend(rows or [])
            token = data.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token

    # ------------------------------------------------------------ market data
    def stock_bars(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        feed: str,
        adjustment: str = "raw",
    ) -> list[dict]:
        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": to_utc_iso(start),
            "end": to_utc_iso(end),
            "limit": 10000,
            "adjustment": adjustment,
            "feed": feed,
            "sort": "asc",
        }
        return self._paginate_symbol_map(f"{self.data_base_url}/v2/stocks/bars", params, "bars").get(symbol, [])

    def stock_latest_bar(self, symbol: str, *, feed: str) -> dict | None:
        data = self._request(
            "GET", f"{self.data_base_url}/v2/stocks/bars/latest", params={"symbols": symbol, "feed": feed}
        ) or {}
        return (data.get("bars") or {}).get(symbol)

    def option_bars(
        self, symbols: Iterable[str], timeframe: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> dict[str, list]:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": to_utc_iso(start),
            "end": to_utc_iso(end),
            "limit": 10000,
            "sort": "asc",
        }
        return self._paginate_symbol_map(f"{self.data_base_url}/v1beta1/options/bars", params, "bars")

    def option_latest_trades(self, symbols: Iterable[str], *, feed: str) -> dict[str, dict]:
        data = self._request(
            "GET",
            f"{self.data_base_url}/v1beta1/options/trades/latest",
            params={"symbols": ",".join(symbols), "feed": feed},
        ) or {}
        return data.get("trades") or {}

    # --------------------------------------------------------------- trading
    def calendar(self, start, end) -> list[dict]:
        return self._request(
            "GET",
            f"{self.trading_base_url}/v2/calendar",
            params={"start": str(start), "end": str(end)},
        ) or []

    def clock(self) -> dict:
        return self._request("GET", f"{self.trading_base_url}/v2/clock")

    def account(self) -> dict:
        return self._request("GET", f"{self.trading_base_url}/v2/account")

    def option_contracts(
        self,
        *,
        underlying: str,
        expiration_date,
        contract_type: str | None = None,
        status: str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "underlying_symbols": underlying,
            "expiration_date": str(expiration_date),
            "limit": 10000,
        }
        if contract_type:
            params["type"] = contract_type
        if status:
            params["status"] = status
        if strike_gte is not None:
            params["strike_price_gte"] = f"{strike_gte:.2f}"
        if strike_lte is not None:
            params["strike_price_lte"] = f"{strike_lte:.2f}"
        out: list[dict] = []
        url = f"{self.trading_base_url}/v2/options/contracts"
        while True:
            data = self._request("GET", url, params=params) or {}
            out.extend(data.get("option_contracts") or [])
            token = data.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token

    def submit_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        client_order_id: str | None = None,
    ) -> dict:
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy/sell, got {side!r}")
        if int(qty) <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        body = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        log.info(
            "Submitting %s order to %s: %s", "LIVE" if self.live else "PAPER", self.trading_base_url, body
        )
        try:
            # Never auto-retry an order POST: a retry after a timeout could double the position.
            return self._request("POST", f"{self.trading_base_url}/v2/orders", json_body=body, retry=False)
        except AlpacaError:
            if client_order_id:
                existing = self.get_order_by_client_order_id(client_order_id)
                if existing:
                    log.warning("Order POST errored but Alpaca has order %s; using it", existing.get("id"))
                    return existing
            raise

    def get_order_by_client_order_id(self, client_order_id: str) -> dict | None:
        return self._request(
            "GET",
            f"{self.trading_base_url}/v2/orders:by_client_order_id",
            params={"client_order_id": client_order_id},
            allow_404=True,
        )

    def get_order(self, order_id: str) -> dict:
        return self._request("GET", f"{self.trading_base_url}/v2/orders/{order_id}")

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"{self.trading_base_url}/v2/orders/{order_id}", allow_404=True)

    def get_position(self, symbol: str) -> dict | None:
        return self._request("GET", f"{self.trading_base_url}/v2/positions/{symbol}", allow_404=True)
