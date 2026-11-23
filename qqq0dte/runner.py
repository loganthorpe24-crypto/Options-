"""Paper (default) / live runner. Uses the same signal and exit code as the backtester.

Modes:
  paper    (default)  orders go to paper-api.alpaca.markets
  --dry-run            no orders at all; entries/exits are marked at the latest option trade
  --live               orders go to api.alpaca.markets (REAL MONEY). Only via the CLI flag.

One session per call to run_session(). Every 10-minute check is logged with all
three conditions for both sides, and appended to logs/checks_YYYY-MM-DD.csv.
State is persisted per mode to state/YYYY-MM-DD_<mode>.json, so a restart can
never re-enter, and paper and live state are never mixed.

Once a position is open, the runner keeps going no matter what: data/API errors
skip that minute, and the 15:45 exit is retried until the market closes.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .alpaca import AlpacaError, AlpacaSubscriptionError
from .config import Config
from .data import DataUnavailable, MarketDataProvider, TradingDay, bars_from_alpaca, regular_session
from .exits import TrailingStop
from .indicators import opening_range, price_at
from .options import occ_symbol, parse_occ, select_strike
from .signals import CheckResult, DaySetup, atr_for_day, build_day_setup, check_times, evaluate_check
from .timeutil import ET, ONE_MINUTE, et_timestamp, hhmm, now_et

log = logging.getLogger(__name__)

TERMINAL_ORDER_STATES = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced"}
EXIT_RETRY_SECONDS = 5.0


class SystemClock:
    def now(self) -> pd.Timestamp:
        return now_et()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def sleep_until(self, ts: pd.Timestamp) -> None:
        while True:
            remaining = (ts - self.now()).total_seconds()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 15.0))


@dataclass
class SessionState:
    date: str
    mode: str = ""
    first_signal: dict | None = None
    entry: dict | None = None
    trail: dict | None = None
    pending_exit: dict | None = None
    exit_orders: list[str] = field(default_factory=list)
    exit: dict | None = None
    notes: list[str] = field(default_factory=list)


class StateStore:
    """One JSON file per (date, mode): paper, live and dry-run state never mix."""

    def __init__(self, directory: str | Path, mode_tag: str) -> None:
        self.dir = Path(directory)
        self.mode_tag = mode_tag

    def path(self, date: dt.date, mode_tag: str | None = None) -> Path:
        return self.dir / f"{date.isoformat()}_{mode_tag or self.mode_tag}.json"

    def load(self, date: dt.date, mode_tag: str | None = None) -> SessionState:
        p = self.path(date, mode_tag)
        if not p.exists():
            return SessionState(date=date.isoformat(), mode=mode_tag or self.mode_tag)
        raw = json.loads(p.read_text())
        known = {f.name for f in dataclasses.fields(SessionState)}
        return SessionState(**{k: v for k, v in raw.items() if k in known})

    def save(self, state: SessionState) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state.mode = self.mode_tag
        p = self.path(dt.date.fromisoformat(state.date))
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2, default=str))
        tmp.replace(p)


def open_position(state: SessionState) -> bool:
    return bool(state.entry and state.entry.get("status") == "filled" and not state.exit)


class Runner:
    def __init__(
        self,
        cfg: Config,
        client,
        provider: MarketDataProvider,
        *,
        live: bool = False,
        dry_run: bool = False,
        clock=None,
    ) -> None:
        if live and dry_run:
            raise ValueError("--live and --dry-run are mutually exclusive")
        if bool(getattr(client, "live", False)) != bool(live):
            raise RuntimeError(
                f"Safety check failed: runner live={live} but client live={getattr(client, 'live', None)}"
            )
        self.cfg = cfg
        self.client = client
        self.provider = provider
        self.live = live
        self.dry_run = dry_run
        self.clock = clock or SystemClock()
        self.mode_tag = "dryrun" if dry_run else ("live" if live else "paper")
        self.store = StateStore(cfg.runner.state_dir, self.mode_tag)
        self.log_dir = Path(cfg.runner.log_dir)
        self.gap = cfg.data.max_bar_gap_minutes

    # ------------------------------------------------------------- helpers
    @property
    def mode(self) -> str:
        if self.dry_run:
            return "DRY-RUN (no orders)"
        return "LIVE (REAL MONEY)" if self.live else "PAPER"

    def banner(self) -> str:
        url = getattr(self.client, "trading_base_url", "?")
        if self.live:
            return (
                "\n" + "!" * 78 + "\n!!  LIVE TRADING ENABLED via --live: orders use REAL MONEY at "
                f"{url}\n!!  0DTE options can expire worthless. This is not financial advice.\n" + "!" * 78
            )
        return f"Mode: {self.mode}; trading endpoint {url}. No live orders (no --live flag)."

    def _append_csv(self, name: str, row: dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / name
        new = not path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)

    def preflight(self) -> None:
        log.info(self.banner())
        acct = self.client.account() or {}
        if acct.get("trading_blocked") or acct.get("account_blocked"):
            raise AlpacaError(f"Alpaca account is blocked from trading: status={acct.get('status')}")
        level = acct.get("options_trading_level")
        if level is None:
            log.warning("Alpaca account did not report options_trading_level; cannot verify options permission.")
        elif int(level) < 2 and not self.dry_run:
            raise AlpacaError(
                f"Alpaca options_trading_level is {level}; buying calls/puts needs level 2. "
                "Enable options trading for this account in the Alpaca dashboard."
            )
        try:
            self.client.stock_latest_bar(self.cfg.strategy.underlying, feed=self.cfg.data.live_stock_feed)
        except AlpacaSubscriptionError as exc:
            raise AlpacaSubscriptionError(
                f"{exc}\n-> Your plan can't stream data.live_stock_feed={self.cfg.data.live_stock_feed!r}. "
                "Set data.live_stock_feed: iex in config.yaml (free plan). Note that IEX volume makes VWAP and the "
                "opening range differ from the SIP data used in backtests."
            ) from exc

    def _warn_other_modes(self, today: dt.date) -> None:
        for tag in ("live", "paper", "dryrun"):
            if tag == self.mode_tag:
                continue
            other = self.store.load(today, tag)
            if open_position(other):
                lvl = logging.CRITICAL if tag == "live" else logging.WARNING
                log.log(
                    lvl,
                    "A %s position from today is still OPEN (%s x %s) and is NOT managed by this %s run. "
                    "Restart with %s to manage it.",
                    tag.upper(), other.entry.get("qty"), other.entry.get("symbol"), self.mode_tag,
                    {"live": "--live", "paper": "no flag", "dryrun": "--dry-run"}[tag],
                )

    # ---------------------------------------------------------- market data
    def _today_bars(self, day: TradingDay, t: pd.Timestamp, *, wait: bool = True) -> pd.DataFrame:
        """Today's completed regular-session bars up to time t (waits briefly for the bar ending at t)."""
        deadline = t + pd.Timedelta(seconds=self.cfg.runner.bar_wait_max_seconds)
        while True:
            raw = self.client.stock_bars(
                self.cfg.strategy.underlying, "1Min", day.open_ts, t,
                feed=self.cfg.data.live_stock_feed, adjustment=self.cfg.data.adjustment,
            )
            bars = regular_session(bars_from_alpaca(raw), day)
            bars = bars.loc[bars.index <= t - ONE_MINUTE]
            if not bars.empty and bars.index[-1] == t - ONE_MINUTE:
                return bars
            if not wait or self.clock.now() >= deadline:
                if bars.empty:
                    raise DataUnavailable(f"Alpaca returned no {self.cfg.strategy.underlying} bars up to {hhmm(t)}")
                log.warning(
                    "Bar ending %s not available after waiting; last bar ends %s (no trades that minute or feed lag)",
                    hhmm(t), hhmm(bars.index[-1] + ONE_MINUTE),
                )
                return bars
            self.clock.sleep(2.0)

    def _option_last_price(self, symbol: str) -> float:
        trades = self.client.option_latest_trades([symbol], feed=self.cfg.data.option_feed)
        px = (trades.get(symbol) or {}).get("p")
        if px is None:
            raise DataUnavailable(
                f"Alpaca returned no latest trade for {symbol} (feed={self.cfg.data.option_feed}); cannot mark price"
            )
        return float(px)

    def _live_strikes(self, day: TradingDay, side: str, price: float) -> list[float]:
        root = self.cfg.strategy.underlying
        prefix = f"{root}{day.date:%y%m%d}{side[0].upper()}"
        contracts = self.client.option_contracts(
            underlying=root, expiration_date=day.date, contract_type=side, status="active",
            strike_gte=math.floor(price - 30), strike_lte=math.ceil(price + 30),
        )
        strikes = sorted(
            {float(c["strike_price"]) for c in contracts if str(c.get("symbol", "")).startswith(prefix) and c.get("tradable", True)}
        )
        if not strikes:
            raise DataUnavailable(
                f"Alpaca /v2/options/contracts returned no active tradable {root} {side} contracts expiring "
                f"{day.date} near {price:.2f}"
            )
        return strikes

    def _wait_fill(self, order: dict) -> dict:
        """Poll until the order is terminal. Returns the final order (filled_qty may be < qty or 0)."""
        deadline = self.clock.now() + pd.Timedelta(seconds=self.cfg.runner.order_fill_timeout_seconds)
        current = order
        while str(current.get("status")) not in TERMINAL_ORDER_STATES and self.clock.now() < deadline:
            self.clock.sleep(1.0)
            current = self.client.get_order(order["id"])
        if str(current.get("status")) not in TERMINAL_ORDER_STATES:
            log.error("Order %s still %s after timeout; cancelling the remainder", order["id"], current.get("status"))
            self.client.cancel_order(order["id"])
            current = self.client.get_order(order["id"])
        filled = float(current.get("filled_qty") or 0)
        if str(current.get("status")) != "filled":
            log.error("Order %s ended %s with %s filled", order["id"], current.get("status"), filled)
        return current

    # ------------------------------------------------------------- session
    def run_session(self) -> SessionState | None:
        cfg, rules, a = self.cfg, self.cfg.strategy, self.cfg.assumptions
        now = self.clock.now()
        today = now.date()
        calendar = self.provider.trading_days(today - dt.timedelta(days=70), today)
        if not calendar or calendar[-1].date != today:
            log.info("%s is not a trading day. Nothing to do.", today)
            return None
        day, history = calendar[-1], calendar[:-1]
        self._warn_other_modes(today)
        if day.is_early_close and a.early_close_policy == "skip":
            log.warning("%s closes early (%s). early_close_policy=skip, so no trading today.", today, day.close_time)
            return None
        if now >= day.close_ts:
            log.info("Session %s already closed.", today)
            return None

        state = self.store.load(today)
        if state.exit:
            log.info("Already traded today (one trade per day): %s", state.exit)
            return state
        if state.entry and state.entry.get("status") == "submitted":
            self._resolve_submitted_entry(state)
        if state.first_signal and not open_position(state):
            log.info("Today's first signal was already seen (%s); no (further) trade today.", state.first_signal)
            return state

        if not state.entry:
            lookback = history[-rules.noise_lookback_days:]
            sessions = [(d, self.provider.minute_bars(d)) for d in lookback]
            daily = None
            if a.atr_source == "alpaca_daily":
                daily = self.provider.daily_bars(history[-(rules.atr_period_days + 1)].date, history[-1].date)
            atr = atr_for_day(day, history, rules, a, daily_bars=daily, minute_bars_fn=self.provider.minute_bars)
            log.info("%d-day ATR = %.4f; noise lookback %s..%s", rules.atr_period_days, atr, lookback[0].date, lookback[-1].date)

            settle = pd.Timedelta(seconds=cfg.runner.bar_settle_seconds)
            or_end = et_timestamp(today, rules.session_open_time) + pd.Timedelta(minutes=rules.opening_range_minutes)
            self.clock.sleep_until(or_end + settle)
            setup = build_day_setup(day, self._today_bars(day, or_end), sessions, atr, rules, a)
            log.info("Setup: %s", setup.describe(rules))

            late = pd.Timedelta(seconds=cfg.runner.late_check_tolerance_seconds)
            signal: CheckResult | None = None
            replayed = False
            for t in check_times(day, rules, a):
                replayed = self.clock.now() > t + late
                if not replayed:
                    self.clock.sleep_until(t + settle)
                bars = self._today_bars(day, t, wait=not replayed)
                setup = self._refresh_opening_range(setup, bars, rules)
                res = evaluate_check(setup, bars, t, rules, a, self.gap)
                log.info("%s%s", "[replay of missed check] " if replayed else "", res.log_line())
                self._append_csv(f"checks_{today.isoformat()}.csv", {**res.as_row(), "replayed": replayed, "mode": self.mode})
                if res.signal:
                    signal = res
                    break
            if signal is None:
                log.info("No signal between %s and %s. No trade today.", rules.entry_window_start, rules.entry_window_end)
                state.notes.append("no signal")
                self.store.save(state)
                return state
            state.first_signal = {"time": signal.check_time.isoformat(), "side": signal.signal, "close": signal.close, "replayed": replayed}
            self.store.save(state)
            if replayed:
                log.warning(
                    "Today's FIRST signal (%s at %s) happened before the runner was running. First-signal-only rule, "
                    "so no trade today.", signal.signal.upper(), hhmm(signal.check_time),
                )
                return state
            self._enter(day, setup.atr, signal, state)

        if open_position(state):
            self._manage(day, state)
        return state

    def _refresh_opening_range(self, setup: DaySetup, bars: pd.DataFrame, rules) -> DaySetup:
        """The opening range is rebuilt from the (by now complete) 9:30-9:44 bars at every check, so a bar
        that arrived late at 9:45 can't leave the range wrong for the rest of the day."""
        hi, lo, n = opening_range(bars, setup.open_ts, rules.opening_range_minutes)
        if (hi, lo, n) != (setup.or_high, setup.or_low, setup.or_bars):
            log.warning(
                "Opening range updated from late bars: [%.2f, %.2f] (%d bars) -> [%.2f, %.2f] (%d bars)",
                setup.or_low, setup.or_high, setup.or_bars, lo, hi, n,
            )
            return dataclasses.replace(setup, or_high=hi, or_low=lo, or_bars=n)
        return setup

    def run_forever(self) -> None:
        while True:
            try:
                self.run_session()
            except (DataUnavailable, AlpacaError) as exc:
                log.error("Session failed: %s", exc)
            now = self.clock.now()
            upcoming = self.provider.trading_days(now.date() + dt.timedelta(days=1), now.date() + dt.timedelta(days=14))
            if not upcoming:
                raise DataUnavailable("No upcoming trading days in the calendar")
            wake = et_timestamp(upcoming[0].date, "09:20")
            log.info("Sleeping until %s", wake)
            self.clock.sleep_until(wake)

    # ---------------------------------------------------------------- entry
    def _enter(self, day: TradingDay, atr: float, signal: CheckResult, state: SessionState) -> None:
        rules, a = self.cfg.strategy, self.cfg.assumptions
        side = signal.signal
        strikes = self._live_strikes(day, side, signal.close)
        strike = select_strike(signal.close, side, strikes, method=a.strike_selection, offset=a.strike_offset, tie_break=a.atm_tie_break)
        symbol = occ_symbol(rules.underlying, day.date, side, strike)
        root, expiry, right, _ = parse_occ(symbol)
        if root != rules.underlying or expiry != day.date or right != side:
            raise RuntimeError(f"Safety check failed: {symbol} is not a same-day {rules.underlying} {side}")
        qty = a.contracts
        log.info(
            "SIGNAL %s at %s (QQQ %.2f). BUY %d x %s (strike %g, %s) [%s]",
            side.upper(), hhmm(signal.check_time), signal.close, qty, symbol, strike, a.strike_selection, self.mode,
        )
        client_order_id = f"qqq0dte-{day.date.isoformat()}-{self.mode_tag}-entry"
        # Persist intent BEFORE sending, so a crash mid-submit can be resolved on restart (never re-sent).
        state.entry = {
            "status": "submitted", "side": side, "symbol": symbol, "strike": strike, "qty": qty,
            "signal_time": signal.check_time.isoformat(), "entry_underlying": signal.close, "atr": atr,
            "fill_price": None, "order_id": None, "client_order_id": client_order_id,
            "simulated": self.dry_run, "mode": self.mode,
        }
        self.store.save(state)
        if self.dry_run:
            px = self._option_last_price(symbol)
            state.entry.update(status="filled", fill_price=px)
            log.info("DRY-RUN: no order sent. Marked entry at latest trade %.2f", px)
        else:
            order = self.client.submit_order(
                symbol=symbol, qty=qty, side="buy", order_type="market", time_in_force="day",
                client_order_id=client_order_id,
            )
            state.entry["order_id"] = order.get("id")
            self.store.save(state)
            self._record_entry_fill(state, self._wait_fill(order))
        self.store.save(state)

    def _record_entry_fill(self, state: SessionState, filled: dict) -> None:
        qty = int(float(filled.get("filled_qty") or 0))
        if qty <= 0 or not filled.get("filled_avg_price"):
            state.entry.update(status=f"not_filled:{filled.get('status')}")
            log.error("Entry order %s did not fill (%s). No trade today (first signal only).", filled.get("id"), filled.get("status"))
            return
        state.entry.update(status="filled", qty=qty, fill_price=float(filled["filled_avg_price"]), order_id=filled.get("id"))
        log.info("ENTRY FILLED: %d x %s @ %.2f (order %s)", qty, state.entry["symbol"], state.entry["fill_price"], filled.get("id"))

    def _resolve_submitted_entry(self, state: SessionState) -> None:
        """After a restart: find out whether the entry order that was being sent actually exists."""
        e = state.entry
        if e.get("simulated"):
            state.entry.update(status="not_filled:dry_run_interrupted")
        else:
            order = self.client.get_order_by_client_order_id(e["client_order_id"])
            if not order:
                log.warning("Entry %s was never received by Alpaca. No trade today.", e["client_order_id"])
                state.entry.update(status="not_filled:not_found")
            else:
                if str(order.get("status")) not in TERMINAL_ORDER_STATES:
                    order = self._wait_fill(order)
                self._record_entry_fill(state, order)
        self.store.save(state)

    # -------------------------------------------------------------- manage
    def _manage(self, day: TradingDay, state: SessionState) -> None:
        rules, a = self.cfg.strategy, self.cfg.assumptions
        e = state.entry
        entry_time = pd.Timestamp(e["signal_time"]).tz_convert(ET)
        stop = TrailingStop(
            side=e["side"], entry_time=entry_time, entry_underlying=float(e["entry_underlying"]), atr=float(e["atr"]),
            entry_option_price=float(e["fill_price"]), arm_threshold=rules.trail_arm_threshold,
            giveback_fraction=rules.trail_giveback_fraction,
        )
        if state.trail:
            stop.restore(state.trail)
        if state.pending_exit:  # an exit was already decided before a restart: finish it, don't re-decide
            p = state.pending_exit
            self._exit(day, state, stop, p["reason"], pd.Timestamp(p["time"]).tz_convert(ET))
            return
        hard_exit = et_timestamp(day.date, rules.hard_exit_time)
        step = pd.Timedelta(minutes=a.trail_check_interval_minutes)
        settle = pd.Timedelta(seconds=self.cfg.runner.bar_settle_seconds)
        if stop.needs_option_price:
            # Only the latest option trade is available live, so missed minutes can't be replayed.
            t = max(entry_time + step, self.clock.now().ceil("min"))
        else:
            # Resume right after the last observed minute; minutes missed during a restart are replayed
            # from QQQ's 1-minute bars (sleep_until returns at once for past times).
            t = (stop.last_time + step) if stop.last_time is not None else entry_time + step
        while t < hard_exit:
            self.clock.sleep_until(t + settle)
            if self.clock.now() >= hard_exit:
                break
            try:
                und = price_at(self._today_bars(day, t), t, self.gap)
                mark = self._option_last_price(e["symbol"]) if stop.needs_option_price else None
            except (DataUnavailable, AlpacaError) as exc:
                # Never abandon an open position over a data hiccup: skip this minute, keep managing.
                log.error("Trailing-stop observation at %s skipped: %s", hhmm(t), exc)
                t += step
                continue
            was_armed = stop.armed
            fire = stop.update(t, underlying=und, option_price=mark)
            state.trail = stop.to_state()
            self.store.save(state)
            if stop.armed and not was_armed:
                log.info("TRAILING STOP ARMED at %s: %s", hhmm(t), stop.describe())
            if fire:
                self._exit(day, state, stop, "trailing_stop", t)
                return
            t += step
        self.clock.sleep_until(hard_exit)
        self._exit(day, state, stop, f"hard_exit_{rules.hard_exit_time}", hard_exit)

    # ---------------------------------------------------------------- exit
    def _close_position(self, day: TradingDay, state: SessionState, symbol: str) -> list[dict]:
        """Sell the whole position at market, retrying until it is flat or the market closes.

        Before every (re)send it checks earlier exit orders and the live position, so it never
        sells more than is held. Returns the exit orders that filled anything.
        """
        attempt = len(state.exit_orders)
        while True:
            try:
                fills = []
                for cid in state.exit_orders:
                    o = self.client.get_order_by_client_order_id(cid)
                    if o and str(o.get("status")) not in TERMINAL_ORDER_STATES:
                        o = self._wait_fill(o)
                    if o and float(o.get("filled_qty") or 0) > 0:
                        fills.append(o)
                pos = self.client.get_position(symbol)
                held = int(abs(float(pos.get("qty", 0)))) if pos else 0
                if held == 0:
                    return fills
                attempt += 1
                cid = f"qqq0dte-{day.date.isoformat()}-{self.mode_tag}-exit-{attempt}"
                state.exit_orders.append(cid)
                self.store.save(state)
                order = self.client.submit_order(
                    symbol=symbol, qty=held, side="sell", order_type="market", time_in_force="day", client_order_id=cid,
                )
                self._wait_fill(order)
            except AlpacaError as exc:
                log.error("EXIT attempt %d for %s failed: %s", attempt, symbol, exc)
                if self.clock.now() >= day.close_ts:
                    log.critical("COULD NOT CLOSE %s BEFORE THE CLOSE. MANUAL ACTION REQUIRED.", symbol)
                    raise
                self.clock.sleep(EXIT_RETRY_SECONDS)

    def _exit(self, day: TradingDay, state: SessionState, stop: TrailingStop, reason: str, t: pd.Timestamp) -> None:
        e = state.entry
        symbol = e["symbol"]
        if not state.pending_exit:
            state.pending_exit = {"reason": reason, "time": t.isoformat()}
            state.trail = stop.to_state()
            self.store.save(state)
        log.info("EXIT decided (%s) at %s for %s [%s]", reason, hhmm(t), symbol, self.mode)
        entry_px = float(e["fill_price"])
        if self.dry_run:
            try:
                px = self._option_last_price(symbol)
            except (DataUnavailable, AlpacaError) as exc:
                px = math.nan
                log.error("DRY-RUN: could not mark exit price: %s", exc)
            qty, order_ids = int(e["qty"]), []
            log.info("DRY-RUN: no order sent. Marked exit at latest trade %.2f", px)
        else:
            fills = self._close_position(day, state, symbol)  # SELL FIRST; everything below is bookkeeping
            qty = sum(int(float(o.get("filled_qty") or 0)) for o in fills)
            notional = sum(float(o["filled_avg_price"]) * float(o["filled_qty"]) for o in fills if o.get("filled_avg_price"))
            px = notional / qty if qty else math.nan
            order_ids = [o.get("id") for o in fills]
            if not fills:
                log.error("EXIT %s: no open position in %s and no exit fills (closed manually?).", reason, symbol)
            elif qty != int(e["qty"]):
                log.error("EXIT sold %d contracts but entry was %s; CHECK THE ACCOUNT", qty, e["qty"])
        try:
            und = price_at(self._today_bars(day, t, wait=False), t, self.gap)
            if reason != "trailing_stop":
                stop.mark_exit(t, underlying=und, option_price=None if not stop.needs_option_price else px)
        except (DataUnavailable, AlpacaError) as exc:
            und = math.nan
            log.warning("Could not read QQQ at the exit for logging: %s", exc)
        pnl = (px - entry_px) * 100 * qty if qty and not math.isnan(px) else math.nan
        state.exit = {
            "reason": reason, "time": t.isoformat(), "sent_at": self.clock.now().isoformat(),
            "fill_price": None if math.isnan(px) else px, "order_ids": order_ids, "qty": qty,
            "pnl": None if math.isnan(pnl) else round(pnl, 2),
            "pnl_pct": None if math.isnan(px) or not entry_px else round((px / entry_px - 1) * 100, 2),
            "underlying": None if math.isnan(und) else und, "trail": stop.to_state(),
            "trail_exit_value": stop.exit_value, "trail_retracement_from_peak": stop.retracement_from_peak,
        }
        state.trail = stop.to_state()
        self.store.save(state)
        log.info("EXIT (%s) %d x %s @ %.2f; entry %.2f; P&L %+.2f. Trailing: %s", reason, qty, symbol, px, entry_px, pnl, stop.describe())
        self._append_csv(
            "trades.csv",
            {
                "date": day.date.isoformat(), "mode": self.mode, "side": e["side"], "signal_time": hhmm(pd.Timestamp(e["signal_time"])),
                "strike": e["strike"], "option_symbol": symbol, "contracts": qty, "entry_price": entry_px,
                "exit_time": hhmm(t), "exit_price": px, "exit_reason": reason, "pnl": state.exit["pnl"],
                "trail_basis": stop.basis, "trail_peak": stop.peak, "trail_peak_time": hhmm(stop.peak_time),
                "trail_armed": stop.armed, "trail_armed_time": hhmm(stop.armed_time) if stop.armed_time is not None else "",
                "trail_armed_value": stop.armed_value, "trail_exit_value": stop.exit_value,
                "trail_retracement_from_peak": stop.retracement_from_peak,
            },
        )
