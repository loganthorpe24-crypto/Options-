"""Config loading, validation, and the assumption registry.

Every setting the creator did not specify is listed in ASSUMPTION_NOTES (config
keys) or FIXED_ASSUMPTIONS (behavior in code). Both are logged at every start
by `log_assumptions`.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from . import exits
from .timeutil import parse_hhmm

log = logging.getLogger(__name__)


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Creator rules. The defaults ARE the creator's values.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyRules:
    underlying: str = "QQQ"
    session_open_time: str = "09:30"
    opening_range_minutes: int = 15
    noise_lookback_days: int = 20
    noise_multiplier: float = 1.25
    atr_period_days: int = 5
    atr_breakout_multiplier: float = 0.3
    entry_window_start: str = "10:00"
    entry_window_end: str = "14:00"
    check_interval_minutes: int = 10
    hard_exit_time: str = "15:45"
    trail_arm_threshold: float = 4.0
    trail_giveback_fraction: float = 0.5


CREATOR_RULES: dict[str, Any] = dataclasses.asdict(StrategyRules())


# --------------------------------------------------------------------------- #
# Assumptions: things the creator left out.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Assumptions:
    strike_selection: str = "atm"
    strike_offset: int = 0
    atm_tie_break: str = "lower"
    contracts: int = 10
    session_open_price: str = "first_bar_open"
    noise_move_units: str = "dollars"
    vwap_price: str = "hlc3"
    atr_source: str = "alpaca_daily"
    entry_window_end_inclusive: bool = True
    trail_check_interval_minutes: int = 1
    fill_model: str = "next_bar_open"
    option_fill_max_staleness_minutes: int = 5
    slippage_per_share: float = 0.0
    commission_per_contract: float = 0.0
    early_close_policy: str = "skip"


@dataclass(frozen=True)
class DataSettings:
    stock_feed: str = "sip"
    live_stock_feed: str = "sip"
    option_feed: str = "indicative"
    adjustment: str = "raw"
    cache_dir: str = "data_cache"
    strike_grid_fallback: float | None = 1.0
    max_bar_gap_minutes: int = 5
    calendar_source: str = "alpaca"
    strike_source: str = "alpaca_contracts"


@dataclass(frozen=True)
class RunnerSettings:
    log_dir: str = "logs"
    state_dir: str = "state"
    bar_settle_seconds: float = 3.0
    bar_wait_max_seconds: float = 45.0
    late_check_tolerance_seconds: float = 120.0
    order_fill_timeout_seconds: float = 60.0


@dataclass(frozen=True)
class BacktestSettings:
    output_dir: str = "output"


@dataclass(frozen=True)
class Config:
    strategy: StrategyRules = StrategyRules()
    assumptions: Assumptions = Assumptions()
    data: DataSettings = DataSettings()
    runner: RunnerSettings = RunnerSettings()
    backtest: BacktestSettings = BacktestSettings()
    source_path: str | None = None

    @property
    def contracts(self) -> int:
        return self.assumptions.contracts


CHOICES: dict[tuple[str, str], set[str]] = {
    ("assumptions", "strike_selection"): {"atm", "itm", "otm"},
    ("assumptions", "atm_tie_break"): {"lower", "higher"},
    ("assumptions", "session_open_price"): {"first_bar_open"},
    ("assumptions", "noise_move_units"): {"dollars", "percent"},
    ("assumptions", "vwap_price"): {"hlc3", "bar_vwap", "close"},
    ("assumptions", "atr_source"): {"alpaca_daily", "rth_from_minutes"},
    ("assumptions", "fill_model"): {"next_bar_open", "signal_bar_close"},
    ("assumptions", "early_close_policy"): {"skip"},
    ("data", "stock_feed"): {"sip", "iex", "delayed_sip"},
    ("data", "live_stock_feed"): {"sip", "iex", "delayed_sip"},
    ("data", "option_feed"): {"indicative", "opra"},
    ("data", "calendar_source"): {"alpaca", "pandas_market_calendars"},
    ("data", "strike_source"): {"alpaca_contracts", "grid"},
    ("data", "adjustment"): {"raw", "split", "dividend", "all"},
}

# Human-readable description of every config-level assumption (logged at startup).
ASSUMPTION_NOTES: dict[str, str] = {
    "assumptions.strike_selection": "STRIKE_SELECTION. Creator gives no strike rule. atm = listed 0DTE strike closest to QQQ at the signal close.",
    "assumptions.strike_offset": "Strikes to move ITM/OTM from ATM (only with strike_selection itm/otm).",
    "assumptions.atm_tie_break": "If QQQ is exactly halfway between two strikes, which one ATM picks.",
    "assumptions.contracts": "Position size. Creator gives no sizing rule (his example used 10 contracts).",
    "assumptions.session_open_price": "'The open' for the noise band = open of the 9:30 one-minute bar.",
    "assumptions.noise_move_units": "Noise-band move measured in dollars (|price-open|) or percent (|price/open-1|).",
    "assumptions.vwap_price": "Per-bar price used in the session VWAP from 1-minute bars.",
    "assumptions.atr_source": "Where the daily bars for the 5-day ATR come from.",
    "assumptions.entry_window_end_inclusive": "Whether the 2:00pm check is part of the 10:00-2:00 window.",
    "assumptions.trail_check_interval_minutes": "How often the trailing stop is evaluated (on QQQ 1-minute closes).",
    "assumptions.fill_model": "Backtest fill price for option entries and exits (from Alpaca 1-minute trade bars).",
    "assumptions.option_fill_max_staleness_minutes": "Max minutes to the nearest option trade bar before a fill is a data error.",
    "assumptions.slippage_per_share": "Backtest slippage per share (worse entry and worse exit).",
    "assumptions.commission_per_contract": "Backtest commission per contract per side.",
    "assumptions.early_close_policy": "Half days (1pm close) can't fit the 3:45 exit or the 2:00 window, so they are skipped.",
    "data.strike_grid_fallback": "If Alpaca lists no contracts for a past expiry, assume this strike spacing ($). The contract must still have real trade bars.",
    "data.max_bar_gap_minutes": "If the newest QQQ bar at a check / trailing-stop minute is older than this, it's a data error (no stale prices).",
    "data.calendar_source": "Trading days / half days: Alpaca /v2/calendar, or the NYSE calendar from the pandas_market_calendars package.",
    "data.strike_source": "Strike list: Alpaca /v2/options/contracts, or a grid of data.strike_grid_fallback spacing (the contract must still have real Alpaca trade bars).",
    "data.stock_feed": "Alpaca stock feed for backtest bars (sip = consolidated tape).",
    "data.live_stock_feed": "Alpaca stock feed for the runner (free plans: iex; IEX volume makes VWAP differ from SIP).",
    "runner.late_check_tolerance_seconds": "Checks the runner evaluates later than this are replay-only. If one was the first signal, no trade.",
}

# Behavior that is fixed in code but still not specified by the creator.
FIXED_ASSUMPTIONS: list[str] = [
    "Price 'at' a check time T = close of the last completed 1-minute bar ending at or before T "
    "(the 10:30 check uses the close of the 10:29-10:30 bar). VWAP and bands use the same cut-off.",
    "Opening range = max high / min low of the 1-minute bars starting 9:30 through 9:44.",
    "The noise-band average excludes today and uses the 20 completed trading days before the trade date. "
    "Minutes with no trade carry the last close forward. On early-close days in the lookback, "
    "minutes after the close are left out of that minute's average.",
    "5-day ATR uses the 5 completed daily bars before the trade date (needs 6 bars for the first True Range). "
    "Today's partial bar is never used.",
    "VWAP is cumulative over regular-session bars from 9:30 (no pre-market).",
    "Entry and trailing-stop exits are MARKET orders, time_in_force=day. The creator only says 'at market' for the 3:45 exit.",
    "The 3:45pm hard exit takes priority. The trailing stop is only evaluated strictly before 3:45pm.",
    "Backtest option fills come from Alpaca historical 1-minute TRADE bars (not bid/ask quotes), so the "
    "spread is only modeled through slippage_per_share.",
    "If the runner starts late and a missed check was the day's first signal, no trade is taken (first signal only).",
]


def _coerce(value: Any, default: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{key} must be true/false, got {value!r}")
        return value
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, bool) or float(value) != int(float(value)):
            raise ConfigError(f"{key} must be an integer, got {value!r}")
        return int(float(value))
    if isinstance(default, float) or default is None:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{key} must be a number, got {value!r}") from exc
    return str(value)


def _build(cls, raw: Any, section: str):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{section}]: {', '.join(unknown)} (typo?)")
    defaults = cls()
    kwargs = {}
    for name in known:
        if name in raw:
            kwargs[name] = _coerce(raw[name], getattr(defaults, name), f"{section}.{name}")
    return cls(**kwargs)


def validate(cfg: Config) -> None:
    for (section, key), allowed in CHOICES.items():
        value = getattr(getattr(cfg, section), key)
        if value not in allowed:
            raise ConfigError(f"{section}.{key} = {value!r} is not one of {sorted(allowed)}")
    s, a = cfg.strategy, cfg.assumptions
    for key in ("session_open_time", "entry_window_start", "entry_window_end", "hard_exit_time"):
        parse_hhmm(getattr(s, key))
    if parse_hhmm(s.entry_window_start) > parse_hhmm(s.entry_window_end):
        raise ConfigError("entry_window_start is after entry_window_end")
    if parse_hhmm(s.entry_window_end) >= parse_hhmm(s.hard_exit_time):
        raise ConfigError("entry_window_end must be before hard_exit_time")
    positives = {
        "strategy.opening_range_minutes": s.opening_range_minutes,
        "strategy.noise_lookback_days": s.noise_lookback_days,
        "strategy.atr_period_days": s.atr_period_days,
        "strategy.check_interval_minutes": s.check_interval_minutes,
        "assumptions.contracts": a.contracts,
        "assumptions.trail_check_interval_minutes": a.trail_check_interval_minutes,
        "assumptions.option_fill_max_staleness_minutes": a.option_fill_max_staleness_minutes,
        "data.max_bar_gap_minutes": cfg.data.max_bar_gap_minutes,
    }
    for key, value in positives.items():
        if value <= 0:
            raise ConfigError(f"{key} must be > 0, got {value}")
    if not 0 < s.trail_giveback_fraction <= 1:
        raise ConfigError("strategy.trail_giveback_fraction must be in (0, 1]")
    if a.strike_selection == "atm" and a.strike_offset != 0:
        raise ConfigError("strike_offset must be 0 when strike_selection is 'atm'")
    if a.strike_offset < 0:
        raise ConfigError("strike_offset must be >= 0 (choose itm or otm for direction)")
    if cfg.data.strike_grid_fallback is not None and cfg.data.strike_grid_fallback <= 0:
        raise ConfigError("data.strike_grid_fallback must be > 0 or null")
    if cfg.data.strike_source == "grid" and not cfg.data.strike_grid_fallback:
        raise ConfigError("data.strike_source: grid needs data.strike_grid_fallback set to the strike spacing")


def load_config(path: str | Path | None = "config.yaml") -> Config:
    if path is None:
        cfg = Config()
    else:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p.resolve()}")
        raw = yaml.safe_load(p.read_text()) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{p} must contain a YAML mapping")
        allowed = {"strategy", "assumptions", "data", "runner", "backtest"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ConfigError(f"Unknown top-level section(s) in {p}: {', '.join(unknown)}")
        cfg = Config(
            strategy=_build(StrategyRules, raw.get("strategy"), "strategy"),
            assumptions=_build(Assumptions, raw.get("assumptions"), "assumptions"),
            data=_build(DataSettings, raw.get("data"), "data"),
            runner=_build(RunnerSettings, raw.get("runner"), "runner"),
            backtest=_build(BacktestSettings, raw.get("backtest"), "backtest"),
            source_path=str(p.resolve()),
        )
    validate(cfg)
    return cfg


def assumption_report(cfg: Config) -> list[str]:
    lines = [f"Config: {cfg.source_path or '(built-in defaults)'}", "Creator rules in effect:"]
    for key, value in dataclasses.asdict(cfg.strategy).items():
        note = "" if value == CREATOR_RULES[key] else f"   <-- CHANGED (creator: {CREATOR_RULES[key]!r})"
        lines.append(f"  strategy.{key} = {value!r}{note}")
    lines.append("  one trade per day, first signal only, no re-entry (hard-coded)")
    lines.append("ASSUMPTIONS (the creator did not specify these):")
    for path, note in ASSUMPTION_NOTES.items():
        section, key = path.split(".")
        lines.append(f"  {path} = {getattr(getattr(cfg, section), key)!r}  # {note}")
    lines.append(
        f"  TRAIL_MEASUREMENT_BASIS = {exits.TRAIL_MEASUREMENT_BASIS!r}  "
        f"# constant at top of qqq0dte/exits.py: {exits.BASIS_DESCRIPTIONS[exits.TRAIL_MEASUREMENT_BASIS]}"
    )
    lines.append("Fixed behavior (not specified by the creator):")
    lines.extend(f"  - {item}" for item in FIXED_ASSUMPTIONS)
    return lines


def log_assumptions(cfg: Config, logger: logging.Logger | None = None) -> None:
    logger = logger or log
    for line in assumption_report(cfg):
        logger.info(line)
    for key, value in dataclasses.asdict(cfg.strategy).items():
        if value != CREATOR_RULES[key]:
            logger.warning(
                "strategy.%s = %r differs from the creator's rule (%r). This is no longer his strategy.",
                key, value, CREATOR_RULES[key],
            )
