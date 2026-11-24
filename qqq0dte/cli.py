"""Command line: backtest | validate | run | assumptions."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

from .alpaca import AlpacaClient, AlpacaError
from .config import ConfigError, assumption_report, load_config, log_assumptions
from .data import AlpacaDataProvider, DataUnavailable
from .timeutil import ET, now_et, parse_date

log = logging.getLogger("qqq0dte")

DISCLAIMER = (
    "Educational code that implements a social-media creator's described rules. Not financial advice. "
    "0DTE options can go to zero."
)


class ETFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # noqa: N802
        ts = pd.Timestamp(record.created, unit="s", tz="UTC").tz_convert(ET)
        return ts.strftime(datefmt or "%Y-%m-%d %H:%M:%S ET")


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = ETFormatter("%(asctime)s %(levelname)-7s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _provider(cfg, *, offline: bool, feed: str | None = None, client=None) -> AlpacaDataProvider:
    if client is None and not offline:
        client = AlpacaClient.from_env(live=False)
    return AlpacaDataProvider(
        client,
        underlying=cfg.strategy.underlying,
        stock_feed=feed or cfg.data.stock_feed,
        adjustment=cfg.data.adjustment,
        cache_dir=cfg.data.cache_dir,
        strike_grid_fallback=cfg.data.strike_grid_fallback,
        offline=offline,
        calendar_source=cfg.data.calendar_source,
        strike_source=cfg.data.strike_source,
    )


def cmd_backtest(args, cfg) -> int:
    from .backtest import Backtester
    from .report import format_summary, summarize, write_outputs

    start, end = parse_date(args.start), parse_date(args.end)
    out_dir = Path(args.out or Path(cfg.backtest.output_dir) / f"backtest_{start}_{end}")
    setup_logging(args.verbose, out_dir / "backtest.log")
    log.info(DISCLAIMER)
    log_assumptions(cfg, log)
    result = Backtester(cfg, _provider(cfg, offline=args.offline)).run(start, end)
    paths = write_outputs(result, cfg, out_dir)
    print()
    print(format_summary(summarize(result)))
    print(f"\nOutputs written to {out_dir}/ ({', '.join(p.name for p in paths.values())})")
    return 0


def cmd_validate(args, cfg) -> int:
    from .validate import CREATOR_EXAMPLE, CreatorExample, run_validation

    out_dir = Path(args.out or Path(cfg.backtest.output_dir) / "validation")
    setup_logging(args.verbose, out_dir / "validate.log")
    log.info(DISCLAIMER)
    log_assumptions(cfg, log)
    example = CREATOR_EXAMPLE if args.date is None else CreatorExample(date=parse_date(args.date))
    text, code, _ = run_validation(cfg, _provider(cfg, offline=args.offline), example=example, sensitivity=not args.no_sensitivity)
    print(text)
    (out_dir / "validation.txt").write_text(text + "\n")
    return code


def cmd_run(args, cfg) -> int:
    from .runner import Runner

    today = now_et().date()
    setup_logging(args.verbose, Path(cfg.runner.log_dir) / f"run_{today.isoformat()}.log")
    if args.live and args.dry_run:
        log.error("--live and --dry-run are mutually exclusive")
        return 2
    log.info(DISCLAIMER)
    log_assumptions(cfg, log)
    # The ONLY place a live client can be created: the explicit --live flag.
    # Short timeouts/retries: the exit loop retries on its own, and a 3:45 sell must not wait minutes on one call.
    client = AlpacaClient.from_env(live=args.live, live_confirmed=args.live, timeout=10, max_retries=2)
    provider = _provider(cfg, offline=False, feed=cfg.data.live_stock_feed, client=client)
    runner = Runner(cfg, client, provider, live=args.live, dry_run=args.dry_run)
    runner.preflight()
    if args.loop:
        runner.run_forever()
    else:
        runner.run_session()
    return 0


def cmd_assumptions(args, cfg) -> int:
    print("\n".join(assumption_report(cfg)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m qqq0dte", description="QQQ 0DTE opening-range/noise-band/VWAP strategy. " + DISCLAIMER)
    p.add_argument("--config", default="config.yaml", help="path to config.yaml (default: ./config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="run the backtest on Alpaca historical data")
    bt.add_argument("--start", required=True, help="YYYY-MM-DD (Alpaca options history starts 2024-02)")
    bt.add_argument("--end", default=None, help="YYYY-MM-DD (default: yesterday)")
    bt.add_argument("--out", default=None, help="output directory")
    bt.add_argument("--offline", action="store_true", help="use only the local data cache (no network)")
    bt.set_defaults(func=cmd_backtest)

    va = sub.add_parser("validate", help="compare against the creator's Nov 20, 2026 trade")
    va.add_argument("--date", default=None, help="override the date (default 2026-11-20)")
    va.add_argument("--out", default=None)
    va.add_argument("--offline", action="store_true")
    va.add_argument("--no-sensitivity", action="store_true", help="skip the assumption-sensitivity table")
    va.set_defaults(func=cmd_validate)

    rn = sub.add_parser("run", help="paper-trade today's session (PAPER by default)")
    rn.add_argument("--dry-run", action="store_true", help="log signals only, send no orders at all")
    rn.add_argument("--live", action="store_true", help="REAL MONEY: send orders to the live Alpaca account")
    rn.add_argument("--loop", action="store_true", help="keep running, one session per trading day")
    rn.set_defaults(func=cmd_run)

    asm = sub.add_parser("assumptions", help="print every rule and assumption in effect")
    asm.set_defaults(func=cmd_assumptions)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "command", None) == "backtest" and args.end is None:
        args.end = (now_et().date() - dt.timedelta(days=1)).isoformat()
    try:
        cfg = load_config(args.config)
        return int(args.func(args, cfg) or 0)
    except (ConfigError, AlpacaError, DataUnavailable) as exc:
        logging.getLogger("qqq0dte").error("%s: %s", type(exc).__name__, exc)
        if not logging.getLogger().handlers:
            print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
