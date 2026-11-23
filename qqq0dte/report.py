"""Backtest summary statistics and output files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .config import Config, assumption_report


def summarize(result: BacktestResult) -> dict:
    trades = result.trades
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    if len(pnls):
        equity = np.cumsum(pnls)
        running_peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
        max_dd = float(min((equity - running_peak).min(), 0.0))
    else:
        max_dd = 0.0
    status = Counter(d.status for d in result.days)
    no_trade_days = [d for d in result.days if d.status != "traded"]

    def r(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 2)

    return {
        "period_start": result.start.isoformat(),
        "period_end": result.end.isoformat(),
        "trading_days": len(result.days),
        "trades": len(trades),
        "put_trades": sum(t.side == "put" for t in trades),
        "call_trades": sum(t.side == "call" for t in trades),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "breakeven": int(len(pnls) - len(wins) - len(losses)),
        "win_rate_pct": r(len(wins) / len(pnls) * 100) if len(pnls) else None,
        "avg_win": r(wins.mean()) if len(wins) else None,
        "avg_loss": r(losses.mean()) if len(losses) else None,
        "largest_win": r(pnls.max()) if len(pnls) else None,
        "largest_loss": r(pnls.min()) if len(pnls) else None,
        "total_pnl": r(pnls.sum()) if len(pnls) else 0.0,
        "max_drawdown": r(max_dd),
        "days_with_no_trade": len(no_trade_days),
        "days_no_signal": status.get("no_signal", 0),
        "days_skipped_early_close": status.get("skipped", 0),
        "days_data_error": status.get("data_error", 0),
        "data_error_days": [f"{d.date}: {d.reason}" for d in result.days if d.status == "data_error"],
        "exit_reasons": dict(Counter(t.exit_reason for t in trades)),
        "trailing_stop_armed_trades": sum(bool(t.trail_armed) for t in trades),
    }


def format_summary(s: dict) -> str:
    def money(x):
        return "n/a" if x is None else f"${x:,.2f}"

    win_rate = "n/a" if s["win_rate_pct"] is None else f"{s['win_rate_pct']:.1f}%"
    lines = [
        f"Backtest {s['period_start']} .. {s['period_end']}  ({s['trading_days']} trading days)",
        f"  Trades:            {s['trades']}  (puts {s['put_trades']}, calls {s['call_trades']})",
        f"  Win rate:          {win_rate}  ({s['wins']} W / {s['losses']} L / {s['breakeven']} BE)",
        f"  Avg win:           {money(s['avg_win'])}",
        f"  Avg loss:          {money(s['avg_loss'])}",
        f"  Largest win/loss:  {money(s['largest_win'])} / {money(s['largest_loss'])}",
        f"  Total P&L:         {money(s['total_pnl'])}",
        f"  Max drawdown:      {money(s['max_drawdown'])}  (on cumulative trade P&L)",
        f"  Days with no trade:{s['days_with_no_trade']:>4}  = no signal {s['days_no_signal']}"
        f" + early-close skipped {s['days_skipped_early_close']} + DATA ERRORS {s['days_data_error']}",
        f"  Exit reasons:      {s['exit_reasons']}",
        f"  Trailing stop armed on {s['trailing_stop_armed_trades']} of {s['trades']} trades",
    ]
    if s["data_error_days"]:
        lines.append("  DATA ERRORS (excluded from P&L, nothing was filled with made-up prices):")
        lines.extend(f"    - {x}" for x in s["data_error_days"])
    return "\n".join(lines)


def write_outputs(result: BacktestResult, cfg: Config, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "trades": out / "trades.csv",
        "checks": out / "checks.csv",
        "days": out / "days.csv",
        "summary_json": out / "summary.json",
        "summary_txt": out / "summary.txt",
        "assumptions": out / "assumptions.txt",
    }
    trade_cols = [
        "date", "signal_time", "side", "strike", "option_symbol", "contracts", "entry_time", "entry_price",
        "exit_time", "exit_price", "exit_reason", "pnl", "pnl_pct",
    ]
    rows = [t.as_row() for t in result.trades]
    trades_df = pd.DataFrame(rows)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=trade_cols)
    else:
        trades_df = trades_df[trade_cols + [c for c in trades_df.columns if c not in trade_cols]]
    trades_df.to_csv(paths["trades"], index=False)
    pd.DataFrame([c.as_row() for d in result.days for c in d.checks]).to_csv(paths["checks"], index=False)
    pd.DataFrame(
        [
            {
                "date": d.date.isoformat(),
                "status": d.status,
                "signal_side": d.signal_side or "",
                "signal_time": d.signal_time or "",
                "reason": d.reason,
                "pnl": d.trade.pnl if d.trade else "",
            }
            for d in result.days
        ]
    ).to_csv(paths["days"], index=False)
    summary = summarize(result)
    paths["summary_json"].write_text(json.dumps(summary, indent=2))
    paths["summary_txt"].write_text(format_summary(summary) + "\n")
    paths["assumptions"].write_text("\n".join(assumption_report(cfg)) + "\n")
    return paths
