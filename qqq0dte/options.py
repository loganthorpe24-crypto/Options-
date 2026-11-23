"""OCC option symbols and strike selection."""

from __future__ import annotations

import datetime as dt
import re

_OCC = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<date>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


def occ_symbol(root: str, expiry: dt.date, right: str, strike: float) -> str:
    """OCC symbol as used by Alpaca, e.g. QQQ261120P00632000."""
    r = {"call": "C", "put": "P", "C": "C", "P": "P"}[right]
    return f"{root}{expiry:%y%m%d}{r}{int(round(strike * 1000)):08d}"


def parse_occ(symbol: str) -> tuple[str, dt.date, str, float]:
    m = _OCC.match(symbol)
    if not m:
        raise ValueError(f"Not an OCC option symbol: {symbol!r}")
    expiry = dt.datetime.strptime(m["date"], "%y%m%d").date()
    return m["root"], expiry, "call" if m["right"] == "C" else "put", int(m["strike"]) / 1000.0


def select_strike(
    price: float,
    side: str,
    strikes: list[float],
    *,
    method: str = "atm",
    offset: int = 0,
    tie_break: str = "lower",
) -> float:
    """Pick a strike from the listed strikes.

    atm: strike closest to `price` (ties go to `tie_break`).
    itm/otm: start at ATM and move `offset` listed strikes in/out of the money
             (puts: ITM = higher strikes; calls: ITM = lower strikes).
    """
    if side not in ("call", "put"):
        raise ValueError(f"side must be call/put, got {side!r}")
    ladder = sorted(set(float(s) for s in strikes))
    if not ladder:
        raise ValueError("no strikes to choose from")
    best = min(abs(s - price) for s in ladder)
    tied = [s for s in ladder if abs(abs(s - price) - best) < 1e-9]
    atm = min(tied) if tie_break == "lower" else max(tied)
    if method == "atm":
        return atm
    if method not in ("itm", "otm"):
        raise ValueError(f"Unknown strike_selection {method!r}")
    i = ladder.index(atm)
    toward_higher = (side == "put") == (method == "itm")
    j = i + offset if toward_higher else i - offset
    if not 0 <= j < len(ladder):
        raise ValueError(f"strike_offset {offset} runs off the listed strikes ({ladder[0]}..{ladder[-1]})")
    return ladder[j]
