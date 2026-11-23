"""Exit logic: the trailing stop (the 3:45pm hard exit is enforced by the callers).

# =============================================================================
# TRAILING-STOP MEASUREMENT BASIS: the single switch
#
# The creator says: track the favorable move from entry "in ATR units", arm the
# trailing stop at +4 ATR, then exit after 50% of the peak favorable move is
# given back. He does not say WHAT is measured in ATR units. This is the biggest
# ambiguity in his rules, so it lives in exactly one constant:
#
#   "UNDERLYING_MOVE_OVER_ATR5"  (DEFAULT: the definition you specified)
#       favorable = (QQQ now - QQQ at entry) / 5-day ATR   for calls
#                 = (QQQ at entry - QQQ now) / 5-day ATR   for puts
#       QQQ prices are 1-minute closes. The ATR is the same 5-day daily ATR the
#       entry signal used.
#       NOTE: QQQ's 5-day ATR is usually several dollars, so +4 ATR is a very
#       large intraday move. Expect most trades to exit at 3:45pm.
#
#   "OPTION_PREMIUM_RETURN"  (ALTERNATIVE, NOT the default and NOT the stated rule)
#       favorable = (option price now - entry premium) / entry premium
#       With trail_arm_threshold 4.0, that arms at +400%.
#
# To try another definition, change this constant (or add a branch to
# favorable_move()). The value is logged at startup and recorded on every trade.
# =============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

TRAIL_MEASUREMENT_BASIS = "UNDERLYING_MOVE_OVER_ATR5"

BASIS_DESCRIPTIONS = {
    "UNDERLYING_MOVE_OVER_ATR5": "favorable QQQ move from entry divided by the 5-day ATR (1-minute closes)",
    "OPTION_PREMIUM_RETURN": "(option price - entry premium) / entry premium",
}
BASIS_NEEDS_OPTION_PRICE = {
    "UNDERLYING_MOVE_OVER_ATR5": False,
    "OPTION_PREMIUM_RETURN": True,
}


def favorable_move(
    basis: str,
    *,
    side: str,
    entry_underlying: float,
    underlying: float,
    atr: float,
    entry_option_price: float | None = None,
    option_price: float | None = None,
) -> float:
    """Favorable move since entry, in the units of `basis`. Returns NaN if it can't be measured."""
    if basis == "UNDERLYING_MOVE_OVER_ATR5":
        if not atr or atr <= 0:
            raise ValueError(f"ATR must be positive, got {atr}")
        if side == "call":
            return (underlying - entry_underlying) / atr
        if side == "put":
            return (entry_underlying - underlying) / atr
        raise ValueError(f"side must be 'call' or 'put', got {side!r}")
    if basis == "OPTION_PREMIUM_RETURN":
        if option_price is None or entry_option_price is None or not entry_option_price > 0:
            return math.nan
        if isinstance(option_price, float) and math.isnan(option_price):
            return math.nan
        return (option_price - entry_option_price) / entry_option_price
    raise ValueError(
        f"Unknown TRAIL_MEASUREMENT_BASIS {basis!r}. Known: {', '.join(BASIS_DESCRIPTIONS)}"
    )


@dataclass
class TrailingStop:
    """Tracks the peak favorable move, arms at `arm_threshold`, and fires after `giveback_fraction` is given back.

    The rule, literally:
      * favorable move is measured from entry (see TRAIL_MEASUREMENT_BASIS)
      * once the peak favorable move reaches +arm_threshold, the stop is armed
      * once armed, exit when the current favorable move <= peak * (1 - giveback_fraction)
    """

    side: str
    entry_time: pd.Timestamp
    entry_underlying: float
    atr: float
    entry_option_price: float | None
    arm_threshold: float
    giveback_fraction: float
    basis: str = field(default_factory=lambda: TRAIL_MEASUREMENT_BASIS)

    peak: float = 0.0
    peak_time: pd.Timestamp | None = None
    peak_underlying: float | None = None
    armed: bool = False
    armed_time: pd.Timestamp | None = None
    armed_value: float | None = None
    armed_underlying: float | None = None
    last_value: float | None = None
    last_time: pd.Timestamp | None = None
    triggered: bool = False
    exit_value: float | None = None
    exit_time: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.basis not in BASIS_DESCRIPTIONS:
            raise ValueError(f"Unknown trailing-stop basis {self.basis!r}")
        if self.peak_time is None:
            self.peak_time = self.entry_time
            self.peak_underlying = self.entry_underlying

    @property
    def needs_option_price(self) -> bool:
        return BASIS_NEEDS_OPTION_PRICE[self.basis]

    @property
    def stop_level(self) -> float | None:
        """Favorable-move level that triggers the exit (only meaningful once armed)."""
        return self.peak * (1.0 - self.giveback_fraction) if self.armed else None

    def _value(self, underlying: float, option_price: float | None) -> float:
        return favorable_move(
            self.basis,
            side=self.side,
            entry_underlying=self.entry_underlying,
            underlying=underlying,
            atr=self.atr,
            entry_option_price=self.entry_option_price,
            option_price=option_price,
        )

    def update(self, t: pd.Timestamp, *, underlying: float, option_price: float | None = None) -> bool:
        """Feed one observation. Returns True when the trailing stop says exit."""
        if self.triggered:
            return True
        value = self._value(underlying, option_price)
        if math.isnan(value):
            return False
        self.last_value, self.last_time = value, t
        if value > self.peak:
            self.peak, self.peak_time, self.peak_underlying = value, t, underlying
        if not self.armed and self.peak >= self.arm_threshold:
            self.armed = True
            self.armed_time, self.armed_value, self.armed_underlying = t, value, underlying
        if self.armed and value <= self.peak * (1.0 - self.giveback_fraction):
            self.triggered = True
            self.exit_value, self.exit_time = value, t
            return True
        return False

    def mark_exit(self, t: pd.Timestamp, *, underlying: float, option_price: float | None = None) -> None:
        """Record the favorable move at a non-trailing exit (for example the 3:45 hard exit), for the logs."""
        value = self._value(underlying, option_price)
        if not math.isnan(value):
            if value > self.peak:
                self.peak, self.peak_time, self.peak_underlying = value, t, underlying
            self.last_value, self.last_time = value, t
            self.exit_value, self.exit_time = value, t

    @property
    def retracement_from_peak(self) -> float | None:
        """Fraction of the peak favorable move given back at exit (0.5 = gave back 50%)."""
        if self.exit_value is None or self.peak <= 0:
            return None
        return (self.peak - self.exit_value) / self.peak

    def to_state(self) -> dict:
        def ts(x):
            return None if x is None else pd.Timestamp(x).isoformat()

        return {
            "basis": self.basis,
            "peak": self.peak,
            "peak_time": ts(self.peak_time),
            "peak_underlying": self.peak_underlying,
            "armed": self.armed,
            "armed_time": ts(self.armed_time),
            "armed_value": self.armed_value,
            "armed_underlying": self.armed_underlying,
            "last_value": self.last_value,
            "last_time": ts(self.last_time),
        }

    def restore(self, state: dict) -> None:
        """Restore peak/arming after a runner restart."""
        if state.get("basis") not in (None, self.basis):
            raise ValueError(
                f"Saved trailing state used basis {state.get('basis')!r} but code now uses {self.basis!r}; "
                "refusing to mix definitions mid-trade."
            )

        def ts(x):
            return None if x is None else pd.Timestamp(x)

        self.peak = float(state.get("peak") or 0.0)
        self.peak_time = ts(state.get("peak_time")) or self.entry_time
        self.peak_underlying = state.get("peak_underlying", self.entry_underlying)
        self.armed = bool(state.get("armed"))
        self.armed_time = ts(state.get("armed_time"))
        self.armed_value = state.get("armed_value")
        self.armed_underlying = state.get("armed_underlying")
        self.last_value = state.get("last_value")
        self.last_time = ts(state.get("last_time"))

    def describe(self) -> str:
        unit = "ATR" if self.basis == "UNDERLYING_MOVE_OVER_ATR5" else "x premium"
        parts = [f"basis={self.basis}", f"peak={self.peak:+.3f} {unit} at {_hm(self.peak_time)} (QQQ {self.peak_underlying})"]
        if self.armed:
            parts.append(f"ARMED at {_hm(self.armed_time)} (value {self.armed_value:+.3f}, QQQ {self.armed_underlying})")
        else:
            parts.append(f"never armed (needs {self.arm_threshold:+.2f} {unit})")
        if self.exit_value is not None:
            retr = self.retracement_from_peak
            parts.append(
                f"exit value={self.exit_value:+.3f} {unit}"
                + ("" if retr is None else f", retraced {retr:.1%} of peak")
            )
        return "; ".join(parts)


def _hm(ts: pd.Timestamp | None) -> str:
    return "" if ts is None else pd.Timestamp(ts).strftime("%H:%M")
