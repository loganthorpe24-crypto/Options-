"""Time helpers. Every timestamp in this project is US/Eastern (America/New_York)."""

from __future__ import annotations

import datetime as dt

import pandas as pd

ET = "America/New_York"
ONE_MINUTE = pd.Timedelta(minutes=1)
FULL_SESSION_MINUTES = 390  # 9:30 -> 16:00


def parse_hhmm(value: str | dt.time) -> dt.time:
    if isinstance(value, dt.time):
        return value
    try:
        hh, mm = str(value).strip().split(":")[:2]
        return dt.time(int(hh), int(mm))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Expected a HH:MM time, got {value!r}") from exc


def et_timestamp(day: dt.date, at: dt.time | str) -> pd.Timestamp:
    """Timezone-aware ET timestamp for `day` at wall-clock time `at`."""
    return pd.Timestamp(dt.datetime.combine(day, parse_hhmm(at))).tz_localize(ET)


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET)


def to_utc_iso(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"Refusing to send a naive timestamp to the API: {ts}")
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def hhmm(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert(ET).strftime("%H:%M")


def fmt_et(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert(ET).strftime("%Y-%m-%d %H:%M:%S")


def parse_date(value: str | dt.date) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))
