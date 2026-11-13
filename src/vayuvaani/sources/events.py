"""Event calendar: festivals, fireworks nights, holidays and dry season."""

from __future__ import annotations

import pandas as pd

from ..config import Settings

EVENT_TYPES = ["fireworks", "festival", "holiday", "dry_season"]


def load(settings: Settings) -> pd.DataFrame:
    df = pd.read_csv(settings.path("events"), comment="#", skipinitialspace=True)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    unknown = set(df["type"]) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f"Unknown event types in calendar: {unknown}")
    return df


def daily_flags(events: pd.DataFrame, start, end) -> pd.DataFrame:
    """One row per date with a 0/1 column per event type."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    out = pd.DataFrame({"date": days})
    for t in EVENT_TYPES:
        out[f"event_{t}"] = 0
    for _, e in events.iterrows():
        mask = (out["date"] >= e["start"]) & (out["date"] <= e["end"])
        out.loc[mask, f"event_{e['type']}"] = 1
    return out
