"""CPCB data.

Two routes:
1. data.gov.in real-time API: latest readings for all Bengaluru stations (no history).
   Used to refresh "now" before each forecast run.
2. CSV/XLSX files downloaded by hand from the CPCB CCR portal (app.cpcbccr.com).
   Use these if a station is missing from OpenAQ. List each file in settings.json.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import Settings, secret
from .http import get

log = logging.getLogger(__name__)


def latest_readings(settings: Settings, city: str = "Bengaluru") -> pd.DataFrame:
    """Latest pollutant readings from the data.gov.in CPCB real-time resource.

    Field names on data.gov.in have changed over time, so this parser accepts
    both the older (avg_value) and newer (pollutant_avg) names.
    """
    cfg = settings.sources["data_gov_in"]
    params = {
        "api-key": secret("DATA_GOV_IN_API_KEY"),
        "format": "json",
        "limit": 1000,
        "filters[city]": city,
    }
    records = get(f"{cfg['base_url']}/{cfg['resource_id']}", params=params).json().get("records", [])
    rows = []
    for r in records:
        value = r.get("avg_value", r.get("pollutant_avg"))
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        rows.append({
            "station": r.get("station"),
            "lat": pd.to_numeric(r.get("latitude"), errors="coerce"),
            "lon": pd.to_numeric(r.get("longitude"), errors="coerce"),
            "time": pd.to_datetime(r.get("last_update"), dayfirst=True, errors="coerce"),
            "parameter": str(r.get("pollutant_id", "")).lower().replace(".", ""),
            "value": value,
        })
    return pd.DataFrame(rows)


def _read_ccr_file(path: Path) -> pd.DataFrame:
    """Read a CCR export. The real header row ('From Date', 'To Date', ...) is
    usually preceded by a few title rows, so search for it."""
    reader = pd.read_excel if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv
    raw = reader(path, header=None)
    header_row = None
    for i in range(min(len(raw), 40)):
        cells = [str(c).strip().lower() for c in raw.iloc[i].tolist()]
        if "from date" in cells:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not find a 'From Date' header in {path.name}")
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row]]
    return df


def load_ccr_files(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load manually downloaded CCR files listed in settings.json."""
    entries = settings.raw.get("cpcb_csv_stations") or []
    folder = settings.path("raw") / "cpcb"
    stations, frames = [], []
    for e in entries:
        path = folder / e["file"]
        if not path.exists():
            log.warning("CPCB file not found: %s", path)
            continue
        df = _read_ccr_file(path)
        time = pd.to_datetime(df["From Date"], dayfirst=True, errors="coerce")
        for col, param in (("PM2.5", "pm25"), ("PM10", "pm10")):
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col].replace({"None": None, "NA": None}), errors="coerce")
            frames.append(pd.DataFrame({
                "station_id": e["station_id"], "time": time, "parameter": param, "value": vals,
            }))
        stations.append({
            "station_id": e["station_id"], "name": e["name"], "lat": e["lat"], "lon": e["lon"],
            "provider": "CPCB CCR", "is_monitor": True,
        })
    meas = pd.concat(frames, ignore_index=True).dropna(subset=["time"]) if frames else pd.DataFrame(
        columns=["station_id", "time", "parameter", "value"])
    return pd.DataFrame(stations), meas
