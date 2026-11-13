"""VayuVaani desktop app.

Runs a small web server on this computer only (127.0.0.1) and opens the
dashboard in the browser. Everything the command line does is a button here.

    python -m vayuvaani.app
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import aqi, explain, forecast, pipeline, train
from ..config import PROJECT_ROOT, load_settings
from ..model import QuantileForecaster
from ..sources import live

HOST = "127.0.0.1"
PORT = int(os.environ.get("VAYUVAANI_PORT", "8765"))
STATIC = Path(__file__).parent / "static"
KEY_NAMES = {"openaq": "OPENAQ_API_KEY", "firms": "FIRMS_MAP_KEY", "datagov": "DATA_GOV_IN_API_KEY"}

log = logging.getLogger("vayuvaani.app")


# ------------------------------------------------------------------ background jobs

class Job:
    """One task running in the background, with its log lines and step progress."""

    def __init__(self):
        self.lock = threading.Lock()
        self.name = None
        self.running = False
        self.error = None
        self.steps: list[dict] = []
        self.lines: deque[str] = deque(maxlen=200)
        self.finished_at = None

    def snapshot(self) -> dict:
        with self.lock:
            return {"name": self.name, "running": self.running, "error": self.error,
                    "steps": [dict(s) for s in self.steps], "log": list(self.lines)[-40:],
                    "finished_at": self.finished_at}


JOB = Job()


class _JobLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        with JOB.lock:
            JOB.lines.append(msg)


def _run_job(name: str, steps: list[tuple[str, callable]]) -> None:
    with JOB.lock:
        JOB.name, JOB.running, JOB.error, JOB.finished_at = name, True, None, None
        JOB.steps = [{"label": label, "status": "waiting"} for label, _ in steps]
        JOB.lines.clear()
    try:
        for i, (label, fn) in enumerate(steps):
            with JOB.lock:
                JOB.steps[i]["status"] = "running"
            log.info("%s...", label)
            fn()
            with JOB.lock:
                JOB.steps[i]["status"] = "done"
    except Exception as exc:  # show any failure in the app instead of crashing
        log.error("%s", exc)
        log.debug(traceback.format_exc())
        with JOB.lock:
            JOB.error = _friendly_error(exc)
            for s in JOB.steps:
                if s["status"] == "running":
                    s["status"] = "failed"
    finally:
        with JOB.lock:
            JOB.running = False
            JOB.finished_at = time.time()


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "401" in text or "403" in text:
        return "The data service rejected the API key. Check that you pasted the right key in Settings."
    if "ConnectionError" in type(exc).__name__ or "Max retries" in text or "NameResolution" in text:
        return "Couldn't reach the internet. Check your connection and try again."
    return text or type(exc).__name__


def start_job(task: str) -> str | None:
    if JOB.running:
        return "Another task is still running. Wait for it to finish."
    s = load_settings()
    mode_file = s.path("processed") / "mode.txt"

    def set_mode(mode):
        return lambda: mode_file.write_text(mode, encoding="utf-8")

    if task == "demo":
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=400)
        steps = [
            ("Creating demo data", lambda: pipeline.fetch_demo(s, start.isoformat(), end.isoformat())),
            ("Cleaning and merging", lambda: pipeline.build(s)),
            ("Training the forecast model", lambda: train.run(s, pipeline.load_processed(s))),
            ("Making a forecast", lambda: forecast.replay(s)),
            ("Finishing", set_mode("demo")),
        ]
    elif task == "real":
        if not os.getenv("OPENAQ_API_KEY"):
            return "Add your OpenAQ API key in Settings first."
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=730)
        no_fires = not os.getenv("FIRMS_MAP_KEY")
        steps = [
            ("Downloading 2 years of air, weather and fire data (10–20 min)",
             lambda: pipeline.fetch(s, start.isoformat(), end.isoformat(), skip_fires=no_fires)),
            ("Cleaning and merging", lambda: pipeline.build(s)),
            ("Training the forecast model", lambda: train.run(s, pipeline.load_processed(s))),
            ("Making a live forecast", lambda: forecast.live(s)),
            ("Finishing", set_mode("real")),
        ]
    elif task == "retrain":
        mode = mode_file.read_text(encoding="utf-8").strip() if mode_file.exists() else ""
        if not (s.path("raw") / "measurements.csv.gz").exists():
            return "No downloaded data yet. Set up demo or real data first."
        last = (lambda: forecast.live(s)) if mode == "real" else (lambda: forecast.replay(s))
        steps = [
            ("Cleaning and merging", lambda: pipeline.build(s)),
            ("Training the forecast model (then retraining on all data)", lambda: train.run(s, pipeline.load_processed(s))),
            ("Making a forecast", last),
        ]
    elif task == "refresh":
        mode = mode_file.read_text(encoding="utf-8").strip() if mode_file.exists() else ""
        if mode == "real":
            steps = [("Getting the latest readings and weather forecast", lambda: forecast.live(s))]
        elif mode == "demo":
            steps = [("Making a forecast from demo data", lambda: forecast.replay(s))]
        else:
            return "Set up the app first (demo or real data)."
    else:
        return f"Unknown task: {task}"

    threading.Thread(target=_run_job, args=(task, steps), daemon=True).start()
    return None


# ------------------------------------------------------------------ data for the page

def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def status() -> dict:
    s = load_settings()
    processed, models, reports = s.path("processed"), s.path("models"), s.path("reports")
    mode_file = processed / "mode.txt"
    fc = reports / "latest_forecast.csv"
    return {
        "keys": {k: bool(os.getenv(v)) for k, v in KEY_NAMES.items()},
        "mode": mode_file.read_text(encoding="utf-8").strip() if mode_file.exists() else None,
        "model_ready": (models / "forecaster.joblib").exists(),
        "forecast_ready": fc.exists(),
        "forecast_updated": time.strftime("%d %b %Y, %I:%M %p", time.localtime(fc.stat().st_mtime)) if fc.exists() else None,
        "job": JOB.snapshot(),
    }


def forecast_payload() -> dict:
    s = load_settings()
    path = s.path("reports") / "latest_forecast.csv"
    if not path.exists():
        return {"stations": []}
    df = pd.read_csv(path, parse_dates=["origin_time", "target_time"])
    fpath = s.path("reports") / "latest_features.csv"
    if fpath.exists():
        cols = ["station_id", "target_time", "t_temp", "t_rh", "t_wind_speed", "t_rain"]
        feats = pd.read_csv(fpath, usecols=lambda c: c in cols, parse_dates=["target_time"])
        df = df.merge(feats, on=["station_id", "target_time"], how="left")
    stations = []
    for sid, g in df.groupby("station_id", sort=False):
        g = g.sort_values("horizon")
        hours = []
        for _, r in g.iterrows():
            hours.append({
                "time": r["target_time"].strftime("%Y-%m-%dT%H:%M"),
                "q10": _clean(float(r["pm25_q10"])), "q25": _clean(float(r["pm25_q25"])),
                "q50": _clean(float(r["pm25_q50"])), "q75": _clean(float(r["pm25_q75"])),
                "q90": _clean(float(r["pm25_q90"])),
                "aqi": _clean(float(r["aqi_pm25"])), "category": r["category"],
                "p_poor": _clean(float(r["p_poor_plus"])),
                "actual": _clean(float(r["actual_pm25"])) if "actual_pm25" in r else None,
                "temp": _clean(float(r["t_temp"])) if "t_temp" in r else None,
                "rh": _clean(float(r["t_rh"])) if "t_rh" in r else None,
                "wind": _clean(float(r["t_wind_speed"])) if "t_wind_speed" in r else None,
                "rain": _clean(float(r["t_rain"])) if "t_rain" in r else None,
            })
        first = g.iloc[0]
        stations.append({
            "id": sid, "name": first["name"], "lat": float(first["lat"]), "lon": float(first["lon"]),
            "issued": first["origin_time"].strftime("%Y-%m-%dT%H:%M"),
            "now_pm25": _clean(float(first["pm25_origin"])),
            "hours": hours,
        })
    off_path = s.path("reports") / "offline_stations.json"
    offline = json.loads(off_path.read_text(encoding="utf-8")) if off_path.exists() else []
    return {"stations": stations, "offline": offline}


def evaluation_payload() -> dict:
    path = load_settings().path("reports") / "evaluation.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


_MODEL = {"mtime": None, "fc": None}


def _model() -> QuantileForecaster | None:
    path = load_settings().path("models") / "forecaster.joblib"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _MODEL["mtime"] != mtime:
        _MODEL["fc"], _MODEL["mtime"] = QuantileForecaster.load(path), mtime
    return _MODEL["fc"]


def _live_error(exc: Exception) -> str:
    text = str(exc)
    code = text.split()[0] if text[:3].isdigit() else ""
    return f"couldn't reach Open-Meteo{' (HTTP ' + code + ')' if code else ''}"


def explain_payload(station_id: str, target_time: str | None) -> dict:
    fc = _model()
    path = load_settings().path("reports") / "latest_features.csv"
    if fc is None or not path.exists():
        return {"error": "Run a forecast first."}
    feats = pd.read_csv(path)
    return explain.explain_station(fc, feats, station_id, target_time or None, load_settings().timezone)


def live_payload() -> dict:
    s = load_settings()
    lat, lon = s.city["centre"]
    out = {"lat": lat, "lon": lon}
    try:
        w = live.weather_now(lat, lon, s.timezone)
        cur = w.get("current", {})
        daily = w.get("daily", {})
        out["weather"] = {
            "time": cur.get("time"), "temp": cur.get("temperature_2m"), "feels": cur.get("apparent_temperature"),
            "rh": cur.get("relative_humidity_2m"), "wind": cur.get("wind_speed_10m"),
            "wind_dir": cur.get("wind_direction_10m"), "rain": cur.get("precipitation"),
            "cloud": cur.get("cloud_cover"),
            "desc": live.WEATHER_CODES.get(cur.get("weather_code"), ""),
            "tmax": (daily.get("temperature_2m_max") or [None])[0],
            "tmin": (daily.get("temperature_2m_min") or [None])[0],
            "sunrise": (daily.get("sunrise") or [None])[0], "sunset": (daily.get("sunset") or [None])[0],
        }
    except Exception as exc:
        out["weather_error"] = _live_error(exc)
    try:
        a = live.air_quality(lat, lon, s.timezone)
        cur = a.get("current", {})
        pm25 = cur.get("pm2_5")
        hourly = a.get("hourly", {})
        out["air"] = {
            "time": cur.get("time"), "pm25": pm25, "pm10": cur.get("pm10"),
            "no2": cur.get("nitrogen_dioxide"), "o3": cur.get("ozone"), "co": cur.get("carbon_monoxide"),
            "so2": cur.get("sulphur_dioxide"),
            "index_pm25": _clean(float(aqi.pm25_sub_index(pm25))) if pm25 is not None else None,
            "category": aqi.pm25_category(pm25) if pm25 is not None else None,
            "hourly": [{"time": t, "pm25": v} for t, v in zip(hourly.get("time", []), hourly.get("pm2_5", []))],
        }
    except Exception as exc:
        out["air_error"] = _live_error(exc)
    return out


def cams_payload(station_id: str) -> dict:
    s = load_settings()
    path = s.path("reports") / "latest_forecast.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["station_id", "lat", "lon"]).drop_duplicates("station_id")
    row = df[df["station_id"] == station_id]
    if row.empty:
        return {}
    try:
        a = live.air_quality(float(row["lat"].iloc[0]), float(row["lon"].iloc[0]), s.timezone)
    except Exception as exc:
        return {"error": _live_error(exc)}
    h = a.get("hourly", {})
    return {"hourly": [{"time": t, "pm25": v} for t, v in zip(h.get("time", []), h.get("pm2_5", []))]}


def accuracy_payload() -> dict:
    s = load_settings()
    ev = evaluation_payload()
    path = s.path("reports") / "test_predictions.csv.gz"
    if not ev or not path.exists():
        return ev
    res = pd.read_csv(path, usecols=["station_id", "horizon", "pm25_q50", "pm25_q10", "pm25_q90", "actual", "persistence"])
    res = res.dropna(subset=["actual"])
    names = {}
    fpath = s.path("reports") / "latest_forecast.csv"
    if fpath.exists():
        f = pd.read_csv(fpath, usecols=["station_id", "name"]).drop_duplicates("station_id")
        names = dict(zip(f["station_id"], f["name"]))
    by_h = res.groupby("horizon").apply(lambda g: pd.Series({
        "model": float((g["pm25_q50"] - g["actual"]).abs().mean()),
        "persistence": float((g["persistence"] - g["actual"]).abs().mean()),
        "coverage": float(((g["actual"] >= g["pm25_q10"]) & (g["actual"] <= g["pm25_q90"])).mean()),
    }), include_groups=False).reset_index()
    by_s = res.groupby("station_id").apply(lambda g: pd.Series({
        "model": float((g["pm25_q50"] - g["actual"]).abs().mean()),
        "persistence": float((g["persistence"] - g["actual"]).abs().mean()),
    }), include_groups=False).reset_index()
    sample = res[res["horizon"] == 24].sample(min(800, (res["horizon"] == 24).sum()), random_state=1)
    ev["by_hour"] = by_h.round(2).to_dict("records")
    ev["by_station"] = [{**r, "name": names.get(r["station_id"], r["station_id"])} for r in by_s.round(2).to_dict("records")]
    ev["scatter"] = sample[["pm25_q50", "actual"]].round(1).values.tolist()
    return ev


TTS_LANGS = {"en": ("en", "co.in"), "kn": ("kn", "com"), "hi": ("hi", "co.in")}


def tts_audio(lang: str, text: str) -> bytes:
    """Voice clip for an alert, made with gTTS (Google Text-to-Speech) and cached on disk."""
    import hashlib
    from gtts import gTTS
    if lang not in TTS_LANGS:
        raise ValueError("unsupported language")
    text = text.strip()[:600]
    folder = load_settings().path("reports") / "tts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (hashlib.sha1(f"{lang}|{text}".encode("utf-8")).hexdigest()[:16] + ".mp3")
    if not path.exists():
        code, tld = TTS_LANGS[lang]
        gTTS(text=text, lang=code, tld=tld).save(str(path))
    return path.read_bytes()


def save_keys(body: dict) -> None:
    s = load_settings()
    env_path = s.root / ".env"
    existing = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    for short, env in KEY_NAMES.items():
        if short in body and body[short] is not None:
            value = str(body[short]).strip()
            if value:
                existing[env] = value
                os.environ[env] = value
    lines = ["# VayuVaani API keys (saved by the app). Keep this file private."]
    lines += [f"{k}={v}" for k, v in existing.items()]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def do_GET(self):
        global _LAST_SEEN
        _LAST_SEEN = time.time()
        url = urlparse(self.path)
        route = url.path
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if route in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/ping":
                return self._json({"ok": True})
            elif route == "/api/status":
                self._json(status())
            elif route == "/api/forecast":
                self._json(forecast_payload())
            elif route == "/api/evaluation":
                self._json(evaluation_payload())
            elif route == "/api/accuracy":
                self._json(accuracy_payload())
            elif route == "/api/explain":
                self._json(explain_payload(q.get("station", ""), q.get("time")))
            elif route == "/api/live":
                self._json(live_payload())
            elif route == "/api/cams":
                self._json(cams_payload(q.get("station", "")))
            elif route == "/api/tts":
                try:
                    audio = tts_audio(q.get("lang", "en"), q.get("text", ""))
                    self._send(200, audio, "audio/mpeg")
                except ImportError:
                    self._json({"error": "gTTS is not installed"}, 503)
                except Exception as exc:
                    self._json({"error": _live_error(exc).replace("Open-Meteo", "the voice service")}, 502)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def do_POST(self):
        # Only accept requests from the app's own page.
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"):
            return self._json({"error": "forbidden"}, 403)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad request"}, 400)
        route = self.path.split("?", 1)[0]
        if route == "/api/run":
            err = start_job(body.get("task", ""))
            return self._json({"ok": err is None, "error": err}, 200 if err is None else 409)
        if route == "/api/keys":
            save_keys(body)
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)


_LAST_SEEN = time.time()


def _find_app_browser():
    """Edge (built into Windows) or Chrome, used only as a window with no tabs or address bar."""
    import shutil
    cands = []
    for env in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            cands += [Path(base) / "Microsoft/Edge/Application/msedge.exe",
                      Path(base) / "Google/Chrome/Application/chrome.exe"]
    for c in cands:
        if c.exists():
            return str(c)
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def open_app_window(url: str):
    """Open VayuVaani in its own app window. Returns the window process, or None."""
    exe = _find_app_browser()
    if not exe:
        webbrowser.open(url)
        return None
    profile = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VayuVaani" / "window"
    profile.mkdir(parents=True, exist_ok=True)
    args = [exe, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1440,920",
            "--no-first-run", "--no-default-browser-check", "--disable-features=Translate,msEdgeSidebarV2"]
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    return subprocess.Popen(args, creationflags=flags)


def _quiet_console():
    """When started without a console (pythonw), send output to app_log.txt instead."""
    if sys.stdout is None or sys.stderr is None:
        log = open(PROJECT_ROOT / "app_log.txt", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log


def main(open_browser: bool = True) -> None:
    _quiet_console()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    handler = _JobLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    load_settings()  # loads .env into the environment

    url = f"http://{HOST}:{PORT}"
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError:
        # Already running (e.g. opened twice): just show the window.
        if open_browser:
            open_app_window(url)
        return
    print(f"VayuVaani is running at {url}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if not open_browser:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return

    proc = open_app_window(url)
    started = time.time()
    if proc is not None:
        proc.wait()
    # If the window handed off to an already-open browser (or none was found),
    # stay up while the page keeps pinging, then stop on our own.
    if proc is None or time.time() - started < 8:
        while time.time() - _LAST_SEEN < 45:
            time.sleep(5)
    server.shutdown()


if __name__ == "__main__":
    main()
