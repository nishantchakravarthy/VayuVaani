"""Build the web version of VayuVaani: the same app as static files.

It opens on any phone, tablet or computer (Android, iOS, macOS, Windows, Linux) in a
browser and can be installed to the home screen. A scheduled job (GitHub Actions) runs
this every hour: it makes a fresh forecast with the ready-made model and writes every
answer the app needs as JSON files next to the page.

    python -m vayuvaani.web site/            # forecast now, then build
    python -m vayuvaani.web site/ --no-refresh
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from . import forecast
from .app import server
from .config import load_settings

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "app" / "static"


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str, separators=(",", ":")), encoding="utf-8")


def build(out: Path, refresh: bool = True) -> None:
    s = load_settings()
    if not (s.path("models") / "forecaster.joblib").exists() and server._prebuilt_ready():
        server._install_prebuilt(s)
    if refresh:
        forecast.live(s)

    out.mkdir(parents=True, exist_ok=True)
    data = out / "data"
    st = server.status()
    fpath = s.path("reports") / "latest_forecast.csv"
    st.update(local=False, web=True, job={"name": None, "running": False, "error": None, "steps": [], "log": []},
              built_epoch=int(time.time()),
              forecast_epoch=int(fpath.stat().st_mtime) if fpath.exists() else int(time.time()),
              keys={"openaq": True, "firms": True, "datagov": False}, phone={"on": False, "serving": False, "urls": []})
    _write(data / "status.json", st)

    fc = server.forecast_payload()
    _write(data / "forecast.json", fc)
    _write(data / "evaluation.json", server.evaluation_payload())
    _write(data / "accuracy.json", server.accuracy_payload())
    try:
        _write(data / "live.json", server.live_payload())
    except Exception as exc:  # weather service down: the page shows its own message
        _write(data / "live.json", {"air_error": str(exc), "weather_error": str(exc)})

    for st_ in fc.get("stations", []):
        sid = st_["id"]
        try:
            _write(data / "explain" / f"{sid}.json", server.explain_payload(sid, None))
        except Exception as exc:
            _write(data / "explain" / f"{sid}.json", {"error": str(exc)})
        try:
            _write(data / "cams" / f"{sid}.json", server.cams_payload(sid))
        except Exception as exc:
            _write(data / "cams" / f"{sid}.json", {"error": str(exc)})

    page = (STATIC / "index.html").read_text(encoding="utf-8")
    page = page.replace("<head>", "<head>\n<script>window.VV_STATIC=true</script>", 1)
    (out / "index.html").write_text(page, encoding="utf-8")
    for name in server.ASSETS:
        if (STATIC / name).exists():
            shutil.copyfile(STATIC / name, out / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    log.info("Web version written to %s (%d stations)", out, len(fc.get("stations", [])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-refresh", action="store_true", help="use the forecast already on disk")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    build(a.out, refresh=not a.no_refresh)


if __name__ == "__main__":
    main()
