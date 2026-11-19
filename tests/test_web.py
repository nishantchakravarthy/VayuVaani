"""The web version: every answer the app asks for is written as a file."""

import json

from vayuvaani import pipeline, train, forecast
from vayuvaani.app import server
from vayuvaani import web


def test_web_build_writes_every_answer(settings, tmp_path, monkeypatch):
    pipeline.fetch_demo(settings, "2026-03-01", "2026-05-15")
    pipeline.build(settings)
    train.run(settings, pipeline.load_processed(settings))
    forecast.replay(settings, "2026-05-10T08:00")
    monkeypatch.setattr(server, "load_settings", lambda: settings)
    monkeypatch.setattr(web, "load_settings", lambda: settings)
    monkeypatch.setattr(server, "live_payload", lambda: {"weather_error": "offline", "air_error": "offline"})
    monkeypatch.setattr(server, "cams_payload", lambda sid: {"error": "offline"})
    out = tmp_path / "site"
    web.build(out, refresh=False)
    for f in ("index.html", "manifest.webmanifest", "sw.js", "icon-192.png", "data/status.json",
              "data/forecast.json", "data/evaluation.json", "data/accuracy.json", "data/live.json"):
        assert (out / f).exists(), f
    assert "window.VV_STATIC=true" in (out / "index.html").read_text(encoding="utf-8")
    fc = json.loads((out / "data/forecast.json").read_text())
    assert fc["stations"]
    for st in fc["stations"]:
        assert (out / "data/explain" / f"{st['id']}.json").exists()
    status = json.loads((out / "data/status.json").read_text())
    assert status["local"] is False and status["web"] is True
