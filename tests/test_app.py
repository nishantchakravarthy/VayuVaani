"""App server helpers: the ready-made model and answer caching."""

import os
import time

from vayuvaani.app import server


def test_prebuilt_install_copies_into_place(settings, tmp_path, monkeypatch):
    pre = tmp_path / "pre"
    pre.mkdir()
    for name in server.PREBUILT_FILES:
        (pre / name).write_text(name)
    monkeypatch.setattr(server, "PREBUILT", pre)
    assert server._prebuilt_ready()
    server._install_prebuilt(settings)
    for name, where in server.PREBUILT_FILES.items():
        dest = settings.path(where) / name
        assert dest.read_text() == name
        assert os.access(dest, os.W_OK)  # a later retrain must be able to overwrite it


def test_cache_reuses_until_file_changes(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a")
    calls = []
    fn = lambda: calls.append(1) or len(calls)
    assert server._cached("t", [f], fn) == 1
    assert server._cached("t", [f], fn) == 1
    time.sleep(0.01)
    f.write_text("b")
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert server._cached("t", [f], fn) == 2
