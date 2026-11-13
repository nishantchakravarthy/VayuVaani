import shutil
from pathlib import Path

import pytest
import json

from vayuvaani.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    """Settings pointing at a temporary project folder, with a small, fast model."""
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "events").mkdir(parents=True)
    shutil.copy(ROOT / "data/events/bengaluru_events.csv", tmp_path / "data/events/bengaluru_events.csv")
    cfg = json.load(open(ROOT / "config/settings.json"))
    cfg["model"].update({"test_days": 15, "calibration_days": 10, "horizons_per_origin": 3})
    cfg["model"]["lightgbm"].update({"n_estimators": 40, "num_leaves": 15})
    with open(tmp_path / "config/settings.json", "w") as f:
        json.dump(cfg, f)
    return load_settings(root=tmp_path)
