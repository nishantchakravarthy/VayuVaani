import json

import numpy as np
import pandas as pd

from vayuvaani import forecast, pipeline, train
from vayuvaani.model import prob_exceed


def test_prob_exceed_behaves():
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    pred = pd.DataFrame([[20, 30, 40, 50, 60]], columns=[f"pm25_q{int(x*100):02d}" for x in q])
    assert abs(prob_exceed(pred, 40, q)[0] - 0.5) < 1e-3
    assert prob_exceed(pred, 10, q)[0] > 0.9
    assert prob_exceed(pred, 200, q)[0] == 0
    p = [prob_exceed(pred, t, q)[0] for t in range(0, 120, 5)]
    assert all(a >= b for a, b in zip(p, p[1:]))


def test_end_to_end(settings):
    pipeline.fetch_demo(settings, "2026-03-01", "2026-05-31")
    pipeline.build(settings)
    report = train.run(settings, pipeline.load_processed(settings))
    assert report["model"]["mae"] < report["persistence"]["mae"]
    assert 0.5 < report["interval_coverage"] <= 1.0
    saved = json.load(open(settings.path("reports") / "evaluation.json"))
    assert saved["rows"] == report["rows"]

    out = forecast.replay(settings)
    assert len(out) == 6 * 48
    cols = ["pm25_q10", "pm25_q25", "pm25_q50", "pm25_q75", "pm25_q90"]
    assert (np.diff(out[cols].to_numpy(), axis=1) >= -0.05).all()
    assert out["p_poor_plus"].between(0, 1).all()
    assert forecast.format_forecast(out)
