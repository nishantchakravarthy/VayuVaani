"""Train, evaluate and save the forecaster.

Split by time: the last `test_days` days are held out. Training origins stop
`max_horizon_hours` before the test period so no training target falls inside it.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import aqi
from .config import Settings
from .features import add_origin_features, make_rows
from .model import QuantileForecaster, qname, summarise

log = logging.getLogger(__name__)

HORIZON_BUCKETS = [(1, 6), (7, 12), (13, 24), (25, 48)]


def split_times(df: pd.DataFrame, test_days: int, max_h: int):
    end = df["time"].max()
    test_start = end - pd.Timedelta(days=test_days)
    train_end = test_start - pd.Timedelta(hours=max_h)
    return train_end, test_start


def baselines(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Two simple forecasts to beat:
    - persistence: PM2.5 stays at its current value;
    - same hour, last observed day: the value at the same clock hour on the latest
      day already observed at the origin.
    """
    lookup = df.set_index(["station_id", "time"])["pm25"]
    days_back = np.ceil(meta["horizon"] / 24).astype(int) * 24
    prev_time = pd.to_datetime(meta["target_time"]) - pd.to_timedelta(days_back, unit="h")
    idx = pd.MultiIndex.from_arrays([meta["station_id"], prev_time])
    return pd.DataFrame({
        "persistence": meta["pm25_origin"].to_numpy(),
        "same_hour": lookup.reindex(idx).to_numpy(),
    })


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    ok = ~np.isnan(y) & ~np.isnan(pred)
    err = pred[ok] - y[ok]
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))), "n": int(ok.sum())}


def evaluate(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
             fc: QuantileForecaster, alert_probability: float = 0.3) -> tuple[dict, pd.DataFrame]:
    pred = summarise(fc.predict(X), fc.quantiles)
    base = baselines(df, meta)
    res = pd.concat([meta, pred, base], axis=1)
    res["actual"] = y.to_numpy()
    res = res[res["actual"].notna()].reset_index(drop=True)

    a = res["actual"].to_numpy()
    lo, hi = res[qname(fc.quantiles[0])], res[qname(fc.quantiles[-1])]
    actual_cat = aqi.categories(aqi.pm25_sub_index(a))
    poor_actual = a >= aqi.pm25_threshold("Poor")
    poor_alert = res["p_poor_plus"].to_numpy() >= alert_probability

    report = {
        "rows": int(len(res)),
        "model": metrics(a, res[qname(0.5)].to_numpy()),
        "persistence": metrics(a, res["persistence"].to_numpy()),
        "same_hour": metrics(a, res["same_hour"].to_numpy()),
        "interval_coverage": float(((a >= lo) & (a <= hi)).mean()),
        "interval_target": float(fc.quantiles[-1] - fc.quantiles[0]),
        "category_accuracy": float((res["category"].to_numpy() == actual_cat.to_numpy()).mean()),
        "poor_alerts": {
            "actual_poor_hours": int(poor_actual.sum()),
            "alert_probability": alert_probability,
            "alerts_sent": int(poor_alert.sum()),
            "precision": float((poor_alert & poor_actual).sum() / poor_alert.sum()) if poor_alert.any() else None,
            "recall": float((poor_alert & poor_actual).sum() / poor_actual.sum()) if poor_actual.any() else None,
        },
        "by_horizon": {},
    }
    report["skill_vs_persistence"] = 1 - report["model"]["mae"] / report["persistence"]["mae"]
    for lo_h, hi_h in HORIZON_BUCKETS:
        m = (res["horizon"] >= lo_h) & (res["horizon"] <= hi_h)
        report["by_horizon"][f"{lo_h}-{hi_h}h"] = {
            "model_mae": metrics(a[m], res.loc[m, qname(0.5)].to_numpy())["mae"],
            "persistence_mae": metrics(a[m], res.loc[m, "persistence"].to_numpy())["mae"],
            "same_hour_mae": metrics(a[m], res.loc[m, "same_hour"].to_numpy())["mae"],
        }
    return report, res


def run(settings: Settings, df: pd.DataFrame) -> dict:
    cfg = settings.model
    max_h = cfg["max_horizon_hours"]
    rng = np.random.default_rng(cfg["random_state"])

    df = add_origin_features(df)
    train_end, test_start = split_times(df, cfg["test_days"], max_h)
    log.info("Training origins up to %s; testing from %s", train_end, test_start)

    t = df["time"].to_numpy()
    # The last calibration_days of the training period are kept aside to calibrate
    # the confidence range; the models never see them.
    calib_start = train_end - pd.Timedelta(days=cfg.get("calibration_days", 30))
    fit_end = calib_start - pd.Timedelta(hours=max_h)
    X_tr, y_tr, _ = make_rows(df, max_h, cfg["horizons_per_origin"], rng=rng,
                              origin_mask=t <= np.datetime64(fit_end))
    X_cal, y_cal, _ = make_rows(df, max_h, cfg["horizons_per_origin"], rng=rng,
                                origin_mask=(t >= np.datetime64(calib_start)) & (t <= np.datetime64(train_end)))
    X_te, y_te, meta_te = make_rows(df, max_h, None, origin_step=3,
                                    origin_mask=t >= np.datetime64(test_start))

    fc = QuantileForecaster(quantiles=cfg["quantiles"], params=cfg.get("lightgbm", {}),
                            random_state=cfg["random_state"])
    fc.fit(X_tr, y_tr)
    fc.calibrate(X_cal, y_cal)
    fc.reference = X_tr.median(numeric_only=True).to_dict()

    report, res = evaluate(df, X_te, y_te, meta_te, fc, cfg.get("alert_probability", 0.3))
    report.update({
        "backend": fc.backend,
        "train_rows": int(len(X_tr)),
        "train_until": str(train_end),
        "test_from": str(test_start),
        "stations": int(df["station_id"].nunique()),
        "top_features": fc.feature_importance().round(1).to_dict(),
    })

    # The evaluation above is honest (the model never saw the test days). For live
    # forecasts, re-calibrate the range on those most recent out-of-sample days
    # (seasons shift, so recent errors are the best guide), then retrain on everything.
    ok = y_te.notna().to_numpy()
    if ok.sum() > 1000:
        fc.calibrate(X_te[ok], y_te[ok])
        report["live_range_widen"] = round(fc.widen, 3)
    if cfg.get("refit_on_all_data", True):
        log.info("Retraining on all data for live forecasts")
        X_all, y_all, _ = make_rows(df, max_h, cfg["horizons_per_origin"], rng=rng)
        final = QuantileForecaster(quantiles=cfg["quantiles"], params=cfg.get("lightgbm", {}),
                                   random_state=cfg["random_state"])
        final.fit(X_all, y_all)
        final.widen, final.reference = fc.widen, fc.reference
        report["final_model_rows"] = int(len(X_all))
        fc = final
    fc.save(settings.path("models") / "forecaster.joblib")
    reports = settings.path("reports")
    with open(reports / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    res.to_csv(reports / "test_predictions.csv.gz", index=False)
    log.info("Saved model and evaluation report")
    return report


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.0f}%"


def _alert_line(a: dict) -> str:
    return (f"'Poor or worse' alerts (sent when chance >= {a['alert_probability'] * 100:.0f}%): "
            f"{a['alerts_sent']} sent, precision {_pct(a['precision'])}, recall {_pct(a['recall'])} "
            f"({a['actual_poor_hours']} actual poor hours)")


def format_report(r: dict) -> str:
    lines = [
        f"Backend: {r['backend']}   Stations: {r['stations']}   Training rows: {r['train_rows']:,}",
        f"Test period from {r['test_from'][:10]} ({r['rows']:,} forecasts)",
        "",
        f"{'':22}{'MAE':>8}{'RMSE':>8}   (µg/m³ PM2.5)",
        f"{'VayuVaani model':22}{r['model']['mae']:8.1f}{r['model']['rmse']:8.1f}",
        f"{'Persistence':22}{r['persistence']['mae']:8.1f}{r['persistence']['rmse']:8.1f}",
        f"{'Same hour, last day':22}{r['same_hour']['mae']:8.1f}{r['same_hour']['rmse']:8.1f}",
        "",
        f"Skill vs persistence: {r['skill_vs_persistence'] * 100:.0f}% lower error",
        f"10–90% range covered the actual value {r['interval_coverage'] * 100:.0f}% of the time "
        f"(target {r['interval_target'] * 100:.0f}%)",
        f"AQI category correct: {r['category_accuracy'] * 100:.0f}%",
        _alert_line(r["poor_alerts"]),
        "",
        "MAE by forecast horizon:",
    ]
    for k, v in r["by_horizon"].items():
        lines.append(f"  {k:>7}: model {v['model_mae']:5.1f}   persistence {v['persistence_mae']:5.1f}   "
                     f"same hour {v['same_hour_mae']:5.1f}")
    return "\n".join(lines)
