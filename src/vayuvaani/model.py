"""Quantile forecaster for PM2.5.

One gradient-boosting model per quantile (10th, 25th, 50th, 75th, 90th percentile).
The median is the forecast; the 10th–90th range is the confidence band, and the
quantiles together give the chance of crossing an AQI category.

Uses LightGBM when installed, otherwise scikit-learn's HistGradientBoosting
(slower, slightly less accurate, but needs no extra install).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import aqi
from .features import FEATURES

log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # pragma: no cover - depends on the machine
    HAS_LGB = False


def _make_regressor(q: float, params: dict, random_state: int, backend: str):
    if backend == "lightgbm":
        return lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=random_state,
                                 verbose=-1, **params)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        max_leaf_nodes=params.get("num_leaves", 63),
        min_samples_leaf=params.get("min_child_samples", 40),
        random_state=random_state,
    )


@dataclass
class QuantileForecaster:
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    params: dict = field(default_factory=dict)
    random_state: int = 42
    backend: str = "lightgbm" if HAS_LGB else "sklearn"
    models: dict = field(default_factory=dict)
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    widen: float = 0.0  # extra width for the outer quantiles, set by calibrate()
    reference: dict = field(default_factory=dict)  # typical training values, for explanations
    # Learn the change from the current reading (log ratio) instead of the level.
    # Tested on real Bengaluru data: clearly better for the first hours ahead.
    relative: bool = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileForecaster":
        # Train on log(1 + PM2.5): errors become relative, and quantiles carry over unchanged.
        target = np.log1p(np.clip(y.to_numpy(), 0, None)) - self._base(X)
        X = X[self.features]
        for q in self.quantiles:
            log.info("Training %s model for quantile %.2f on %d rows", self.backend, q, len(X))
            m = _make_regressor(q, self.params, self.random_state, self.backend)
            m.fit(X, target)
            self.models[q] = m
        return self

    def _base(self, X: pd.DataFrame) -> np.ndarray:
        if getattr(self, "relative", False):
            return np.log1p(np.clip(X["pm25_lag0"].to_numpy(dtype=float), 0, None))
        return np.zeros(len(X))

    def _predict_log(self, X: pd.DataFrame) -> np.ndarray:
        base = self._base(X)
        X = X[self.features]
        preds = np.column_stack([self.models[q].predict(X) for q in self.quantiles]) + base[:, None]
        return np.sort(preds, axis=1)          # quantiles must not cross

    def calibrate(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Widen the outer quantiles so the 10–90% range really covers ~80% of
        unseen data (conformalised quantile regression). Use data the models
        were not trained on."""
        preds = self._predict_log(X)
        t = np.log1p(np.clip(y.to_numpy(), 0, None))
        score = np.maximum(preds[:, 0] - t, t - preds[:, -1])
        target = self.quantiles[-1] - self.quantiles[0]
        n = len(score)
        level = min(1.0, np.ceil((n + 1) * target) / n)
        self.widen = float(max(np.quantile(score, level), 0.0))
        log.info("Calibration widens the outer quantiles by %.3f (log units)", self.widen)
        return self.widen

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = self._predict_log(X)
        if self.widen:
            k = len(self.quantiles)
            for i in range(k // 2):
                share = (k // 2 - i) / (k // 2)   # outermost gets the full widening
                preds[:, i] -= self.widen * share
                preds[:, k - 1 - i] += self.widen * share
        preds = np.clip(np.expm1(preds), 0, None)
        X = X[self.features]
        return pd.DataFrame(preds, columns=[qname(q) for q in self.quantiles], index=X.index)

    def feature_importance(self) -> pd.Series:
        m = self.models[0.5]
        if hasattr(m, "feature_importances_"):
            return pd.Series(m.feature_importances_, index=self.features).sort_values(ascending=False)
        return pd.Series(dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "QuantileForecaster":
        return joblib.load(path)


def qname(q: float) -> str:
    return f"pm25_q{int(round(q * 100)):02d}"


def prob_exceed(pred: pd.DataFrame, threshold: float, quantiles: list[float]) -> np.ndarray:
    """Chance that PM2.5 is at or above `threshold`, from the predicted quantiles.

    Builds a piecewise-linear distribution through the quantile points, with
    0% at zero and 99% a little beyond the 90th percentile.
    """
    cols = [qname(q) for q in quantiles]
    vals = pred[cols].to_numpy()
    q50 = pred[qname(0.5)].to_numpy() if qname(0.5) in pred else vals[:, len(cols) // 2]
    top = vals[:, -1] + np.maximum(vals[:, -1] - q50, 1.0)
    probs = np.array([0.0] + list(quantiles) + [0.99])
    xs = np.column_stack([np.zeros(len(pred)), vals, top])
    xs = np.maximum.accumulate(xs + np.arange(xs.shape[1]) * 1e-6, axis=1)
    # Find the segment containing the threshold in each row, then interpolate linearly.
    k = np.clip((xs <= threshold).sum(axis=1) - 1, 0, xs.shape[1] - 2)
    rows = np.arange(len(pred))
    x0, x1 = xs[rows, k], xs[rows, k + 1]
    frac = np.clip((threshold - x0) / (x1 - x0), 0, 1)
    cdf = probs[k] + frac * (probs[k + 1] - probs[k])
    cdf = np.where(threshold >= xs[:, -1], 1.0, cdf)
    return np.clip(1.0 - cdf, 0, 1)


def summarise(pred: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """Add AQI (PM2.5 sub-index), category and chance of 'Poor or worse' air."""
    out = pred.copy()
    med = out[qname(0.5)]
    out["aqi_pm25"] = aqi.pm25_sub_index(med.to_numpy())
    out["category"] = aqi.categories(out["aqi_pm25"]).to_numpy()
    out["p_moderate_plus"] = prob_exceed(pred, aqi.pm25_threshold("Moderate"), quantiles)
    out["p_poor_plus"] = prob_exceed(pred, aqi.pm25_threshold("Poor"), quantiles)
    return out
