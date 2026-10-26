import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import explain
from src.baseline_model import COST_FN, COST_FP, best_threshold
from src.model_tuning import (
    fit_rf, fit_xgb, load_splits, run, select_winner, threshold_for_costs, RF_MAX_TREES,
)


@pytest.fixture(scope="module")
def small_data():
    return load_splits("synthetic", synthetic_n=1500)


def test_threshold_helper_matches_task2_at_client_costs():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.05).astype(int)
    p = np.clip(0.05 + 0.4 * y * rng.random(400) + 0.1 * rng.random(400), 0, 1)
    assert threshold_for_costs(y, p, COST_FN, COST_FP) == best_threshold(y, p)


def test_xgb_early_stopping_uses_fewer_than_max_trees(small_data):
    X = pd.concat([small_data[0], small_data[1]], ignore_index=True)
    y = np.concatenate([small_data[3], small_data[4]])
    _, n = fit_xgb({"learning_rate": 0.3, "max_depth": 3}, X, y)
    assert 1 <= n < 1000


def test_rf_early_stopping_respects_cap_and_keeps_that_many_trees(small_data):
    X = pd.concat([small_data[0], small_data[1]], ignore_index=True)
    y = np.concatenate([small_data[3], small_data[4]])
    m, n = fit_rf({"max_depth": 5, "class_weight": "balanced"}, X, y)
    assert n <= RF_MAX_TREES and len(m.estimators_) == n == m.n_estimators


def test_select_winner_prefers_faster_model_when_tied():
    tuned = {"A": {"cv_cost_per_1000": 100, "cv_se": 10, "latency_p50_ms": 9},
             "B": {"cv_cost_per_1000": 105, "cv_se": 10, "latency_p50_ms": 2}}
    assert select_winner(tuned)[0] == "B"
    tuned["B"]["cv_cost_per_1000"] = 140
    assert select_winner(tuned)[0] == "A"


def test_end_to_end_small_run_and_shap(small_data, tmp_path):
    summary = run(small_data, n_iter=2, n_splits=3, out_dir=tmp_path, synthetic=True,
                  synthetic_n=1500)
    models, reports = tmp_path / "models" / "synthetic", tmp_path / "reports" / "synthetic"
    bundle = joblib.load(models / "best_model.pkl")
    X_te = small_data[2][bundle["features"]]
    proba = bundle["model"].predict_proba(X_te)[:, 1]
    assert proba.shape == (len(X_te),) and 0 < bundle["threshold"] < 1
    assert json.loads((models / "best_model_summary.json").read_text())["selected"] \
        == summary["selected"]
    assert len(pd.read_csv(reports / "tuning_trials.csv")) == 4   # 2 trials x 2 algorithms
    table = (reports / "comparison_table.md").read_text()
    assert "Flag nothing" in table and "Sensitivity" in table
    idx = explain.main(model_path=models / "best_model.pkl", out_dir=reports, data="synthetic",
                       synthetic_n=1500)
    assert len(idx) >= 3 and (reports / "shap_examples.png").stat().st_size > 5000
