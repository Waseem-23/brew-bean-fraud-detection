import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.baseline_model import (
    COST_FN,
    COST_FP,
    build_classifier,
    fit_model,
    load_processed,
    make_data,
    oof_proba,
    resample_training_data,
    split_data,
    total_cost,
)


def _split():
    X, y = make_data(n_samples=6000, fraud_rate=0.02)
    return split_data(X, y)


def test_smote_balances_training_classes():
    X_tr, _, _, y_tr, _, _ = _split()
    assert y_tr.mean() < 0.05  # heavy imbalance before resampling
    _, y_res = resample_training_data(X_tr, y_tr, sampling_strategy=1.0)
    counts = np.bincount(y_res)
    assert counts[0] == counts[1]
    assert counts[0] == np.bincount(y_tr)[0]  # majority class untouched


def test_smote_partial_ratio():
    X_tr, _, _, y_tr, _, _ = _split()
    _, y_res = resample_training_data(X_tr, y_tr, sampling_strategy=0.5)
    counts = np.bincount(y_res)
    assert abs(counts[1] / counts[0] - 0.5) < 0.01


def test_resampling_never_touches_validation_or_test():
    X_tr, _, _, y_tr, y_val, y_te = _split()
    before = (len(y_val), y_val.mean(), len(y_te), y_te.mean())
    fit_model(X_tr, y_tr, "smote")
    assert before == (len(y_val), y_val.mean(), len(y_te), y_te.mean())
    assert y_te.mean() < 0.05  # test set keeps the real, imbalanced distribution


def test_class_weight_balanced_ratio():
    _, _, _, y_tr, _, _ = _split()
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    counts = np.bincount(y_tr)
    assert abs(w[1] / w[0] - counts[0] / counts[1]) < 1e-6
    assert build_classifier("class_weight").class_weight == "balanced"


def test_total_cost_matrix():
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([0, 1, 1, 1, 0])  # 1 FN, 2 FP
    assert total_cost(y_true, y_pred) == COST_FN + 2 * COST_FP


def test_load_processed_reads_task1_splits(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(0)
    for name, n in [("train", 100), ("val", 30), ("test", 30)]:
        df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
        df["is_fraud"] = (rng.random(n) < 0.1).astype(int)
        df.to_parquet(tmp_path / f"{name}.parquet")
    X_tr, X_val, X_te, y_tr, _, _ = load_processed(tmp_path)
    assert list(X_tr.columns) == ["a", "b", "c"] and "is_fraud" not in X_tr
    assert len(X_tr) == len(y_tr) == 100 and len(X_val) == 30 and len(X_te) == 30


def test_oof_proba_is_valid_and_complete():
    X_tr, _, _, y_tr, _, _ = _split()
    p = oof_proba(X_tr, y_tr, "class_weight")
    assert p.shape == y_tr.shape and p.min() >= 0 and p.max() <= 1
    assert p[y_tr == 1].mean() > p[y_tr == 0].mean()  # model has signal on synthetic data