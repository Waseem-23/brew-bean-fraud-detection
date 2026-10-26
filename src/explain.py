"""SHAP explanations for the selected Task 3 model.

Run:  python -m src.explain [--data processed|synthetic]

Writes
  reports/shap_examples.png  - why the model scored 3-4 individual TEST orders as it did
  reports/shap_global.png    - mean |SHAP| per feature over the whole test set

Note: Task 1 scales/encodes the features, so the numbers next to each feature name
are the scaled values the model actually saw, not raw dollars or hours.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.baseline_model import ROOT  # noqa: E402
from src.model_tuning import load_splits  # noqa: E402

log = logging.getLogger("explain")


def shap_values_fraud_class(model, X, algorithm: str):
    """(values, base value, unit) for the fraud class, across SHAP versions."""
    ex = shap.TreeExplainer(model)
    sv, base = ex.shap_values(X), ex.expected_value
    if isinstance(sv, list):                 # old SHAP: one array per class
        sv, base = sv[1], np.atleast_1d(base)[1]
    elif np.ndim(sv) == 3:                   # new SHAP: (rows, features, classes)
        sv, base = sv[:, :, 1], np.atleast_1d(base)[1]
    unit = "log-odds" if algorithm == "xgboost" else "probability"
    return np.asarray(sv), float(np.atleast_1d(base)[0]), unit


def pick_instances(proba, y, thr) -> list[int]:
    """Up to 4 distinct test orders: highest-scored fraud, lowest-scored fraud,
    highest-scored legitimate order, lowest-scored legitimate order. Titles show the
    real outcome (caught / missed / false alarm / correctly approved)."""
    fraud, legit = np.where(y == 1)[0], np.where(y == 0)[0]
    cands = []
    if len(fraud):
        cands += [fraud[np.argmax(proba[fraud])], fraud[np.argmin(proba[fraud])]]
    if len(legit):
        cands += [legit[np.argmax(proba[legit])], legit[np.argmin(proba[legit])]]
    out = []
    for c in cands:
        if int(c) not in out:
            out.append(int(c))
    return out


def outcome(y_i: int, flagged: bool) -> str:
    return {(1, True): "Fraud CAUGHT", (1, False): "Fraud MISSED",
            (0, True): "FALSE ALARM", (0, False): "Normal order approved"}[(int(y_i), flagged)]


def main(model_path=None, out_dir=None, data=None, synthetic_n=None) -> list[int]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if model_path is None:
        sub = "synthetic" if data == "synthetic" else ""
        model_path = ROOT / "models" / sub / "best_model.pkl"
    bundle = joblib.load(model_path)
    synthetic = bundle["trained_on_synthetic_data"] if data is None else data == "synthetic"
    sub = "synthetic" if synthetic else ""
    out_dir = Path(out_dir) if out_dir else ROOT / "reports" / sub
    model, feats, thr, algo = (bundle["model"], bundle["features"], bundle["threshold"],
                               bundle["algorithm"])
    splits = load_splits("synthetic" if synthetic else "processed",
                         synthetic_n or bundle.get("synthetic_n", 3000))
    X_te, y_te = splits[2][feats].reset_index(drop=True), splits[5]
    proba = model.predict_proba(X_te)[:, 1]
    sv, base, unit = shap_values_fraud_class(model, X_te, algo)

    idx = pick_instances(proba, y_te, thr)
    n = len(idx)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4.6 * rows), squeeze=False)
    axes = axes.ravel()
    for ax, i in zip(axes, idx):
        order = np.argsort(-np.abs(sv[i]))[:8][::-1]
        vals = sv[i][order]
        labels = [f"{feats[j]} = {X_te.iloc[i, j]:.2f}" for j in order]
        ax.barh(labels, vals, color=["#d62728" if v > 0 else "#1f77b4" for v in vals])
        ax.axvline(0, color="k", lw=0.8)
        flagged = bool(proba[i] >= thr)
        ax.set_title(f"{outcome(y_te[i], flagged)} (test row {i})\n"
                     f"P(fraud)={proba[i]:.2f}, threshold={thr:.2f} -> "
                     f"{'HOLD FOR REVIEW' if flagged else 'APPROVE'}", fontsize=10)
        ax.set_xlabel(f"SHAP value ({unit}): red pushes toward fraud, blue toward legitimate")
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Why the model scored these test orders as it did "
                 f"({algo}; average prediction = {base:.2f} {unit})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "shap_examples.png", dpi=140)
    plt.close(fig)

    imp = np.abs(sv).mean(0)
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(feats) + 1.5)))
    ax.barh([feats[j] for j in order], imp[order], color="#555")
    ax.set_xlabel(f"mean |SHAP| ({unit})")
    ax.set_title("Global feature importance (test set)")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_global.png", dpi=140)
    plt.close(fig)
    log.info("Saved SHAP figures to %s (test rows %s)", out_dir, idx)
    return idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["processed", "synthetic"], default=None,
                    help="processed = models/best_model.pkl (default); "
                         "synthetic = models/synthetic/best_model.pkl")
    main(data=ap.parse_args().data or "processed")
