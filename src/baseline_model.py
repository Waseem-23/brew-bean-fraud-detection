"""Baseline fraud model for Brew & Bean Café (Task 2).

Run:  python -m src.baseline_model [--data processed|synthetic]
                                    [--strategy class_weight|smote]

Pipeline: Task 1 splits (data/processed/*.parquet, already scaled and encoded
by the Task 1 pipeline) -> imbalance handling on TRAIN only -> Logistic
Regression -> cost-optimal threshold chosen on cross-validated predictions
over train+val -> final numbers reported on the untouched TEST set.
`--data synthetic` uses make_classification for quick local testing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---- Business cost matrix --------------------------------------------------
COST_FN = 100.0  # fraudulent order we let through (lost goods + chargeback)
COST_FP = 10.0   # legitimate order we block/review (staff time + unhappy customer)

SEED = 42
TARGET = "is_fraud"
ROOT = Path(__file__).resolve().parents[1]


# ---- Data ------------------------------------------------------------------
def make_data(n_samples: int = 20_000, fraud_rate: float = 0.02, seed: int = SEED):
    """Synthetic stand-in for the processed Task 1 data (heavy imbalance)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[1 - fraud_rate, fraud_rate],
        class_sep=1.0,
        flip_y=0.005,
        random_state=seed,
    )
    return X, y


def load_processed(data_dir: Path | str = ROOT / "data" / "processed"):
    """Load the Task 1 splits. They are already scaled/encoded, so no extra
    scaling is applied here (it would double-transform the features)."""
    parts = []
    for name in ("train", "val", "test"):
        df = pd.read_parquet(Path(data_dir) / f"{name}.parquet")
        parts.append((df.drop(columns=TARGET), df[TARGET].to_numpy()))
    cols = list(parts[0][0].columns)
    if any(list(X.columns) != cols for X, _ in parts):
        raise ValueError("train/val/test have different columns")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = parts
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def split_data(X, y, seed: int = SEED):
    """60/20/20 stratified split. Splitting happens BEFORE any resampling or
    scaling so no information from val/test leaks into training."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ---- Imbalance handling ----------------------------------------------------
def resample_training_data(X, y, sampling_strategy: float = 1.0, seed: int = SEED):
    """SMOTE oversampling of the minority class. TRAINING DATA ONLY.

    sampling_strategy = desired minority/majority ratio after resampling
    (1.0 -> perfectly balanced).
    """
    from imblearn.over_sampling import SMOTE

    return SMOTE(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)


def build_classifier(strategy: str) -> LogisticRegression:
    # JUSTIFICATION (default = class_weight='balanced'):
    #  * It reweights the loss, so no synthetic rows are created. Every training
    #    row is a real order, which keeps the pipeline auditable.
    #  * Nothing to re-fit inside each CV fold (SMOTE must be), so it is simpler
    #    and safer for a small IT team to maintain.
    #  * It penalises a missed fraud ~ (majority/minority)x more, which matches
    #    the business view that a false negative is the costlier error.
    # SMOTE is kept as an option (--strategy smote) so we can compare in Task 3.
    cw = "balanced" if strategy == "class_weight" else None
    return LogisticRegression(class_weight=cw, max_iter=1000, random_state=SEED)


def fit_model(X_train, y_train, strategy: str = "class_weight") -> LogisticRegression:
    """Fit on TRAIN only. SMOTE (if chosen) is applied to the training data only."""
    if strategy == "smote":
        X_train, y_train = resample_training_data(X_train, y_train)
    return build_classifier(strategy).fit(X_train, y_train)


# ---- Evaluation ------------------------------------------------------------
def total_cost(y_true, y_pred, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * cost_fn + fp * cost_fp)


def evaluate(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "total_cost": total_cost(y_true, y_pred),
    }


def best_threshold(y_true, proba) -> float:
    """Pick the probability cut-off that minimises business cost."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y_true, (proba >= t).astype(int)) for t in grid]
    best = np.flatnonzero(np.isclose(costs, min(costs)))
    return float(grid[best[len(best) // 2]])  # middle of the tied cheapest range


def oof_proba(X, y, strategy: str, n_splits: int = 5, seed: int = SEED) -> np.ndarray:
    """Out-of-fold fraud probabilities. Each row is scored by a model that never
    saw it, so the probabilities are honest. Used to choose the threshold on
    more fraud cases than the small validation set alone provides."""
    X, y = np.asarray(X), np.asarray(y)
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        out[te] = fit_model(X[tr], y[tr], strategy).predict_proba(X[te])[:, 1]
    return out


def random_baseline(y_true, fraud_rate: float, n_draws: int = 500, seed: int = SEED) -> dict:
    """Expected metrics of a classifier that flags each order as fraud with
    probability = training fraud rate (it ignores the features entirely)."""
    rng = np.random.default_rng(seed)
    rows = [
        evaluate(y_true, (rng.random(len(y_true)) < fraud_rate).astype(int))
        for _ in range(n_draws)
    ]
    return pd.DataFrame(rows).mean().to_dict()


# ---- Report ----------------------------------------------------------------
def write_report(path: Path, results: dict, meta: dict) -> None:
    names = list(results)
    lines = [
        "# Baseline Model Metrics",
        "",
        f"- Strategy: `{meta['strategy']}` | Model: Logistic Regression",
        f"- Cost matrix: FN = ${COST_FN:.0f}, FP = ${COST_FP:.0f}, TP/TN = $0",
        f"- Data: {meta['data_desc']}; {meta['n']:,} orders, "
        f"fraud rate {meta['fraud_rate']:.1%}",
        f"- Evaluated on the held-out test set: {meta['n_test']:,} orders, "
        f"{meta['n_test_fraud']} fraud",
        f"- Threshold for the tuned model: {meta['threshold']:.2f} (chosen on "
        "cross-validated predictions over train+val; the test set is never used for tuning)",
        "",
        "## Comparison table (test set)",
        "",
        "| Model | Precision | Recall | F1 | FN | FP | Total cost ($) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        r = results[name]
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['fn']:.0f} | {r['fp']:.0f} | {r['total_cost']:,.0f} |"
        )
    rand, tuned = results[names[0]], results[names[-1]]
    saving = rand["total_cost"] - tuned["total_cost"]
    verb = "lowers" if saving > 0 else "does not lower"
    lines += [
        "",
        "## Ranking quality on the test set (threshold-free)",
        "",
        "| Metric | Model | Random classifier |",
        "|---|---|---|",
        f"| ROC-AUC | {meta['test_roc_auc']:.3f} | 0.500 |",
        f"| PR-AUC | {meta['test_pr_auc']:.3f} | "
        f"{meta['n_test_fraud'] / meta['n_test']:.3f} |",
    ]
    lines += [
        "",
        "## Reading the results",
        "",
        "- The random classifier is the floor (its row averages 500 random draws). "
        f"The tuned baseline {verb} cost by ${abs(saving):,.0f} "
        f"({abs(saving) / rand['total_cost']:.0%}) compared with it.",
        "- Compare the 0.50 row with the cost-tuned row: the tuned threshold is chosen "
        "to minimise cost, not to maximise recall or F1.",
        "- F1 treats FP and FN equally, so it is for reference only. Total cost is the "
        "headline metric because a missed fraud costs 10x a false alarm.",
        "- Caveat: the test set contains few fraud cases, so small cost differences "
        "are noise.",
    ]
    if tuned["recall"] == 0:
        lines.append(
            "- The cost-optimal rule flags no orders: with the current signal, flagging "
            "costs more in false alarms than it saves. Stronger features or models "
            "(Task 3) are needed before this can beat 'flag nothing'."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---- Main ------------------------------------------------------------------
def main(strategy: str = "class_weight", data: str = "processed", out_dir: Path = ROOT) -> dict:
    if data == "synthetic":
        X, y = make_data()
        splits = split_data(X, y)
        desc = "synthetic (`make_classification`), 60/20/20 stratified split"
    else:
        splits = load_processed(out_dir / "data" / "processed")
        desc = "Task 1 processed splits (`data/processed/*.parquet`)"
    X_tr, X_val, X_te, y_tr, y_val, y_te = splits

    model = fit_model(X_tr, y_tr, strategy)
    X_pool = np.vstack([np.asarray(X_tr), np.asarray(X_val)])
    y_pool = np.concatenate([y_tr, y_val])
    thr = best_threshold(y_pool, oof_proba(X_pool, y_pool, strategy))
    proba_te = model.predict_proba(X_te)[:, 1]

    label = f"Logistic Regression ({strategy})"
    results = {
        "Random classifier (expected)": random_baseline(y_te, y_tr.mean()),
        f"{label}, threshold 0.50": evaluate(y_te, (proba_te >= 0.5).astype(int)),
        f"{label}, cost-tuned threshold": evaluate(y_te, (proba_te >= thr).astype(int)),
    }

    n = len(y_tr) + len(y_val) + len(y_te)
    meta = {
        "strategy": strategy, "threshold": thr, "cost_fn": COST_FN, "cost_fp": COST_FP,
        "data_desc": desc, "n": n,
        "fraud_rate": float((y_tr.sum() + y_val.sum() + y_te.sum()) / n),
        "n_test": len(y_te), "n_test_fraud": int(y_te.sum()),
        "test_roc_auc": float(roc_auc_score(y_te, proba_te)),
        "test_pr_auc": float(average_precision_score(y_te, proba_te)),
        "features": list(getattr(X_tr, "columns", [])),
    }
    (out_dir / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "meta": meta},
                out_dir / "models" / "baseline_model.joblib")
    (out_dir / "models" / "baseline_meta.json").write_text(json.dumps(meta, indent=2))
    write_report(out_dir / "reports" / "baseline_metrics.md", results, meta)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["class_weight", "smote"], default="class_weight")
    ap.add_argument("--data", choices=["processed", "synthetic"], default="processed")
    args = ap.parse_args()
    res = main(args.strategy, args.data)
    meta = json.loads((ROOT / "models" / "baseline_meta.json").read_text())
    print(f"test fraud cases: {meta['n_test_fraud']} of {meta['n_test']} | "
          f"ROC-AUC={meta['test_roc_auc']:.3f} PR-AUC={meta['test_pr_auc']:.3f} | "
          f"threshold={meta['threshold']:.2f}")
    for k, v in res.items():
        print(f"{k}: cost=${v['total_cost']:,.0f} recall={v['recall']:.3f} "
              f"precision={v['precision']:.3f}")