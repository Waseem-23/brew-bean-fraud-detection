"""Advanced model selection and tuning for Brew & Bean Café (Task 3).

Run:  python -m src.model_tuning [--n-iter 30] [--n-splits 5]
                                 [--data processed|synthetic]

What it does
------------
1. Loads the Task 1 splits (data/processed/*.parquet) and pools train + val
   for model selection. The TEST set is never used to tune or to choose.
2. Randomized search (default 30 trials per algorithm) for XGBoost and
   RandomForest. Each trial is scored with stratified K-fold CV on the SAME
   business cost as Task 2 (FN = $100, FP = $10, imported from baseline_model),
   using the cost-optimal threshold found on pooled out-of-fold predictions.
3. Early stopping inside every fold, on a validation slice taken from the
   TRAINING part of that fold (never the CV validation fold):
     * XGBoost: native early stopping.
     * RandomForest: grown in chunks (warm_start), stopped when validation
       log-loss stops improving.
4. Refits the best configuration of each algorithm on train + val, scores it
   once on the test set, and compares it with the Task 2 logistic-regression
   recipe and the trivial "flag nothing" policy.
5. Picks the winner by CROSS-VALIDATED cost (test set = confirmation only).

Outputs
-------
models/best_model.pkl, models/best_model_summary.json,
reports/comparison_table.md, reports/tuning_trials.csv, reports/best_params.json,
logs/tuning.log.   (`--data synthetic` writes into a `synthetic/` sub-folder so
real results are never overwritten.)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split

from src.baseline_model import (
    COST_FN,
    COST_FP,
    ROOT,
    SEED,
    best_threshold,
    evaluate,
    fit_model,
    load_processed,
    make_data,
    oof_proba,
    split_data,
    total_cost,
)

log = logging.getLogger("tuning")

INNER_VAL = 0.20          # slice of each TRAINING fold used only for early stopping
XGB_MAX_TREES, XGB_PATIENCE = 1000, 30
RF_STEP, RF_MAX_TREES, RF_PATIENCE = 50, 300, 2
SENSITIVITY_RATIOS = (2, 5, 10, 20, 50)   # FN:FP cost ratios (the client's is 10:1)


# --------------------------------------------------------------------------- data
def _as_frame(X, cols=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    return pd.DataFrame(np.asarray(X), columns=cols or [f"f{i}" for i in range(np.shape(X)[1])])


def load_splits(data: str = "processed", synthetic_n: int = 3000, data_dir=None):
    """Return X_tr, X_val, X_te, y_tr, y_val, y_te (DataFrames / numpy arrays)."""
    if data == "synthetic":
        X, y = make_data(n_samples=synthetic_n)
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)
    else:
        X_tr, X_val, X_te, y_tr, y_val, y_te = load_processed(
            data_dir or ROOT / "data" / "processed")
    return (_as_frame(X_tr), _as_frame(X_val), _as_frame(X_te),
            np.asarray(y_tr), np.asarray(y_val), np.asarray(y_te))


# --------------------------------------------------------------------------- cost helpers
def cost_per_1000(cost: float, n: int) -> float:
    return 1000.0 * cost / n


def threshold_for_costs(y_true, proba, cost_fn: float, cost_fp: float) -> float:
    """Same grid and tie-breaking as Task 2's best_threshold, but with explicit costs
    (used only for the cost-ratio sensitivity table)."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y_true, (proba >= t).astype(int), cost_fn, cost_fp) for t in grid]
    best = np.flatnonzero(np.isclose(costs, min(costs)))
    return float(grid[best[len(best) // 2]])


# --------------------------------------------------------------------------- search spaces
def xgb_space() -> dict:
    return {
        "learning_rate": st.loguniform(0.02, 0.3),
        "max_depth": st.randint(2, 7),
        "min_child_weight": st.randint(1, 10),
        "subsample": st.uniform(0.6, 0.4),
        "colsample_bytree": st.uniform(0.5, 0.5),
        "reg_lambda": st.loguniform(0.1, 20),
        "reg_alpha": st.loguniform(1e-3, 5),
        "gamma": st.uniform(0, 3),
        # 1 = ignore imbalance ... 30 = very heavy up-weighting of fraud (cost ratio is 10:1)
        "scale_pos_weight": [1, 3, 10, 30],
    }


def rf_space() -> dict:
    return {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": st.randint(1, 25),
        "min_samples_split": st.randint(2, 20),
        "max_features": ["sqrt", "log2", 0.5],
        "max_samples": st.uniform(0.5, 0.5),
        "class_weight": [None, "balanced", "balanced_subsample", "cost"],  # "cost" = {1: FN/FP}
    }


SPACES = {"xgboost": xgb_space, "random_forest": rf_space}


def _clean(params: dict) -> dict:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}


# --------------------------------------------------------------------------- fitting
def fit_xgb(params, X, y, n_estimators=None, seed=SEED):
    common = dict(objective="binary:logistic", tree_method="hist", n_jobs=1,
                  random_state=seed, verbosity=0, **params)
    if n_estimators is not None:  # final refit: tree count already known
        return xgb.XGBClassifier(n_estimators=n_estimators, **common).fit(X, y), n_estimators
    X_a, X_v, y_a, y_v = train_test_split(X, y, test_size=INNER_VAL, stratify=y,
                                          random_state=seed)
    m = xgb.XGBClassifier(n_estimators=XGB_MAX_TREES, eval_metric="logloss",
                          early_stopping_rounds=XGB_PATIENCE, **common)
    m.fit(X_a, y_a, eval_set=[(X_v, y_v)], verbose=False)
    return m, int(m.best_iteration) + 1


def fit_rf(params, X, y, n_estimators=None, seed=SEED):
    p = dict(params)
    if p.get("class_weight") == "cost":
        p["class_weight"] = {0: 1.0, 1: COST_FN / COST_FP}
    base = dict(bootstrap=True, n_jobs=1, random_state=seed, **p)
    if n_estimators is not None:
        return RandomForestClassifier(n_estimators=n_estimators, **base).fit(X, y), n_estimators
    X_a, X_v, y_a, y_v = train_test_split(X, y, test_size=INNER_VAL, stratify=y,
                                          random_state=seed)
    rf = RandomForestClassifier(n_estimators=RF_STEP, warm_start=True, **base)
    best_loss, best_n, stale = np.inf, RF_STEP, 0
    while True:
        with warnings.catch_warnings():  # same data every call, so the warm_start caveat is moot
            warnings.filterwarnings("ignore", message="class_weight presets")
            rf.fit(X_a, y_a)
        loss = log_loss(y_v, rf.predict_proba(X_v)[:, 1].clip(1e-6, 1 - 1e-6), labels=[0, 1])
        if loss < best_loss - 1e-4:
            best_loss, best_n, stale = loss, rf.n_estimators, 0
        else:
            stale += 1
        if stale >= RF_PATIENCE or rf.n_estimators >= RF_MAX_TREES:
            break
        rf.n_estimators += RF_STEP
    rf.estimators_ = rf.estimators_[:best_n]   # keep the best-scoring forest size
    rf.n_estimators = best_n
    return rf, best_n


FITTERS = {"xgboost": fit_xgb, "random_forest": fit_rf}


# --------------------------------------------------------------------------- CV scoring
def fold_stats(y, oof, folds, thr) -> tuple[float, float]:
    """(mean, standard error) of per-1000-order cost across folds at a fixed threshold."""
    c = [cost_per_1000(total_cost(y[va], (oof[va] >= thr).astype(int)), len(va))
         for va in folds]
    return float(np.mean(c)), float(np.std(c, ddof=1) / np.sqrt(len(c)))


def cv_score(family, params, X, y, n_splits, seed=SEED) -> dict:
    t0 = time.perf_counter()
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    folds = [va for _, va in skf.split(X, y)]
    oof, trees = np.zeros(len(y)), []
    for tr, va in skf.split(X, y):
        model, n_trees = FITTERS[family](params, X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        trees.append(n_trees)
    thr = best_threshold(y, oof)
    cost = total_cost(y, (oof >= thr).astype(int))
    _, se = fold_stats(y, oof, folds, thr)
    return {
        "cv_cost": cost, "cv_cost_per_1000": cost_per_1000(cost, len(y)), "cv_se": se,
        "threshold": thr, "cv_pr_auc": float(average_precision_score(y, oof)),
        "cv_roc_auc": float(roc_auc_score(y, oof)),
        "median_trees": int(np.median(trees)), "cv_seconds": time.perf_counter() - t0,
        "oof": oof,
    }


def random_search(family, X, y, n_iter, n_splits, trial_rows) -> dict:
    sampler = ParameterSampler(SPACES[family](), n_iter=n_iter, random_state=SEED)
    best = None
    for i, params in enumerate(sampler, 1):
        params = _clean(params)
        try:
            res = cv_score(family, params, X, y, n_splits)
        except Exception as exc:  # one bad trial must not kill the whole search
            log.exception("trial %d failed", i)
            trial_rows.append({"model": family, "trial": i, "status": f"failed: {exc}",
                               "params": json.dumps(params, default=str)})
            continue
        trial_rows.append({
            "model": family, "trial": i, "status": "ok",
            **{k: v for k, v in res.items() if k != "oof"},
            "params": json.dumps(params, default=str)})
        log.info("[%s %02d/%d] cv_cost=$%.0f (%.1f/1000 +-%.1f) thr=%.2f prAUC=%.3f trees=%d "
                 "%.1fs %s", family, i, n_iter, res["cv_cost"], res["cv_cost_per_1000"],
                 res["cv_se"], res["threshold"], res["cv_pr_auc"], res["median_trees"],
                 res["cv_seconds"], params)
        if best is None or res["cv_cost"] < best["cv_cost"]:
            best = {**res, "params": params, "trial": i}
    if best is None:
        raise RuntimeError(f"every trial failed for {family}")
    log.info("BEST %s: trial %d cv_cost=$%.0f params=%s", family, best["trial"],
             best["cv_cost"], best["params"])
    return best


# --------------------------------------------------------------------------- final evaluation
def latency_ms(model, X: pd.DataFrame, n: int = 200) -> tuple[float, float]:
    """Single-order scoring time, i.e. what the real-time API will see."""
    model.predict_proba(X.iloc[[0]])
    ts = []
    for i in range(n):
        row = X.iloc[[i % len(X)]]
        t0 = time.perf_counter()
        model.predict_proba(row)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


def score_on_test(label, model, thr, X_te, y_te, cv, fit_s, tmp_dir: Path) -> dict:
    proba = model.predict_proba(X_te)[:, 1]
    ev = evaluate(y_te, (proba >= thr).astype(int))
    p50, p95 = latency_ms(model, X_te)
    tmp = tmp_dir / "_size_probe.pkl"
    joblib.dump(model, tmp)
    size_kb = tmp.stat().st_size / 1024
    tmp.unlink()
    return {
        "label": label, "threshold": thr, "proba": proba, **ev,
        "test_cost_per_1000": cost_per_1000(ev["total_cost"], len(y_te)),
        "test_roc_auc": float(roc_auc_score(y_te, proba)),
        "test_pr_auc": float(average_precision_score(y_te, proba)),
        "cv_cost_per_1000": cv["cv_cost_per_1000"], "cv_se": cv["cv_se"],
        "fit_seconds": fit_s, "latency_p50_ms": p50, "latency_p95_ms": p95,
        "size_kb": size_kb,
    }


def select_winner(tuned: dict) -> tuple[str, str]:
    """Lowest CV cost wins. If several models are within one standard error of the
    best, they are statistically tied, so the faster-to-serve one is preferred."""
    best_key = min(tuned, key=lambda k: tuned[k]["cv_cost_per_1000"])
    limit = tuned[best_key]["cv_cost_per_1000"] + tuned[best_key]["cv_se"]
    tied = [k for k, v in tuned.items() if v["cv_cost_per_1000"] <= limit]
    winner = min(tied, key=lambda k: tuned[k]["latency_p50_ms"]) if len(tied) > 1 else best_key
    if len(tied) > 1:
        reason = ("CV cost tied within one standard error "
                  f"({', '.join(tied)}); the faster model was preferred")
    else:
        reason = "lowest cross-validated cost"
    return winner, reason


def sensitivity_table(rows: dict, oofs: dict, y_pool, y_te) -> pd.DataFrame:
    out = []
    for r in SENSITIVITY_RATIOS:
        fn_cost = r * COST_FP
        rec = {"FN:FP ratio": f"{r}:1" + (" (client)" if r == COST_FN / COST_FP else "")}
        for label, row in rows.items():
            thr = threshold_for_costs(y_pool, oofs[label], fn_cost, COST_FP)
            c = total_cost(y_te, (row["proba"] >= thr).astype(int), fn_cost, COST_FP)
            rec[label] = round(cost_per_1000(c, len(y_te)), 1)
        rec["Flag nothing"] = round(cost_per_1000(int(y_te.sum()) * fn_cost, len(y_te)), 1)
        out.append(rec)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- report
def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_comparison(path: Path, rows: list[dict], sens: pd.DataFrame, winner: str,
                     reason: str, meta: dict) -> None:
    n_te, n_fraud = meta["n_test"], meta["n_test_fraud"]
    L = ["# Task 3: model comparison on the cost metric", ""]
    if meta["synthetic"]:
        L += ["> **WARNING: synthetic data.** Numbers only demonstrate the method.", ""]
    L += [
        f"- Cost matrix (same as Task 2): FN = ${COST_FN:.0f}, FP = ${COST_FP:.0f}, TP/TN = $0.",
        f"- Search: {meta['n_iter']} randomized trials per algorithm, "
        f"{meta['n_splits']}-fold stratified CV on train+val ({meta['n_pool']:,} orders, "
        f"{meta['n_pool_fraud']} fraud), early stopping on an inner split.",
        "- Thresholds are chosen on out-of-fold predictions only. The test set "
        f"({n_te:,} orders, {n_fraud} fraud) was used once, for confirmation.",
        f"- Selection rule: {reason}.", "",
        "| Model | CV cost /1000 (± SE) | Test cost ($) | Test cost /1000 | FN | FP | Recall | "
        "Precision | ROC-AUC | PR-AUC | Threshold | Fit (s) | Latency p50 / p95 (ms) | "
        "Size (KB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["total_cost"]):
        name = f"**{r['label']}** ✅" if r["label"] == winner else r["label"]
        L.append(
            f"| {name} | {r['cv_cost_per_1000']:.1f} ± {r['cv_se']:.1f} | "
            f"{r['total_cost']:,.0f} | {r['test_cost_per_1000']:.1f} | {r['fn']} | {r['fp']} | "
            f"{r['recall']:.2f} | {r['precision']:.2f} | {r['test_roc_auc']:.3f} | "
            f"{r['test_pr_auc']:.3f} | {r['threshold']:.2f} | {r['fit_seconds']:.2f} | "
            f"{r['latency_p50_ms']:.1f} / {r['latency_p95_ms']:.1f} | {r['size_kb']:.0f} |")
    none_cost = n_fraud * COST_FN
    L.append(f"| _Flag nothing (no model)_ | - | {none_cost:,.0f} | "
             f"{cost_per_1000(none_cost, n_te):.1f} | {n_fraud} | 0 | 0.00 | - | - | - | - | - "
             "| - | - |")
    w = next(r for r in rows if r["label"] == winner)
    base = next(r for r in rows if "baseline" in r["label"])
    gap_nothing = none_cost - w["total_cost"]
    gap_base = base["total_cost"] - w["total_cost"]
    one_fraud = COST_FN
    L += [
        "", "✅ = selected model.", "",
        "## Reading the results", "",
        f"- Selected model vs flag-nothing on the test set: "
        f"{'saves' if gap_nothing > 0 else 'costs'} ${abs(gap_nothing):,.0f} "
        f"({abs(gap_nothing) / one_fraud:.1f} fraud cases' worth).",
        f"- Selected model vs the Task 2 baseline: "
        f"{'saves' if gap_base > 0 else 'costs'} ${abs(gap_base):,.0f}.",
        f"- **Noise warning:** the test set has only {n_fraud} fraud cases and one missed "
        f"fraud costs ${one_fraud:.0f}, so gaps of a few hundred dollars are within noise. "
        "Trust the CV column (more fraud cases) more than the test column.",
    ]
    if w["total_cost"] >= none_cost:
        L.append("- **The selected model does not beat 'flag nothing' on the test set.** "
                 "With the current features, flagging is not yet worth its false alarms.")
    L += ["", "## Sensitivity to the cost assumption", "",
          "Test cost per 1,000 orders if the FN:FP cost ratio is different (FP fixed at "
          f"${COST_FP:.0f}; each model's threshold is re-chosen on out-of-fold predictions).",
          "", _md_table(sens), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- pipeline
def run(splits, n_iter=30, n_splits=5, out_dir: Path = ROOT, synthetic=False,
        synthetic_n=3000) -> dict:
    sub = "synthetic" if synthetic else ""
    models_dir, reports_dir, logs_dir = (out_dir / "models" / sub, out_dir / "reports" / sub,
                                         out_dir / "logs")
    for d in (models_dir, reports_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_dir / ("tuning_synthetic.log" if synthetic else "tuning.log"),
                             mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    log.setLevel(logging.INFO)

    X_tr, X_val, X_te, y_tr, y_val, y_te = splits
    X_pool = pd.concat([X_tr, X_val], ignore_index=True)
    y_pool = np.concatenate([y_tr, y_val])
    feats = list(X_pool.columns)
    log.info("pool=%s fraud=%d | test=%s fraud=%d | synthetic=%s | n_iter=%d folds=%d",
             X_pool.shape, y_pool.sum(), X_te.shape, y_te.sum(), synthetic, n_iter, n_splits)

    rows, oofs, models, trials, params_out = {}, {}, {}, [], {}

    # Task 2 recipe (logistic regression, class_weight) under the SAME protocol
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    folds = [va for _, va in skf.split(X_pool, y_pool)]
    oof_lr = oof_proba(X_pool, y_pool, "class_weight", n_splits)
    thr_lr = best_threshold(y_pool, oof_lr)
    cost_lr = total_cost(y_pool, (oof_lr >= thr_lr).astype(int))
    _, se_lr = fold_stats(y_pool, oof_lr, folds, thr_lr)
    cv_lr = {"cv_cost_per_1000": cost_per_1000(cost_lr, len(y_pool)), "cv_se": se_lr}
    t0 = time.perf_counter()
    lr = fit_model(X_pool, y_pool, "class_weight")
    lr_label = "Logistic Regression (Task 2 baseline recipe)"
    rows[lr_label] = score_on_test(lr_label, lr, thr_lr, X_te, y_te, cv_lr,
                                   time.perf_counter() - t0, models_dir)
    oofs[lr_label], models[lr_label] = oof_lr, lr

    names = {"xgboost": "XGBoost (tuned)", "random_forest": "RandomForest (tuned)"}
    best = {}
    for family in ("xgboost", "random_forest"):
        t0 = time.perf_counter()
        best[family] = random_search(family, X_pool, y_pool, n_iter, n_splits, trials)
        best[family]["search_seconds"] = time.perf_counter() - t0
        b = best[family]
        # the final refit sees ~25% more rows than a CV fold, so allow ~10% more trees
        n_final = max(10, int(round(b["median_trees"] * 1.1)))
        t0 = time.perf_counter()
        model, _ = FITTERS[family](b["params"], X_pool, y_pool, n_estimators=n_final)
        fit_s = time.perf_counter() - t0
        rows[names[family]] = score_on_test(names[family], model, b["threshold"], X_te, y_te,
                                            b, fit_s, models_dir)
        oofs[names[family]], models[names[family]] = b["oof"], model
        params_out[family] = {"params": b["params"], "n_estimators": n_final,
                              "threshold": b["threshold"], "cv_cost": b["cv_cost"],
                              "cv_cost_per_1000": b["cv_cost_per_1000"],
                              "search_seconds": b["search_seconds"], "trial": b["trial"]}

    pd.DataFrame(trials).to_csv(reports_dir / "tuning_trials.csv", index=False)
    (reports_dir / "best_params.json").write_text(json.dumps(params_out, indent=2, default=str))

    tuned = {k: v for k, v in rows.items() if "(tuned)" in k}
    winner, reason = select_winner(tuned)
    log.info("SELECTED %s (%s)", winner, reason)

    meta = {"synthetic": synthetic, "n_iter": n_iter, "n_splits": n_splits,
            "n_pool": len(y_pool), "n_pool_fraud": int(y_pool.sum()),
            "n_test": len(y_te), "n_test_fraud": int(y_te.sum())}
    sens = sensitivity_table(rows, oofs, y_pool, y_te)
    write_comparison(reports_dir / "comparison_table.md", list(rows.values()), sens, winner,
                     reason, meta)

    w = rows[winner]
    family = next(f for f, n in names.items() if n == winner)
    summary = {
        "selected": winner, "algorithm": family, "selection_reason": reason,
        "params": params_out[family]["params"], "n_estimators": params_out[family]["n_estimators"],
        "threshold": w["threshold"], "cost_fn": COST_FN, "cost_fp": COST_FP,
        "trained_on_synthetic_data": synthetic,
        "cv": {"cost_per_1000": w["cv_cost_per_1000"], "standard_error": w["cv_se"],
               "folds": n_splits},
        "test": {k: (float(w[k]) if isinstance(w[k], (float, np.floating)) else w[k])
                 for k in ("total_cost", "test_cost_per_1000", "tp", "fp", "fn", "tn",
                           "precision", "recall", "test_roc_auc", "test_pr_auc")},
        "serving": {k: float(w[k]) for k in
                    ("latency_p50_ms", "latency_p95_ms", "size_kb", "fit_seconds")},
        "all_candidates_test_cost": {k: v["total_cost"] for k, v in rows.items()},
        "features": feats, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgb.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    joblib.dump({"model": models[winner], "threshold": w["threshold"], "features": feats,
                 "algorithm": family, "params": summary["params"], "cost_fn": COST_FN,
                 "cost_fp": COST_FP, "trained_on_synthetic_data": synthetic,
                 "synthetic_n": synthetic_n, "versions": summary["versions"]},
                models_dir / "best_model.pkl")
    (models_dir / "best_model_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("Saved %s", models_dir / "best_model.pkl")
    log.removeHandler(fh)
    fh.close()
    return summary


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Task 3: tune and select the fraud model")
    ap.add_argument("--n-iter", type=int, default=30, help="trials per algorithm")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--data", choices=["processed", "synthetic"], default="processed")
    ap.add_argument("--synthetic-n", type=int, default=3000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    synthetic = args.data == "synthetic"
    splits = load_splits(args.data, args.synthetic_n)
    summary = run(splits, args.n_iter, args.n_splits, ROOT, synthetic, args.synthetic_n)
    print(f"\nSelected: {summary['selected']} | test cost ${summary['test']['total_cost']:,.0f} "
          f"| threshold {summary['threshold']:.2f}")
    return summary


if __name__ == "__main__":
    main()
