# Brew & Bean Café: Fraud Detection

Fraud detection for the Brew & Bean Café online orders.

- **Task 1:** data integrity and leakage-prevention pipeline.
- **Task 2:** cost-sensitive baseline model.

## Project structure

```
brew_bean_fraud/
├── config/config.yaml          # all paths and split parameters
├── data/
│   ├── raw/                    # raw_transactions.csv (input)
│   ├── processed/              # train/val/test .parquet + fitted pipeline
│   └── splits/                 # saved split indices (committed)
├── docs/decision_log.md        # design decisions
├── models/                     # Task 2: baseline_model.joblib
├── reports/                    # Task 2: baseline_metrics.md
├── src/
│   ├── data_pipeline.py        # Task 1: main pipeline
│   ├── make_splits.py          # stratified split + saved indices
│   ├── baseline_model.py       # Task 2: cost-sensitive baseline model
│   ├── features/transformers.py# guard, feature engineer, leakage flagger
│   └── data/generate_sample_data.py  # how the sample data was generated
├── tests/
│   ├── test_pipeline.py        # Task 1 tests
│   └── test_baseline.py        # Task 2 tests
└── requirements.txt
```

## Dependencies

Python 3.12, pandas, numpy, scikit-learn, imbalanced-learn, pyyaml, pyarrow,
joblib, pytest.

```
pip install -r requirements.txt
```

## Quick start

The raw data is not committed. Generate it, then run Task 1, Task 2 and the tests:

```
python -m src.data.generate_sample_data
python -m src.data_pipeline
python -m src.baseline_model
python -m pytest -v
```

---

# Task 1: Data integrity and leakage prevention

Loads raw transactions, cleans them, engineers features, checks for target
leakage, and writes reproducible train/val/test parquet files.

## Configuration

Everything is set in `config/config.yaml`; no paths are hard-coded.

| Key | Meaning |
|---|---|
| `paths.raw_data` | input CSV |
| `paths.processed_dir` | where parquet files are written |
| `paths.splits_dir` | where split indices are saved |
| `paths.pipeline_artifact` | fitted pipeline (joblib) |
| `target_column` | label column (`is_fraud`) |
| `random_state` | seed for reproducible splits |
| `split.*` | train / val / test fractions (0.70 / 0.15 / 0.15) |
| `leakage.correlation_threshold` | flag features with abs(corr) above this (0.8) |

## Pipeline steps

1. **Load** the raw CSV (path from config), parsing timestamps.
2. **Clean**: drop duplicate `transaction_id`s, rows with missing values or
   `amount <= 0`, and rows where `account_created_at > order_time`.
3. **Remove PII**: `customer_name`, `email`, `ip_address`.
4. **Velocity feature**: number of the customer's orders in the previous 24 h
   (looks only at the past).
5. **Stratified split** 70/15/15; indices saved to `data/splits/`.
6. **Fit the sklearn Pipeline on the training split only**:
   - `FutureTimestampGuard`: raises if any timestamp is after the reference time
   - `FraudFeatureEngineer`: account age, order hour, high-amount flag
     (threshold learned from train only)
   - `ColumnTransformer`: scale numeric, one-hot encode categorical
   - `LeakageFlagger`: warns for features with abs(corr) > 0.8 to the target
7. **Transform** train / val / test and save as parquet in `data/processed/`.

Outputs: `data/processed/{train,val,test}.parquet`,
`data/processed/fraud_pipeline.joblib`, `data/splits/{train,val,test}_idx.csv`.

## Leakage checks

- Scaler, encoder, amount threshold and leakage flagger are fitted on train only.
- Future timestamps are rejected by `FutureTimestampGuard`.
- Accounts created after the order are dropped during cleaning.
- `LeakageFlagger` flags any feature with abs(corr) > 0.8 to the target.
- Tests verify all of the above; see `docs/decision_log.md` for limitations.

---

# Task 2: Baseline model (cost-sensitive)

A Logistic Regression baseline trained on the Task 1 splits, with the heavy
class imbalance handled explicitly and evaluated with a business cost matrix.

## How to run

```
python -m src.baseline_model                    # class_weight (default)
python -m src.baseline_model --strategy smote   # SMOTE variant
python -m src.baseline_model --data synthetic   # make_classification test data
```

The script loads `data/processed/{train,val,test}.parquet` (already scaled and
encoded by Task 1, so no extra scaling is applied), trains the model, and writes
`models/baseline_model.joblib` and `reports/baseline_metrics.md`.

## Design decisions

- **Imbalance handling:** `class_weight="balanced"`. It reweights the loss, so no
  synthetic rows are created, every training row is a real order, and there is
  nothing to re-fit inside cross-validation folds. This keeps the pipeline simple
  and auditable. SMOTE is available (`--strategy smote`, training data only) for
  comparison in Task 3. The justification is also in the code comments.
- **Business cost:** a false negative (missed fraud) costs $100 and a false
  positive (blocked or reviewed genuine order) costs $10. Total cost is the
  headline metric; precision, recall and F1 are reported for reference.
- **Threshold:** chosen to minimise total cost on cross-validated (out-of-fold)
  predictions over train+val. The test set is never used for tuning and is used
  once for the final report.
- **Random baseline:** a classifier that flags orders at the training fraud rate,
  averaged over 500 random draws.

## Results (test set: 300 orders, 11 fraud)

| Model | Precision | Recall | Total cost |
|---|---|---|---|
| Random classifier (expected) | 0.034 | 0.035 | $1,172 |
| Logistic Regression, threshold 0.50 | 0.028 | 0.273 | $1,830 |
| Logistic Regression, cost-tuned threshold (0.79) | 0.083 | 0.091 | $1,110 |

Test ROC-AUC is about 0.55. Full details are in `reports/baseline_metrics.md`.

**Finding:** the baseline is only slightly cheaper than a random classifier and
about equal to flagging nothing ($1,100), so the current features carry weak
signal. The test set has only 11 fraud cases, so these differences are noisy.
This is the bar the Task 3 models need to beat.

## Tests

```
python -m pytest -q
```

`tests/test_baseline.py` checks that SMOTE gives the expected training class
distribution (balanced, and a partial ratio), that validation and test data are
never resampled, the class-weight ratio, the cost matrix arithmetic, the loader
for the Task 1 splits, and the out-of-fold probabilities.