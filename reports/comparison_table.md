# Task 3: model comparison on the cost metric

- Cost matrix (same as Task 2): FN = $100, FP = $10, TP/TN = $0.
- Search: 30 randomized trials per algorithm, 5-fold stratified CV on train+val (1,700 orders, 65 fraud), early stopping on an inner split.
- Thresholds are chosen on out-of-fold predictions only. The test set (300 orders, 11 fraud) was used once, for confirmation.
- Selection rule: CV cost tied within one standard error (XGBoost (tuned), RandomForest (tuned)); the faster model was preferred.

| Model | CV cost /1000 (± SE) | Test cost ($) | Test cost /1000 | FN | FP | Recall | Precision | ROC-AUC | PR-AUC | Threshold | Fit (s) | Latency p50 / p95 (ms) | Size (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **XGBoost (tuned)** ✅ | 3488.2 ± 86.6 | 1,040 | 3466.7 | 10 | 4 | 0.09 | 0.20 | 0.707 | 0.086 | 0.12 | 0.29 | 2.1 / 2.4 | 204 |
| Logistic Regression (Task 2 baseline recipe) | 3629.4 ± 83.0 | 1,080 | 3600.0 | 10 | 8 | 0.09 | 0.11 | 0.588 | 0.136 | 0.79 | 0.04 | 0.8 / 1.1 | 2 |
| RandomForest (tuned) | 3470.6 ± 141.4 | 1,120 | 3733.3 | 11 | 2 | 0.00 | 0.00 | 0.741 | 0.110 | 0.15 | 0.32 | 13.8 / 19.6 | 825 |
| _Flag nothing (no model)_ | - | 1,100 | 3666.7 | 11 | 0 | 0.00 | - | - | - | - | - | - | - |

✅ = selected model.

## Reading the results

- Selected model vs flag-nothing on the test set: saves $60 (0.6 fraud cases' worth).
- Selected model vs the Task 2 baseline: saves $40.
- **Noise warning:** the test set has only 11 fraud cases and one missed fraud costs $100, so gaps of a few hundred dollars are within noise. Trust the CV column (more fraud cases) more than the test column.

## Sensitivity to the cost assumption

Test cost per 1,000 orders if the FN:FP cost ratio is different (FP fixed at $10; each model's threshold is re-chosen on out-of-fold predictions).

| FN:FP ratio | Logistic Regression (Task 2 baseline recipe) | XGBoost (tuned) | RandomForest (tuned) | Flag nothing |
|---|---|---|---|---|
| 2:1 | 733.3 | 800.0 | 733.3 | 733.3 |
| 5:1 | 1833.3 | 1800.0 | 1900.0 | 1833.3 |
| 10:1 (client) | 3600.0 | 3466.7 | 3733.3 | 3666.7 |
| 20:1 | 7133.3 | 5833.3 | 5566.7 | 7333.3 |
| 50:1 | 9466.7 | 8300.0 | 7033.3 | 18333.3 |
