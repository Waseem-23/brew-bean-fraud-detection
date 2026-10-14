# Baseline Model Metrics

- Strategy: `class_weight` | Model: Logistic Regression
- Cost matrix: FN = $100, FP = $10, TP/TN = $0
- Data: Task 1 processed splits (`data/processed/*.parquet`); 2,000 orders, fraud rate 3.8%
- Evaluated on the held-out test set: 300 orders, 11 fraud
- Threshold for the tuned model: 0.79 (chosen on cross-validated predictions over train+val; the test set is never used for tuning)

## Comparison table (test set)

| Model | Precision | Recall | F1 | FN | FP | Total cost ($) |
|---|---|---|---|---|---|---|
| Random classifier (expected) | 0.034 | 0.035 | 0.034 | 11 | 11 | 1,172 |
| Logistic Regression (class_weight), threshold 0.50 | 0.028 | 0.273 | 0.051 | 8 | 103 | 1,830 |
| Logistic Regression (class_weight), cost-tuned threshold | 0.083 | 0.091 | 0.087 | 10 | 11 | 1,110 |

## Ranking quality on the test set (threshold-free)

| Metric | Model | Random classifier |
|---|---|---|
| ROC-AUC | 0.547 | 0.500 |
| PR-AUC | 0.134 | 0.037 |

## Reading the results

- The random classifier is the floor (its row averages 500 random draws). The tuned baseline lowers cost by $62 (5%) compared with it.
- Compare the 0.50 row with the cost-tuned row: the tuned threshold is chosen to minimise cost, not to maximise recall or F1.
- F1 treats FP and FN equally, so it is for reference only. Total cost is the headline metric because a missed fraud costs 10x a false alarm.
- Caveat: the test set contains few fraud cases, so small cost differences are noise.
