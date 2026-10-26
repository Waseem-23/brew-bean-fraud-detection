Brew & Bean Café: Fraud Detection

Fraud detection for the Brew & Bean Café online orders.

Task 1: data integrity and leakage-prevention pipeline.
Task 2: cost-sensitive baseline model.
Task 3: model tuning, selection and SHAP explainability.
Project structure
brew_bean_fraud/
├── config/config.yaml          # all paths and split parameters
├── data/
│   ├── raw/                    # raw_transactions.csv (input)
│   ├── processed/              # train/val/test .parquet + fitted pipeline
│   └── splits/                 # saved split indices (committed)
├── docs/decision_log.md        # design decisions
├── models/                     # baseline_model.joblib, best_model.pkl
├── reports/                    # metrics, comparison table, SHAP plots
├── src/
│   ├── data_pipeline.py        # Task 1: main pipeline
│   ├── make_splits.py          # stratified split + saved indices
│   ├── baseline_model.py       # Task 2: cost-sensitive baseline model
│   ├── model_tuning.py         # Task 3: tuning and model selection
│   ├── explain.py              # Task 3: SHAP explanations
│   ├── features/transformers.py# guard, feature engineer, leakage flagger
│   └── data/generate_sample_data.py  # how the sample data was generated
├── tests/
│   ├── test_pipeline.py        # Task 1 tests
│   ├── test_baseline.py        # Task 2 tests
│   └── test_model_tuning.py    # Task 3 tests
└── requirements.txt
Dependencies

Python 3.12, pandas, numpy, scikit-learn, imbalanced-learn, pyyaml, pyarrow, joblib, pytest, xgboost, shap, matplotlib.

pip install -r requirements.txt
Quick start

The raw data is not committed. Generate it, then run Task 1, Task 2, Task 3 and the tests:

python -m src.data.generate_sample_data
python -m src.data_pipeline
python -m src.baseline_model
python -m src.model_tuning
python -m src.explain
python -m pytest -v
Task 1: Data integrity and leakage prevention

Loads raw transactions, cleans them, engineers features, checks for target leakage, and writes reproducible train/val/test parquet files.

Configuration

Everything is set in config/config.yaml; no paths are hard-coded.

Key	Meaning
paths.raw_data	input CSV
paths.processed_dir	where parquet files are written
paths.splits_dir	where split indices are saved
paths.pipeline_artifact	fitted pipeline (joblib)
target_column	label column (is_fraud)
random_state	seed for reproducible splits
split.*	train / val / test fractions (0.70 / 0.15 / 0.15)
leakage.correlation_threshold	flag features with abs(corr) above this (0.8)
Pipeline steps
Load the raw CSV (path from config), parsing timestamps.
Clean: drop duplicate transaction_ids, rows with missing values or amount <= 0, and rows where account_created_at > order_time.
Remove PII: customer_name, email, ip_address.
Velocity feature: number of the customer's orders in the previous 24 h (looks only at the past).
Stratified split 70/15/15; indices saved to data/splits/.
Fit the sklearn Pipeline on the training split only:
FutureTimestampGuard: raises if any timestamp is after the reference time
FraudFeatureEngineer: account age, order hour, high-amount flag (threshold learned from train only)
ColumnTransformer: scale numeric, one-hot encode categorical
LeakageFlagger: warns for features with abs(corr) > 0.8 to the target
Transform train / val / test and save as parquet in data/processed/.

Outputs: data/processed/{train,val,test}.parquet, data/processed/fraud_pipeline.joblib, data/splits/{train,val,test}_idx.csv.

Leakage checks
Scaler, encoder, amount threshold and leakage flagger are fitted on train only.
Future timestamps are rejected by FutureTimestampGuard.
Accounts created after the order are dropped during cleaning.
LeakageFlagger flags any feature with abs(corr) > 0.8 to the target.
Tests verify all of the above; see docs/decision_log.md for limitations.
Task 2: Baseline model (cost-sensitive)

A Logistic Regression baseline trained on the Task 1 splits, with the heavy class imbalance handled explicitly and evaluated with a business cost matrix.

How to run
python -m src.baseline_model                    # class_weight (default)
python -m src.baseline_model --strategy smote   # SMOTE variant
python -m src.baseline_model --data synthetic   # make_classification test data

The script loads data/processed/{train,val,test}.parquet (already scaled and encoded by Task 1, so no extra scaling is applied), trains the model, and writes models/baseline_model.joblib and reports/baseline_metrics.md.

Design decisions
Imbalance handling: class_weight="balanced". It reweights the loss, so no synthetic rows are created, every training row is a real order, and there is nothing to re-fit inside cross-validation folds. This keeps the pipeline simple and auditable. SMOTE is available (--strategy smote, training data only) for comparison in Task 3. The justification is also in the code comments.
Business cost: a false negative (missed fraud) costs $100 and a false positive (blocked or reviewed genuine order) costs $10. Total cost is the headline metric; precision, recall and F1 are reported for reference.
Threshold: chosen to minimise total cost on cross-validated (out-of-fold) predictions over train+val. The test set is never used for tuning and is used once for the final report.
Random baseline: a classifier that flags orders at the training fraud rate, averaged over 500 random draws.
Results (test set: 300 orders, 11 fraud)
Model	Precision	Recall	Total cost
Random classifier (expected)	0.034	0.035	$1,172
Logistic Regression, threshold 0.50	0.028	0.273	$1,830
Logistic Regression, cost-tuned threshold (0.79)	0.083	0.091	$1,110

Test ROC-AUC is about 0.55. Full details are in reports/baseline_metrics.md.

Finding: the baseline is only slightly cheaper than a random classifier and about equal to flagging nothing ($1,100), so the current features carry weak signal. The test set has only 11 fraud cases, so these differences are noisy. This is the bar the Task 3 models need to beat.

Tests
python -m pytest -q

tests/test_baseline.py checks that SMOTE gives the expected training class distribution (balanced, and a partial ratio), that validation and test data are never resampled, the class-weight ratio, the cost matrix arithmetic, the loader for the Task 1 splits, and the out-of-fold probabilities.

Task 3: Advanced model selection, tuning and explainability

Two stronger algorithms (XGBoost, RandomForest) are tuned with a randomized search and compared with the Task 2 logistic-regression baseline on the same business cost (FN = $100, FP = $10, imported from src/baseline_model.py).

How to run
pip install -r requirements.txt              # now includes xgboost, shap, matplotlib
python -m src.model_tuning                   # 30 trials per algorithm, 5-fold CV (~3 min)
python -m src.explain                        # SHAP plots for the selected model
python -m pytest -q

--data synthetic runs everything on make_classification data and writes into models/synthetic/ and reports/synthetic/, so real results are never overwritten.

Output	Path
Search script (logs every trial)	src/model_tuning.py, logs/tuning.log, reports/tuning_trials.csv
Best parameters per algorithm	reports/best_params.json
Selected model + performance summary	models/best_model.pkl, models/best_model_summary.json
Cost comparison table	reports/comparison_table.md
SHAP plots	reports/shap_examples.png, reports/shap_global.png
Method
Data: train + val are pooled for model selection (1,700 orders, 65 fraud). The Task 1 split indices are untouched. The test set (300 orders, 11 fraud) is never used to tune or to choose; it is scored once for confirmation.
Search: ParameterSampler, 30 trials per algorithm (60 total), fixed seed. scale_pos_weight (XGBoost) and class_weight (RandomForest) are part of the search, so the cost-sensitive imbalance handling from Task 2 is tuned, not assumed.
Stratified 5-fold CV keeps the fraud ratio in every fold.
Trial score = business cost, not AUC: out-of-fold probabilities are pooled, the cost-minimising threshold is found on them (same function as Task 2), and the cost at that threshold is the trial's score.
Early stopping runs inside each fold on a slice of the training part only, so the CV fold stays untouched. XGBoost uses native early stopping. RandomForest has none, so it is grown in chunks of 50 trees and stopped when validation log-loss stops improving. The final refit uses the median stopped tree count + 10%.
Selection rule: lowest cross-validated cost. If models are within one standard error of each other they are treated as tied and the faster one wins.
Reference rows: the Task 2 recipe re-scored under the identical protocol (refit on train+val, so its test cost is $1,080 here vs $1,110 in Task 2, which trained on train only), and "flag nothing".
Results
Model	CV cost /1000 (± SE)	Test cost	FN	FP	Test ROC-AUC	Latency p50	Size
XGBoost (tuned), selected	3,488 ± 87	$1,040	10	4	0.707	2.1 ms	204 KB
RandomForest (tuned)	3,471 ± 141	$1,120	11	2	0.741	13.8 ms	825 KB
Logistic Regression (Task 2 recipe)	3,629 ± 83	$1,080	10	8	0.588	0.8 ms	2 KB
Flag nothing	3,824	$1,100	11	0	-	-	-

("Flag nothing" CV value = 65 fraud x $100 / 1,700 orders x 1,000.) Full table with precision, recall, thresholds and fit times: reports/comparison_table.md.

What the numbers say

In cross-validation (65 fraud cases) both tuned models are about 9% cheaper than flagging nothing (XGBoost 8.8%, RandomForest 9.2%) and about 4% cheaper than the Task 2 baseline (which is 5.1% cheaper than flagging nothing). XGBoost and RandomForest are statistically tied (a gap of 18 vs standard errors of 87 and 141).
On the test set (11 fraud) the differences are $40 to $80, less than one missed fraud, so they are noise. The test set confirms only that nothing is badly wrong: the selected model is not worse than flagging nothing.
The tuned models rank orders better than the baseline (test ROC-AUC 0.71 and 0.74 vs 0.59), but the signal is still weak overall, and PR-AUC is low for every model. Tuning helped a little; it did not change the conclusion from Task 2 that the current features carry limited information about fraud.
Model selection: trade-offs (speed vs accuracy vs cost)
Cost (the decision metric). XGBoost and RandomForest tie on CV cost, so cost alone does not separate them. Both beat the baseline in CV by a modest margin.
Accuracy. Recall is very low for all models at the cost-optimal threshold (the selected model catches 1 of 11 test frauds). That is by design: with weak signal, flagging many orders costs more in false alarms ($10 each) than it saves. Precision and recall are reference metrics; cost decides.
Speed and size. Single-order scoring: logistic regression 0.8 ms, XGBoost 2.1 ms, RandomForest 13.8 ms (p95 19.6 ms). Model size: 2 KB, 204 KB, 825 KB. At dozens of orders a day all three are far inside any real-time limit, so speed is a tie-breaker, not a constraint.
Why XGBoost was selected. Tied on cost with RandomForest, but about 6x faster to score and 4x smaller. It also had the lower test cost, though that is noise.
Maintainability. XGBoost adds one dependency. The saved bundle (best_model.pkl) carries the model, threshold, feature list and library versions for the Task 4 API.
Sensitivity to the cost assumption (important). The ranking depends on the true FN:FP ratio (test cost per 1,000 orders, thresholds re-chosen on out-of-fold data):
FN:FP	Baseline	XGBoost	RandomForest	Flag nothing
5:1	1,833	1,800	1,900	1,833
10:1 (assumed)	3,600	3,467	3,733	3,667
20:1	7,133	5,833	5,567	7,333
50:1	9,467	8,300	7,033	18,333

At 10:1 the models barely beat doing nothing. If a missed fraud really costs 20x or more a false alarm, the models save much more and RandomForest becomes the cheaper choice. The $100 / $10 costs are assumptions and must be confirmed with the client before the model is relied on.

Explainability

reports/shap_examples.png shows, for four test orders (highest-scored fraud, lowest-scored fraud, highest-scored legitimate order, lowest-scored legitimate order), which features pushed the score up (red) or down (blue) and by how much. reports/shap_global.png shows average importance over the whole test set. Feature values are the scaled/encoded values from Task 1, not raw dollars or hours. XGBoost SHAP values are in log-odds.

Limitations
Small data: 65 fraud cases for selection and 11 for testing. Differences under roughly $100 per 300 orders cannot be trusted.
Winner's curse: picking the best of 30 noisy trials makes the CV cost slightly optimistic. The test result is the unbiased (but noisy) check.
Weak signal: most trials score close to "flag nothing". More informative features (or more data) are likely to help more than further tuning.
Synthetic data: if the Task 1 data is generated sample data, results describe that data, not Brew & Bean's real fraud.