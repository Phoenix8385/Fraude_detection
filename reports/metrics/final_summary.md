# Final TEST evaluation (Phase 9, run once)

Frozen Phase 8 models and thresholds applied to the TEST split (pre-registered in
docs/memory.md and reports/metrics/preregistration.json). Primary metric: PR-AUC.
Time split = headline. 95% stratified bootstrap CIs (1,000 resamples, seed 42).

| split | model | calibration | t_review | PR-AUC [CI] | recall [CI] | precision [CI] | alerts/1k [CI] | cost € [CI] | no-model € | flag-all € |
|---|---|---|---|---|---|---|---|---|---|---|
| time (headline) | logreg | isotonic | 0.02 | 0.7594 [0.6625, 0.8447] | 0.824 [0.730, 0.905] | 0.459 [0.397, 0.525] | 2.34 [2.03, 2.68] | 3,042 [1,013, 6,186] | 7,728 | 283,725 |
| stratified | xgb_u | none | 0.06 | 0.8259 [0.7501, 0.8969] | 0.811 [0.737, 0.884] | 0.762 [0.691, 0.843] | 1.78 [1.59, 2.01] | 4,288 [1,647, 7,659] | 14,766 | 283,730 |

## t_block (HIGH / HOLD)

| split | t_block | precision [CI] | recall [CI] | TP | FP |
|---|---|---|---|---|---|
| time | 0.23 | 0.943 [0.877, 1.000] | 0.676 [0.568, 0.770] | 50 | 3 |
| stratified | 0.22 | 0.916 [0.856, 0.964] | 0.800 [0.726, 0.874] | 76 | 7 |

## Context-only baselines (Phase 5, frozen; cannot change any decision)

| split | baseline | PR-AUC [CI] | ROC-AUC |
|---|---|---|---|
| time | dummy | 0.0013 [0.0013, 0.0013] | 0.5000 |
| time | logreg | 0.7578 [0.6487, 0.8578] | 0.9841 |
| time | iforest | 0.0457 [0.0348, 0.0618] | 0.9494 |
| stratified | dummy | 0.0017 [0.0017, 0.0017] | 0.5000 |
| stratified | logreg | 0.6860 [0.5785, 0.7816] | 0.9678 |
| stratified | iforest | 0.0798 [0.0566, 0.1231] | 0.9398 |

## Stratified vs time — why the results differ

The time split is the pre-registered headline because it trains on earlier transactions and
tests on later ones, which is how the model would be used after deployment. The stratified
split shuffles rows across the whole period: it keeps the fraud rate equal in every part, but
it does not reproduce any change in the data over time. On test, PR-AUC is 0.7594
[0.6625, 0.8447] on the time split versus 0.8259 [0.7501, 0.8969] on the stratified split.
The two numbers are not a like-for-like comparison: they come from different test rows and
different frozen champions (tuned Logistic Regression with isotonic calibration on time,
tuned XGBoost without calibration on stratified), and both rest on few frauds — 74 on time,
95 on stratified — so both confidence intervals are wide and they overlap. The project's
existing evidence shows that the fraud rate varies across the time-ordered parts (train
0.201%, valid 0.100%, test 0.130%; splits_time.json), but nothing in these results isolates
the cause of the PR-AUC gap, and no causal reason is claimed here. The stratified result
should therefore not be read as stronger evidence of future real-world performance; the time
result is the better estimate. At the frozen review thresholds, test recall is below the 0.85
target on both splits: 0.824 (61 of 74) at t_review 0.02 on time, where the threshold was
already a documented validation fallback, and 0.811 (77 of 95) at t_review 0.06 on
stratified, which had met the target on validation (0.9149). These test results are reported
as observed and are not used to change any model, calibration or threshold.
