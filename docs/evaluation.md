# Evaluation Protocol (SACRED — do not change without logging in memory.md)

## Splits
- Exact duplicates dropped BEFORE splitting.
- A: stratified 60/20/20, seed 42. B: time-ordered 60/20/20 (PRIMARY headline).
- VALID further split 50/50 stratified → valid_cal (calibration) and valid_thr (threshold).

## What each part is used for
| Part | Used for |
|---|---|
| train | fitting, CV for tuning (StratifiedKFold / TimeSeriesSplit) |
| valid (full) | model comparison (Phases 5–7) |
| valid_cal | fitting calibrator |
| valid_thr | calibration check, threshold selection, SHAP sample |
| test | Phase 9 final evaluation ONLY, run once |

## Metrics (all computed by metrics.compute_metrics)
Primary: PR-AUC (average precision). Also: ROC-AUC, precision, recall, F1, specificity,
FPR, confusion matrix, alerts per 1,000, % fraud amount caught, expected cost.
Accuracy is never a headline metric.

## Cost
```
Expected cost = Σ Amount(missed fraud)      ← money lost on false negatives
              + review_cost × (TP + FP)     ← analyst time for every alert
```
expected_cost = sum(Amount of FN) + review_cost × (TP + FP)
review_cost default €5 (assumption); sensitivity at €2, €5, €20.
An alert is raised when fraud probability >= threshold.

## Threshold rule
Grid 0.01–0.95 step 0.01 on valid_thr. Choose min expected_cost subject to
recall >= 0.85. If none qualifies, choose max-recall threshold and document.

## Calibration rule
Isotonic on valid_cal. Keep only if Brier on valid_thr improves.

## Required tests
data: schema pass/fail, dedupe count · features: math, no mutation · splits: no overlap,
ratios, time order · metrics: hand-computed confusion matrix · threshold: hand-computed
choice, recall constraint · predict: prob in [0,1], column order, batch length ·
api: health, model-info, 200, missing 422, extra 422, NaN 422, batch>1000 422,
503 without model, 401 with wrong key.
# Evaluation addendum (append to docs/evaluation.md)

## Uncertainty
Every headline metric (PR-AUC, recall, precision, alerts/1k, cost) is reported with a 95% percentile
stratified bootstrap CI (1,000 resamples, seed from config). Model comparisons use a PAIRED bootstrap
(same resampled indices). Report prob_a_better.

## Champion rule (pre-registered)
Highest tuned validation PR-AUC. If the paired-bootstrap 95% CI of the difference vs tuned Logistic
Regression includes 0: tie-break 1 = lower expected cost at recall >= 0.85 on validation;
tie-break 2 = simpler model.

## Validation halves
Time split: chronological halves (earlier = valid_cal, later = valid_thr). Stratified split: stratified halves.
Warn if a half has < 30 positives.

## Calibration
Candidates none/sigmoid/isotonic; pick lowest Brier on valid_thr; within 5% of 'none' -> 'none'.

## Decision policy
t_review = min expected cost s.t. recall >= 0.85 on valid_thr.
t_block = smallest threshold with precision >= 0.90 and >= 5 true positives above it; else None.
Tiers: p >= t_block HIGH/HOLD; p >= t_review MEDIUM/REVIEW; else LOW/APPROVE.
Report threshold stability (200 bootstrap resamples, median + IQR). Do not use it to alter the choice.

## Final test (Phase 9)
Pre-register choices in memory.md first. Report no-model cost vs model cost. Frozen baselines shown for
context only. evaluate.py refuses to re-run without --force.

## Drift
PSI against models/reference_profile.json: <0.1 stable, 0.1-0.2 watch, >0.2 investigate;
"insufficient_data" below 200 rows. Simulated drift is labelled simulated.

## Additional tests
openapi.json snapshot · log-privacy · DB-failure tolerance · no-test-leakage · rate limit 429 · coverage >= 85%.
