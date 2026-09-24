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
expected_cost = sum(Amount of FN) + review_cost × (TP + FP)
review_cost default €5 (assumption); sensitivity at €2, €5, €20.

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
