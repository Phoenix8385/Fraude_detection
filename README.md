# Fraud & Anomaly Detection System

End-to-end credit-card fraud detection pipeline (work in progress).

> Sections are filled in as each phase produces real results;
> every number here will come from `reports/metrics/`.

## Problem

## Why it's hard

## Dataset

## Architecture

## EDA findings

## Validation design

## Models compared

## Results

## Threshold & cost

Business cost of a threshold:

```
Expected cost = Σ Amount(missed fraud)      ← money lost on false negatives
              + review_cost × (TP + FP)     ← analyst time for every alert
```

**Assumption:** `review_cost` = €5 per alert. This is not a measured figure; a sensitivity
table at €2 / €5 / €20 will be added once the threshold is selected.

## Calibration

> **Small-data warning.** Calibration and both decision thresholds are estimated on halves
> of the validation part that contain very few frauds: on the time split (headline),
> `valid_cal` has 24 frauds and `valid_thr` 33; on the stratified split, 47 and 47
> (`reports/metrics/calibration_{split}.json`). With so few positives, one fraud more or
> less moves recall by about 3 percentage points on the time split, and the bootstrap
> interquartile range of `t_review` on the time split is 0.02–0.34
> (`reports/metrics/policy_time.json`). Treat the calibration choice and the thresholds as
> rough estimates, not precise operating points.

## Explainability

The production model (`logreg-iso-v1.0.0`, time split) is a scaled Logistic Regression, so each
feature's **model contribution** is its coefficient × scaled value, in log-odds of the
**uncalibrated** model; with the intercept as base they sum exactly to the model's log-odds.
No `shap` library is needed at serving time (`shap` is used only for the plots in
`reports/figures/shap_*_time.png`). Contributions describe the model's arithmetic, not causes
of fraud, and `V1`–`V28` are anonymised PCA components without business meaning. Largest mean
|contribution| on 2,000 `valid_thr` rows: V14, V4, V10, V11, V3
(`reports/metrics/explain_time.json`). Details and limitations: [MODEL_CARD.md](MODEL_CARD.md).

## API

## Run locally / Docker

## Testing & CI

## Limitations

## Future work
