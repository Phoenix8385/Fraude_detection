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

## API

## Run locally / Docker

## Testing & CI

## Limitations

## Future work
