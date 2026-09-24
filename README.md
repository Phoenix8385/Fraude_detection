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

## Explainability

## API

## Run locally / Docker

## Testing & CI

## Limitations

## Future work
