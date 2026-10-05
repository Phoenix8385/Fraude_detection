# Model card — `logreg-iso-v1.0.0`

Every number below is copied from `reports/metrics/*` or `models/model_meta.json`; the source
file is named next to each block.

## Model
- **What it is:** a risk-scoring model for card transactions. It outputs a fraud probability
  and a recommended action; it is not a real-time fraud platform and makes no final decision.
- **Type:** tuned Logistic Regression (scaled features, `C = 0.000802`, `class_weight = None`)
  with isotonic calibration. 32 input features: `Time`, `V1`–`V28`, `Amount`, `log_amount`,
  `hour_of_day`. (`models/model_meta.json`, `reports/metrics/best_params_logreg_time.json`)
- **Artifact:** `models/model.joblib`, byte-identical to the frozen Phase 8 model
  `models/final_model_time.joblib` (MD5 `bf798b5c02c429051303c99c7ab75240`); MLflow run
  `5561385ba3144bf9ae85f73d7e56d43c`; calibrated in commit `ef3d4b6`.
- **Selection:** pre-registered champion rule on the time split. Tuned XGBoost and tuned LR were
  statistically indistinguishable on validation PR-AUC (paired 95% CI of the difference
  [−0.0208, +0.0239], `prob_a_better` 0.546); tie-break 1 could not decide (neither reached
  recall 0.85 on the grid); tie-break 2 chose the simpler model. (`champion_time.json`)

## Intended use
Rank transactions for analyst review in an offline / batch-or-request scoring setting, using
the decision policy below. Not for automated account closure, not for individual-level
decisions without human review, and not for any dataset other than the one it was built on.

## Data
Kaggle `mlg-ulb/creditcardfraud`: card transactions spanning about two days (`Time` runs to
172,792 s; `splits_time.json`). 283,726 rows after removing exact duplicates; 473 fraud
(0.167%). `V1`–`V28` are anonymised PCA components: no business meaning may be given to them.
(`data_summary.json`)
Split: time-ordered 60/20/20 (headline). Validation was halved chronologically into
`valid_cal` (calibration) and `valid_thr` (thresholds). The test part was scored once.

## Decision policy (frozen, `policy_time.json`)
| tier | action | rule |
|---|---|---|
| HIGH | HOLD | probability ≥ t_block = 0.23 |
| MEDIUM | REVIEW | probability ≥ t_review = 0.02 |
| LOW | APPROVE | otherwise |

`t_review` is a **documented fallback**: no threshold on the 0.01–0.95 grid reached the recall
target 0.85 on `valid_thr` (validation recall 0.8485, 28 of 33 frauds). Review cost assumption:
€5 per alert (an assumption, not a measured figure).

## Performance — one-shot TEST evaluation (`final_time.json`, `final_summary.md`)
Time-split test part: 56,745 rows, 74 frauds. 95% stratified bootstrap CIs, 1,000 resamples.

| metric | value [95% CI] |
|---|---|
| PR-AUC (primary) | 0.7594 [0.6625, 0.8447] |
| ROC-AUC | 0.9762 |
| Recall at t_review | 0.824 [0.730, 0.905] (61 of 74) |
| Precision at t_review | 0.459 [0.397, 0.525] |
| Alerts per 1,000 transactions | 2.34 [2.03, 2.68] |
| Expected cost at t_review | €3,042.48 [€1,013.04, €6,185.73] |
| Fraud amount caught | 69.2% |
| No-model cost (flag nothing) | €7,727.67 |
| Precision / recall at t_block | 0.943 [0.877, 1.000] / 0.676 [0.568, 0.770] |

Tiers on test: HOLD 53 (50 fraud), REVIEW 80 (11 fraud), APPROVE 56,612 (13 fraud).
For context only, the secondary stratified split (a different model, tuned XGBoost) scored
test PR-AUC 0.8259 [0.7501, 0.8969]; it is not evidence of better future performance
(`final_stratified.json`, `final_summary.md`).

## Calibration (`calibration_time.json`)
Isotonic was selected: Brier on `valid_thr` 0.000266 vs 0.000855 uncalibrated (68.9% lower).
It was fitted on `valid_cal`, which holds only 24 frauds, so the calibration map rests on few
positives. Isotonic ties also lowered `valid_thr` PR-AUC slightly (0.8375 → 0.8294).

## Explainability (`explain_time.json`, `reports/figures/shap_*_time.png`)
Per-transaction explanations are each feature's **model contribution** in log-odds of the
**uncalibrated** Logistic Regression: coefficient × scaled feature value, plus the intercept
(−7.1327) as base. They sum exactly to the model's log-odds. They describe the model's
arithmetic, **not causes of fraud**, and since `V1`–`V28` are anonymised PCA components they
carry no business meaning. Largest mean |contribution| on a 2,000-row `valid_thr` sample:
V14, V4, V10, V11, V3.

## Limitations
- **Small samples:** 33 frauds in `valid_thr`, 24 in `valid_cal`, 74 in the test part; all
  thresholds and metrics carry wide uncertainty (see CIs and `policy_time.json` stability:
  `t_review` bootstrap IQR 0.02–0.34).
- **Recall target missed:** test recall at `t_review` is 0.824, below the 0.85 target.
- **Short, anonymised data:** about two days of transactions; behaviour may differ in other
  periods. The fraud rate
  already varies across the time-ordered parts (train 0.201%, valid 0.100%, test 0.130%;
  `splits_time.json`).
- **Cost model:** €5 per review is assumed; a missed fraud costs exactly its `Amount`.
- **Library versions:** the pickled model expects sklearn 1.9.1 (full list in
  `model_meta.json`); loading warns on version drift.

## Monitoring
`models/reference_profile.json` stores quantile bin edges and reference proportions for
`Amount` (10 bins) and the served score (4 bins after merging tied isotonic values), computed
on `valid_thr`, for PSI drift checks (Phase 17). No raw feature values are stored.
