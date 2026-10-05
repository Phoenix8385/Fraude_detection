# Project Memory — decisions & current state

## Fixed decisions
- Headline split: time-based. Primary metric: PR-AUC.
- Test set opened once (Phase 9). Threshold chosen on valid_thr by expected cost, recall >= 0.85.
- Isolation Forest included to justify "Anomaly" in the title.
- SHAP in serving via XGBoost pred_contribs (no shap lib in Docker).
- Previously reported numbers (P 0.618 / R 0.857 / F1 0.718 / ROC-AUC 0.977) are NOT used
  until reproduced under this protocol.

## Data facts (fill in Phase 2/4)
- Rows after dedupe: 283,726 (raw 284,807) — source: reports/metrics/data_summary.json
- Duplicates dropped (fraud among them): 1,081 (19). Fraud after dedupe: 473 (rate 0.001667)
- Fraud count per part (train/valid/test) — stratified: 284/94/95   time: 342/57/74
  (source: reports/metrics/splits_*.json)

## Results log (fill as you go)
- Phase 5 baselines (valid PR-AUC, strat / time): dummy 0.0017 / 0.0010, logreg 0.7882 / 0.7806,
  iforest 0.1291 / 0.0231 — source: reports/metrics/experiments.csv
- Phase 6 winner + reason: xgb_weighted. Time split (headline) valid PR-AUC 0.7880 — best of
  all models; +0.0082 vs plain xgb (0.7798), +0.0107 vs xgb_smote_10, +0.0174 vs xgb_smote,
  +0.0074 vs logreg. Stratified: 0.8888, 0.0011 behind plain xgb (0.8899). Margins are small
  vs 57 time-valid frauds, so treated as a near-tie; weighted chosen because it leads on the
  headline split and is the simplest imbalance fix (one parameter, no synthetic rows).
  SMOTE was slowest (~7.5–7.8 s vs ~4 s fit) and lowest on time. Both xgb_weighted and xgb
  go to Phase 7 tuning. Source: reports/metrics/model_comparison.md
- Phase 7 tuned valid PR-AUC (untuned → tuned): strat xgb_weighted 0.8888→0.8873 (−0.0015),
  xgb 0.8899→0.8852 (−0.0047); time xgb_weighted 0.7880→0.7643 (−0.0238), xgb 0.7798→0.7807
  (+0.0009). Tuning did not improve validation PR-AUC. Source: reports/metrics/tuning_summary.md
- Phase 8 calibration kept? Brier before/after (valid_thr): time (logreg) ISOTONIC kept,
  0.000855 -> 0.000266; stratified (xgb_u) NONE (isotonic 0.000294 only 3.8% below 0.000306)
  — source: reports/metrics/calibration_{split}.json
- Phase 8 frozen thresholds (strat / time): t_review 0.06 / 0.02 (time: recall target NOT met),
  t_block 0.22 / 0.23 — source: reports/metrics/policy_{split}.json
- PRE-REGISTRATION before Phase 9 (model, params file, calibrated, threshold, date): written
  2026-10-05, before any test read — see "Phase 9 PRE-REGISTRATION" section below and
  reports/metrics/preregistration.json. Phase 8 commit ef3d4b6.
- Phase 9 final test metrics (one-shot TEST run 2026-10-05, committed in 3a3297b; source:
  reports/metrics/final_{split}.json, final_summary.md; 95% stratified bootstrap CIs,
  1,000 resamples, seed 42). Frozen Phase 8 policy applied as pre-registered; the test
  results were NOT used to change the model, calibration or thresholds.
  TIME (headline) — tuned logreg, isotonic, t_review 0.02, t_block 0.23; 56,745 rows / 74 fraud:
    PR-AUC 0.7594 [0.6625, 0.8447] (ROC-AUC 0.9762). At t_review: recall 0.824 [0.730, 0.905]
    (61/74), precision 0.459 [0.397, 0.525], F1 0.589, 2.34 alerts/1k [2.03, 2.68], expected
    cost €3,042.48 [€1,013.04, €6,185.73], fraud amount caught 69.2%. No-model cost €7,727.67;
    flag-everything €283,725.00. Savings vs no-model €4,685.19 (60.6%) — derived as
    no-model − model from the two stored costs, not itself stored in reports/metrics.
    At t_block: precision 0.943 [0.877, 1.000], recall 0.676 [0.568, 0.770] (50 TP, 3 FP).
    Tiers: HIGH/HOLD 53 (50 fraud, 67.6% of fraud); MEDIUM/REVIEW 80 (11, 14.9%);
    LOW/APPROVE 56,612 (13, 17.6%).
  STRATIFIED (secondary) — tuned xgb_u, none, t_review 0.06, t_block 0.22; 56,746 rows / 95 fraud:
    PR-AUC 0.8259 [0.7501, 0.8969] (ROC-AUC 0.9738). At t_review: recall 0.811 [0.737, 0.884]
    (77/95), precision 0.762 [0.691, 0.843], F1 0.786, 1.78 alerts/1k [1.59, 2.01], expected
    cost €4,287.79 [€1,647.31, €7,658.90], fraud amount caught 74.4%. No-model cost €14,766.31;
    flag-everything €283,730.00. Savings vs no-model €10,478.52 (71.0%) — derived as above.
    At t_block: precision 0.916 [0.856, 0.964], recall 0.800 [0.726, 0.874] (76 TP, 7 FP).
    Tiers: HIGH/HOLD 83 (76 fraud, 80.0%); MEDIUM/REVIEW 18 (1, 1.1%); LOW/APPROVE 56,645 (18, 18.9%).
  Recall at t_review is below 0.85 on BOTH test splits (time was a pre-declared validation
  fallback; stratified met 0.9149 on validation). Context-only Phase 5 baselines, test PR-AUC
  (time / stratified): dummy 0.0013 / 0.0017, untuned logreg 0.7578 / 0.6860, iforest
  0.0457 / 0.0798. Stratified-vs-time paragraph: final_summary.md (no causal claim made).
- Phase 11 p95 latency:
- Live URL:

## Phase 1 decisions (2026-09-24)
- Repo lives at fraude/fraud-detection-guide/fraud-detection-system (guide files stay outside the repo).
- Python: system 3.11.1 install is broken (_ssl DLL mismatch: _ssl.pyd built for OpenSSL 3,
  only libssl-1_1.dll present) → pip had no HTTPS. .venv rebuilt from uv-managed CPython 3.11.15.
  Recreate with: "$APPDATA/uv/python/cpython-3.11.15-windows-x86_64-none/python.exe" -m venv .venv
- Package installed editable (pip install -e .) so `from fraud_detection...` works outside pytest.
- Versions resolved (see requirements-lock.txt): scikit-learn 1.9.1, imbalanced-learn 0.14.2,
  xgboost 3.2.0, shap 0.51.0, pandas 3.0.6, pandera 0.33.1, mlflow 3.16.1, fastapi 0.141.1,
  pydantic 2.13.5. All import together.
- config.py validates on load: split sums to 1.0, target_recall in (0,1], review cost >= 0.
- Watch: pandas 3.x is a major release (copy-on-write default, string dtype changes) —
  check pandera/imblearn behaviour in Phase 2.

## Phase 2 decisions (2026-09-24)
- pandera imported as `pandera.pandas` (0.33 API). Validation is lazy → raises SchemaErrors with all failures.
- CSV stores Time as whole seconds (int64); load_raw casts all 30 features to float64 so the
  schema's "all features float" holds. Class stays int64.
- Duplicates = identical in ALL 31 columns incl. Class (same features + different label is kept).
- Summary has two extra keys beyond the spec: rows_raw, duplicate_fraud_dropped.
- Real CSV passed schema: no nulls, Amount >= 0, Time >= 0, Class in {0,1}.

## Phase 3 decisions (2026-09-24)
- features.py: log_amount = log1p(Amount), hour_of_day = (Time // 3600) % 24. MODEL_FEATURES = 32 cols.
- CAVEAT: Time = seconds since first transaction, not clock time. hour_of_day is a 24h cycle
  from dataset start; it matches real clock hours only if the data starts at midnight (undocumented).
- EDA "top 10 by |mean difference|" uses z-scored features (raw diffs are unit-dependent);
  raw diffs printed alongside in the notebook.
- EDA Amount histogram uses per-class share weights, not density=True (log bins have unequal widths).
- Findings cell in notebooks/01_eda.ipynb is left for the human to write.
- Notebook generated/executed with: jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

## Phase 4 decisions (2026-09-24)
- Splits keep the clean-dataset row index (written into parquet) as a traceable row id.
- stratified: two train_test_split calls (test 0.2, then valid 0.25 of rest), seed from config.
- time: stable sort on Time, cut at round(0.6n) and round(0.8n). Boundary Times can tie
  (valid max = test min = 145,234 s): same-second rows may sit on both sides of a cut.
- Time split fraud rate is NOT constant: train 0.201%, valid 0.100%, test 0.130%.
- valid_cal / valid_thr (50/50 of valid, evaluation.md) not built yet — belongs to Phase 8.

## Phase 5 decisions (2026-09-24)
- MLflow backend: SQLite (mlflow.db) + artifacts in mlartifacts/ — NOT file:./mlruns.
  Reason: MLflow 3.16 blocks the file store by default (maintenance mode). Human chose SQLite.
  UI: mlflow ui --backend-store-uri sqlite:///mlflow.db
- MLflow saves sklearn models with skops; trusted types allow-listed explicitly in train.py
  (imblearn Pipeline, IsolationForestScorer, sklearn.tree._tree.Tree). Reload verified.
- All models are imblearn Pipelines (ready for SMOTE in Phase 6).
- iforest: fit without labels; score = -score_samples, min-max scaled on TRAIN, clipped to [0,1].
  It is a ranking score, not a probability; threshold 0.5 on it is arbitrary.
- Phase 5 compares at fixed threshold 0.5; threshold-dependent metrics (precision/recall/cost)
  are NOT a fair comparison between models — rank by PR-AUC only.
- Metric edge cases: precision/recall/f1 = 0 on zero division; PR/ROC-AUC = NaN if one class.

## Phase 6 decisions (2026-09-24)
- 4 XGBoost variants share XGB_BASE_PARAMS (n_estimators 400, max_depth 5, lr 0.05,
  subsample 0.8, colsample_bytree 0.8, hist, aucpr); only imbalance handling differs.
- xgb_weighted: scale_pos_weight = n_legit/n_fraud of TRAIN (strat ≈ 598, time ≈ 497).
- SMOTE inside imblearn Pipeline → fit-time only; test proves model sees > len(train) rows
  and predict returns len(valid). SMOTE runs on UNSCALED features, so its nearest-neighbour
  distances are dominated by Time and Amount (large units). Not changed; noted as a caveat.
- MLflow skops trusted types extended (XGBClassifier, Booster, SMOTE, KDTree, EuclideanDistance64).
- compare.py reloads each logged model from MLflow and asserts PR-AUC matches experiments.csv.
- Valid PR-AUC (strat / time): xgb 0.8899/0.7798, xgb_weighted 0.8888/0.7880,
  xgb_smote 0.8876/0.7706, xgb_smote_10 0.8811/0.7773 — source: reports/metrics/model_comparison.md

## Phase 7 decisions (2026-09-24)
- RandomizedSearchCV(n_iter=20, scoring=average_precision, random_state=42) on TRAIN only;
  StratifiedKFold(5, shuffle) for stratified, TimeSeriesSplit(5) on Time-sorted rows for time.
  All params prefixed "model__" (every model is a Pipeline with step "model").
- Best params per model/split in reports/metrics/best_params_{model}_{split}.json;
  tuned runs logged to MLflow with tags phase=7, stage=tuned; CSV model name "<model>_tuned".
- CAVEAT: xgb_weighted scale_pos_weight computed once from full TRAIN, reused in every CV fold
  (fold hold-out labels contribute to one count ratio). Mild; does not touch valid/test.
- Finding: CV fold std (0.046–0.073) is far larger than any tuned-vs-untuned change.
  Time split: TimeSeriesSplit favoured small models (200 trees, depth 3) that did worse on the
  later validation period (drift; see Phase 4 fraud-rate shift).
- train.py refactor: scoring/logging moved into score_and_log(); re-run of untuned
  xgb_weighted/time reproduced Phase 6 exactly (PR-AUC 0.7880).

## Open issues
- Time-split valid has only 57 fraud; halving for valid_cal/valid_thr leaves ~28 fraud for
  threshold selection → recall >= 0.85 estimate will be noisy (1 fraud ≈ 3.5 pp recall).
- RESOLVED 2026-09-24: Fraude_detection gitlink removed (commit 68f2930); README conflict
  markers removed and €5 review-cost assumption stated (uncommitted at time of writing).
- Phase 6: on TIME valid, best XGB (0.7880) beats logreg (0.7806) by only 0.0074 PR-AUC;
  plain xgb (0.7798) is below logreg. PRD criterion "final model beats LR on valid PR-AUC"
  is currently marginal on the headline split. Differences between XGB variants
  (≤ 0.017) are small relative to 57 validation frauds (1 fraud ≈ 1.75 pp recall).
- Phase 6: all supervised PR curves (time split) drop sharply at recall ≈ 0.75–0.80;
  the recall >= 0.85 target will likely cost low precision on the time split (Phase 8).
- Phase 7: tuned xgb_weighted on time valid (0.7643) is BELOW logreg (0.7806). The best
  time-split model remains UNTUNED xgb_weighted (0.7880). Final-model choice for Phase 8 pending.

## Phase 6 results (time-split validation PR-AUC)
weighted XGB 0.7880 | LR 0.7806 | plain XGB 0.7798 | SMOTE-10 0.7773 | SMOTE 0.7706

## Locked decisions
Champion (provisional): weighted XGB
Rule: highest tuned valid PR-AUC; if paired-bootstrap CI vs tuned LR includes 0, tie-break on expected cost at recall >= 0.85, then simplicity
t_review: min expected cost with recall >= 0.85
t_block: precision >= 0.90
Test set opened once in Phase 9 only

## Phase 6 results (time-split validation PR-AUC)
weighted XGB 0.7880 | LR 0.7806 | plain XGB 0.7798 | SMOTE-10 0.7773 | SMOTE 0.7706

## Locked decisions
Champion (provisional): weighted XGB
Rule: highest tuned valid PR-AUC; if paired-bootstrap CI vs tuned LR includes 0, tie-break on expected cost at recall >= 0.85, then simplicity
t_review: min expected cost with recall >= 0.85
t_block: precision >= 0.90
Test set opened once in Phase 9 only

## Phase 7 (v2) decisions & results (2026-10-05) — Optuna + bootstrap + champion
Source for every number below: reports/metrics/champion_{stratified,time}.json,
champion_summary.md, best_params_{xgb_u,logreg}_{split}.json (all VALIDATION; test not used).
- Tuner: Optuna TPESampler(seed=42), 50 trials/model/split, 5-fold CV on TRAIN only
  (StratifiedKFold shuffled | TimeSeriesSplit on time-sorted rows), objective = mean CV PR-AUC.
  RandomizedSearchCV (earlier Phase 7 attempt, tuning_summary.md / best_params_xgb*_*.json) is
  superseded; those files are kept only as history.
- Models tuned: "xgb_u" = XGBoost with scale_pos_weight = (n_legit/n_fraud)^u, u in [0,1]
  (u=0 plain, u=1 Phase 6 xgb_weighted); "logreg" = scaled LR, C (log 1e-4..1e2) + class_weight
  {balanced, None}. Phase 5/6 configs enqueued as first trials. Human chose this u definition.
- FIXES the Phase 7 caveat above: xgb_u's weight is recomputed from each CV fold's own
  training labels (fold hold-out labels no longer influence it).
- Bootstrap: 1,000 stratified resamples, seed 42, 95% percentile CI; paired on shared indices.
- 5-seed check: tuned configs refit with seeds 42-46 (config stability_seeds). LR (lbfgs) is
  deterministic, so its seed std is 0 by construction.
- Champion rule interpretation (written in champion.choose_champion before results were seen):
  tie-break 1 needs "cost at recall >= 0.85"; a model that cannot reach 0.85 has no such cost.
  If neither reaches it, tie-break 1 cannot decide -> tie-break 2.
- Operating points (min cost s.t. recall >= 0.85) are chosen AND reported on full validation
  (optimistic); used only for tie-break 1. Phase 8 re-selects on valid_thr.

### Results — TIME split (headline), valid fraud = 57
- xgb_u tuned: best u = 0.1226 (scale_pos_weight 2.14, vs 497 at u=1). Params: n_estimators 650,
  max_depth 6, learning_rate 0.0213, subsample 0.803, colsample_bytree 0.955,
  min_child_weight 1.34, reg_lambda 3.17. CV PR-AUC 0.8031 ± 0.0463.
  Valid PR-AUC 0.7773 [95% CI 0.6703, 0.8687]; 5-seed 0.7797 ± 0.0020.
- logreg tuned: C = 0.000802, class_weight = None. CV PR-AUC 0.7695 ± 0.0763.
  Valid PR-AUC 0.7759 [0.6693, 0.8699]; 5-seed 0.7759 ± 0.0000.
- Paired PR-AUC (xgb_u - logreg): +0.0013 [-0.0208, +0.0239], prob_a_better = 0.546.
  **NOT statistically distinguishable: the 95% CI includes 0.**
- Tie-break 1: neither model reaches recall 0.85 at any grid threshold (both best at t=0.01:
  xgb_u recall 0.807, cost €5,135; logreg recall 0.772, cost €4,799) -> cannot decide.
- Tie-break 2 -> **CHAMPION (time, headline) = tuned Logistic Regression.**
  Robustness: even if tie-break 1 compared the below-target costs, logreg (€4,799) < xgb_u
  (€5,135), so the champion would be the same.
- vs Phase 6 untuned valid PR-AUC: logreg 0.7806 -> 0.7759 (-0.0046); xgb_u 0.7773 vs untuned
  xgb_weighted 0.7880 (-0.0108) and xgb 0.7798 (-0.0025). Tuning did NOT improve validation
  PR-AUC on the time split; CV gains (0.7873 Phase 6 config -> 0.8031) did not carry over to
  the later validation period. Untuned models are not candidates under the rule.

### Results — STRATIFIED split, valid fraud = 94
- xgb_u tuned: best u = 0.9273 (scale_pos_weight 375.9). Params: n_estimators 700, max_depth 7,
  learning_rate 0.0407, subsample 0.578, colsample_bytree 0.616, min_child_weight 1.37,
  reg_lambda 0.873. CV 0.8520 ± 0.0695. Valid PR-AUC 0.8848 [0.8232, 0.9461]; 5-seed 0.8830 ± 0.0016.
- logreg tuned: C = 0.000697, class_weight = None. CV 0.7430 ± 0.0610.
  Valid PR-AUC 0.7832 [0.6960, 0.8685]; 5-seed 0.7832 ± 0.0000.
- Paired (xgb_u - logreg): +0.1016 [+0.0433, +0.1572], prob_a_better = 1.000 -> CI excludes 0.
  **CHAMPION (stratified) = xgb_u, by the main rule** (no tie-break needed).
- vs Phase 6 untuned: logreg 0.7882 -> 0.7832 (-0.0050); xgb_u 0.8848 vs xgb 0.8899 (-0.0050),
  xgb_weighted 0.8888 (-0.0040). Again no validation gain from tuning.

### Takeaways
- The "Locked decisions" entry "Champion (provisional): weighted XGB" is SUPERSEDED by the
  pre-registered rule: headline (time) champion = tuned LR; stratified champion = xgb_u.
- Split disagreement: on the stratified (random) split XGB is clearly better; on the
  time-ordered split its advantage disappears. With 57 validation frauds, the CI width (~0.20
  PR-AUC) is ~100x the seed-to-seed std (0.002) — data scarcity, not seed noise, dominates.
- Optuna picked small u on time (0.12) and near-full weighting on stratified (0.93).
- Both tuned LRs prefer class_weight=None with strong regularisation (C ~ 7e-4 to 8e-4).
- Recall 0.85 is not reachable on full time-validation within the 0.01-0.95 grid for either
  model -> carried to Phase 8 (grid floor decision pending, see Open issues).
- Process: champion CLI re-run reproduced both champion_*.json byte-for-byte.
- ruff: passes with calibrate.py excluded; calibrate.py's 3 pre-existing errors left for Phase 8
  (human decision). Dependency change: optuna 5.0.0 (+ colorlog 6.12.0) added; lock regenerated
  as UTF-8 without the local editable line.
- Observation: data/processed/*_test.parquet last-access time is 2026-09-30 15:09 (before this
  session; cause unknown — not Phase 7 code). Phase 7 code never opens them
  (tests/test_no_test_leakage.py).

## Phase 8 decisions & results (2026-10-05) — calibration + two-threshold policy
Source for every number below: reports/metrics/calibration_{split}.json, policy_{split}.json,
tier_table_{split}.csv, threshold_table_{split}.csv, cost_sensitivity_{split}.csv
(valid_cal / valid_thr only; test never read). Models = FROZEN Phase 7 champions reloaded from
MLflow (time: logreg run 5561385b; stratified: xgb_u run 15a37803). No retuning, no re-selection.
- Halves: time = chronological (earlier valid_cal / later valid_thr); stratified = stratified
  50/50, seed 42; indices in valid_subsplit_indices_{split}.json. Time: valid_cal 28,372 rows /
  **24 fraud (WARNING < 30)**, valid_thr 28,373 / 33. Stratified: 47 / 47 fraud.
- API path: sklearn 1.9.1 -> CalibratedClassifierCV(FrozenEstimator(champion), method=...),
  ensemble="auto" -> ONE calibrator fitted on all valid_cal predictions. cv="prefit" branch
  exists only for sklearn < 1.6 (not used). Model weights never refitted (tested).
- Calibrators fitted on valid_cal ONLY; candidates scored on valid_thr ONLY with Brier,
  log-loss, ECE (10 equal-width bins), PR-AUC (metrics.calibration_metrics).
- Selection rule (KING Phase 8 spec = evaluation.md v2 addendum): lowest Brier; if the best
  calibrated candidate is within 5% of 'none' -> 'none'. Exactly 5% counts as "within"
  (float-safe). The older "isotonic, keep if Brier improves" line in evaluation.md is
  superseded by this rule (KING spec, 2026-10-05).
- t_review = min expected cost s.t. recall >= 0.85 on the 0.01-0.95 grid; if unreachable, the
  highest-recall threshold (cheapest among ties) with fallback=true. t_block = smallest
  threshold with precision >= 0.90 AND >= 5 TP, else None. Equality meets a threshold.
- Stability (200 stratified bootstrap resamples, seed 42) and cost sensitivity are REPORT ONLY.

### TIME split (headline) — tuned logreg — calibration: ISOTONIC
| candidate | Brier | log-loss | ECE | PR-AUC |
|---|---|---|---|---|
| none | 0.000855 | 0.004161 | 0.000824 | 0.8375 |
| sigmoid | 0.000290 | 0.002070 | 0.000264 | 0.8375 |
| isotonic | 0.000266 | 0.002005 | 0.000182 | 0.8294 |
- How applied: isotonic lowest Brier, 68.9% below 'none' (> 5%) -> isotonic. Trade-off:
  isotonic ties lower valid_thr PR-AUC 0.8375 -> 0.8294; the rule is Brier, so isotonic stands.
- **t_review = 0.02 — FALLBACK (recall >= 0.85 NOT achievable on the grid)**: recall 0.8485
  (28/33; 29 needed), precision 0.3457, 2.85 alerts/1k, expected cost €541.16.
  5 of 33 valid_thr frauds score below 0.01 (grid floor).
- **t_block = 0.23** — precision 0.9643 (27 TP, 1 FP), recall 0.8182. precision >= 0.90 achievable.
- Tiers (valid_thr): HIGH/HOLD 28 (27 fraud, fraud share 0.964); MEDIUM/REVIEW 53 (1 fraud,
  0.019); LOW/APPROVE 28,292 (5 fraud). The REVIEW band is almost all false positives.
- Stability: t_review median 0.02, IQR [0.02, 0.34], target met in 45.5% of resamples;
  t_block median 0.23, IQR [0.23, 0.23], exists in 100%.
- Cost sensitivity: t_review 0.02 at €2 / €5 / €20 (cost €298.16 / €541.16 / €1,756.16).

### STRATIFIED split — tuned xgb_u — calibration: NONE
| candidate | Brier | log-loss | ECE | PR-AUC |
|---|---|---|---|---|
| none | 0.000306 | 0.001922 | 0.000226 | 0.9089 |
| sigmoid | 0.000305 | 0.002290 | 0.000120 | 0.9089 |
| isotonic | 0.000294 | 0.001817 | 0.000138 | 0.8881 |
- How applied: isotonic lowest Brier but only 3.8% below 'none' (within 5%) -> none.
- **t_review = 0.06 — recall >= 0.85 achieved**: recall 0.9149 (43/47), precision 0.8431,
  1.80 alerts/1k, expected cost €1,521.53.
- **t_block = 0.22** — precision 0.9111 (41 TP, 4 FP), recall 0.8723. precision >= 0.90 achievable.
- Tiers: HIGH 45 (41 fraud, share 0.911); MEDIUM 6 (2, 0.333); LOW 28,322 (4).
- Stability: t_review median 0.06, IQR [0.06, 0.22], target met in 93.5%; t_block median 0.22,
  IQR [0.09, 0.41], exists in 92%.
- Cost sensitivity: t_review 0.06 at €2 / €5 / €20 (€1,368.53 / €1,521.53 / €2,286.53).

### Small-data warning (also in README)
Each validation half holds only 24-47 frauds. On the time split one fraud is ~3 pp of recall,
the t_review IQR spans 0.02-0.34 and the target is met in under half of the resamples. The
time-split isotonic map is fitted on 24 positives. Treat calibration and thresholds as rough.

### Artifacts & housekeeping
- policy_{split}.json (t_review, t_block, operating point, tiers, stability, sensitivity)
  REPLACES frozen_threshold_{split}.json (removed; recoverable from commit 17db32d).
  New: tier_table_{split}.csv; calibration_{split}.json now holds all four candidate metrics.
- models/final_model_time.joblib = isotonic-calibrated tuned logreg;
  models/final_model_stratified.joblib = uncalibrated tuned xgb_u.
- The full-spec re-run reproduced threshold tables, sensitivity tables, plots, sub-split indices
  and both model files byte-for-byte versus commit 17db32d.

### Open for KING before Phase 9 pre-registration
- TIME t_review is a documented FALLBACK (recall 0.8485 < 0.85, one fraud short on valid_thr).
  Accept as-is for pre-registration, or change the grid (a protocol change -> log here).
- valid_cal (time) has 24 frauds (< 30): the isotonic map rests on few positives.

## Phase 9 PRE-REGISTRATION (written 2026-10-05, BEFORE any *_test.parquet read)
Phase 8 commit: ef3d4b636d4152872ea8a1ef9402af7888f3c55b
("feat(policy): calibration and two-threshold decision policy").
Machine-readable copy: reports/metrics/preregistration.json (evaluate.py checks against it).
Authoritative source for frozen policy/thresholds: reports/metrics/policy_{split}.json.
Thresholds are NOT recomputed; the MD5s are an extra integrity check only.

### Frozen decisions
TIME split (HEADLINE)
- champion: tuned Logistic Regression (logreg_tuned), MLflow run 5561385ba3144bf9ae85f73d7e56d43c
- params file: reports/metrics/best_params_logreg_time.json (C 0.000802, class_weight None)
- calibration: isotonic (fitted on valid_cal; 24-positive warning accepted)
- t_review 0.02 | t_block 0.23 | review_cost €5
- validation t_review recall 0.8485 — fallback: TRUE (accepted by KING)
- artifact: models/final_model_time.joblib, md5 bf798b5c02c429051303c99c7ab75240

STRATIFIED split (secondary)
- champion: tuned XGBoost xgb_u (xgb_u_tuned), MLflow run 15a37803c60b4c11b7f6e02007e6ff38
- params file: reports/metrics/best_params_xgb_u_stratified.json (u 0.9273)
- calibration: none
- t_review 0.06 | t_block 0.22 | review_cost €5
- validation t_review recall 0.9149 — fallback: false
- artifact: models/final_model_stratified.joblib, md5 edd571430f01e4d50e7306180de1a458

### Protocol
- Primary metric: PR-AUC. Time split = headline; stratified = secondary. Accuracy never headlined.
- Test scored ONCE. evaluate.py is the only module that loads *_test.parquet; it refuses to
  re-run if final_{split}.json exists unless --force, and prints why that is bad practice.
- Before scoring, evaluate.py refuses to run if: artifact MD5 differs from the above, or
  policy_{split}.json disagrees with this pre-registration (model, run id, calibration,
  t_review, t_block, review cost), or policy and champion_{split}.json disagree.
- Nothing is fitted, tuned, recalibrated, refitted, selected or modified using test data.
  evaluate.py does not load validation data or import threshold-selection/calibration code.
- Metrics (metrics.compute_metrics) at t_review and at t_block: PR-AUC, ROC-AUC, precision,
  recall, F1, specificity, FPR, confusion matrix, alerts/1k, % fraud amount caught,
  expected cost; plus the HIGH/MEDIUM/LOW tier table on test.
- Uncertainty: 1,000 stratified bootstrap resamples, 95% percentile CI, seed 42, for PR-AUC,
  recall, precision, alerts/1k, expected cost.
- Reference costs: no-model = approve everything / flag nothing / all fraud Amounts lost;
  flag-everything = review_cost x every row, nothing missed.
- Context-only baselines (Phase 5 frozen MLflow runs; PR-AUC + ROC-AUC, PR-AUC with bootstrap
  CI): Dummy, untuned Logistic Regression, Isolation Forest — time: e4e4333f / 6c2623cb /
  0f2306c5; stratified: 56fd80ab / 834a7f03 / 18a1b265. They cannot change any decision.
- Outputs: reports/metrics/final_{split}.json, final_summary.md (paragraph on the
  stratified-vs-time gap left for KING), confusion-matrix and PR-curve figures with the
  t_review / t_block operating points marked.
- Pre-declared caveats: small test sets (time 74 fraud, stratified 95 fraud — counts from
  splits_{split}.json, Phase 4) -> wide CIs. Time t_review was a validation fallback, so test
  recall may be below 0.85; that will be reported, not fixed.
