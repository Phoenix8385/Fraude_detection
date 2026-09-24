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
- Phase 7 tuned valid PR-AUC:
- Phase 8 calibration kept? Brier before/after:
- Phase 8 frozen threshold (strat / time):
- PRE-REGISTRATION before Phase 9 (model, params file, calibrated, threshold, date):
- Phase 9 final test metrics:
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
