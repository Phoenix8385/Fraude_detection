"""Champion decision on VALIDATION (docs/evaluation.md, "Champion rule (pre-registered)").

Reads the tuned configs written by tune.py, reloads the logged models from MLflow (no
retraining), and on the full VALIDATION part computes:
- 95% stratified bootstrap CIs for PR-AUC and, at each model's validation operating point,
  recall / precision / alerts per 1,000 / expected cost;
- a PAIRED bootstrap of the PR-AUC difference xgb_u - logreg (prob_a_better = P(xgb_u wins));
- 5-seed refits of each tuned config (seed-to-seed spread of validation PR-AUC).

Rule: highest tuned validation PR-AUC. If the paired 95% CI of the difference vs tuned LR
includes 0: tie-break 1 = lower expected cost at recall >= target on validation;
tie-break 2 = simpler model.

The operating point (min expected cost s.t. recall >= target, threshold.py grid) is chosen
on the same validation rows it is reported on, so its numbers are optimistic. It is used
only for tie-break 1; Phase 8 re-chooses thresholds on valid_thr. The test split is never read.

    python -m fraud_detection.champion --splits stratified time
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd

from fraud_detection.bootstrap import bootstrap_samples, paired_difference, summarize_ci
from fraud_detection.compare import latest_runs
from fraud_detection.config import Config, load_config
from fraud_detection.features import MODEL_FEATURES
from fraud_detection.metrics import compute_metrics
from fraud_detection.schema import TARGET
from fraud_detection.threshold import choose_threshold, threshold_table
from fraud_detection.train import SPLIT_NAMES, load_part, setup_mlflow
from fraud_detection.tune import TUNED_MODELS, build_tuned

logger = logging.getLogger(__name__)

CHALLENGER = "xgb_u"
BASELINE = "logreg"  # the rule compares every candidate against tuned LR
SIMPLICITY_RANK: dict[str, int] = {"logreg": 0, "xgb_u": 1}  # lower = simpler
UNTUNED_REFERENCE: tuple[str, ...] = ("logreg", "xgb", "xgb_weighted")  # Phase 5/6, context only


def choose_champion(
    pr_auc: Mapping[str, float],
    operating_points: Mapping[str, Mapping[str, Any]],
    paired: Mapping[str, Any],
) -> dict[str, str]:
    """Apply the pre-registered champion rule. Pure function.

    Args:
        pr_auc: model -> tuned validation PR-AUC.
        operating_points: model -> {"met_target": bool, "expected_cost": float}.
        paired: paired PR-AUC difference vs tuned LR, with key "ci_includes_zero".

    Tie-break 1 compares cost "at recall >= target": a model whose best threshold does not
    reach the target has no such cost, so it loses to one that does; if neither reaches
    it, tie-break 1 cannot decide and tie-break 2 applies.
    """
    top = max(pr_auc, key=lambda m: pr_auc[m])
    if not paired["ci_includes_zero"]:
        return {
            "champion": top,
            "step": "rule",
            "reason": "highest tuned validation PR-AUC; paired 95% CI of the difference "
            "vs tuned LR excludes 0",
        }

    met = [m for m in pr_auc if operating_points[m]["met_target"]]
    if len(met) == 1:
        return {
            "champion": met[0],
            "step": "tie-break 1",
            "reason": "PR-AUC CI vs tuned LR includes 0; only this model reaches the recall "
            "target on validation",
        }
    if len(met) > 1:
        costs = {m: operating_points[m]["expected_cost"] for m in met}
        if len(set(costs.values())) > 1:
            return {
                "champion": min(costs, key=lambda m: costs[m]),
                "step": "tie-break 1",
                "reason": "PR-AUC CI vs tuned LR includes 0; lower expected cost at the "
                "recall target on validation",
            }
    return {
        "champion": min(pr_auc, key=lambda m: SIMPLICITY_RANK[m]),
        "step": "tie-break 2",
        "reason": "PR-AUC CI vs tuned LR includes 0 and tie-break 1 cannot separate the "
        "models (recall target / equal cost); simpler model",
    }


def operating_point(
    y: pd.Series, proba: np.ndarray, amounts: pd.Series, review_cost: float, target_recall: float
) -> dict[str, Any]:
    """Min-expected-cost threshold s.t. recall >= target on these rows (threshold.py rule)."""
    chosen = choose_threshold(threshold_table(y, proba, amounts, review_cost), target_recall)
    keys = ("threshold", "recall", "precision", "alerts_per_1000", "expected_cost",
            "tp", "fp", "fn", "met_target", "rule")  # fmt: skip
    return {k: chosen[k] for k in keys}


def seed_stability(
    model_key: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    seeds: tuple[int, ...],
    review_cost: float,
) -> dict[str, Any]:
    """Refit one tuned config with each seed; validation PR-AUC per seed, mean and std."""
    values = []
    for seed in seeds:
        model = build_tuned(model_key, params, train[TARGET], seed)
        model.fit(train[MODEL_FEATURES], train[TARGET])
        proba = model.predict_proba(valid[MODEL_FEATURES])[:, 1]
        m = compute_metrics(valid[TARGET], proba, 0.5, valid["Amount"], review_cost)
        values.append(m["pr_auc"])
    return {
        "seeds": list(seeds),
        "valid_pr_auc": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,  # sample std
    }


def _load_tuned(model_key: str, split: str, metrics_dir: Path) -> dict[str, Any]:
    path = metrics_dir / f"best_params_{model_key}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing; run python -m fraud_detection.tune first")
    return json.loads(path.read_text(encoding="utf-8"))


def run_champion(split: str, config: Config) -> dict[str, Any]:
    """Full Phase 7 decision for one split; writes champion_{split}.json."""
    metrics_dir = config.paths.metrics_dir
    train = load_part(split, "train", config.paths.processed_dir)
    valid = load_part(split, "valid", config.paths.processed_dir)
    y, amounts = valid[TARGET], valid["Amount"]
    review_cost = config.costs.review_cost_per_alert
    level = config.bootstrap.ci_level

    tuned = {m: _load_tuned(m, split, metrics_dir) for m in TUNED_MODELS}
    probas, points, ops = {}, {}, {}
    for m, info in tuned.items():
        model = mlflow.sklearn.load_model(f"runs:/{info['mlflow_run_id']}/model")
        probas[m] = model.predict_proba(valid[MODEL_FEATURES])[:, 1]
        ops[m] = operating_point(y, probas[m], amounts, review_cost, config.target_recall)
        points[m] = compute_metrics(y, probas[m], ops[m]["threshold"], amounts, review_cost)
        if not math.isclose(points[m]["pr_auc"], info["valid_pr_auc"], rel_tol=1e-9):
            raise RuntimeError(f"{m}/{split}: reloaded PR-AUC differs from tune.py's value")

    logger.info("%s: bootstrapping %d resamples", split, config.bootstrap.n_resamples)
    samples = bootstrap_samples(
        y, probas, {m: ops[m]["threshold"] for m in probas}, amounts, review_cost,
        config.bootstrap.n_resamples, config.seed,
    )  # fmt: skip
    paired = paired_difference(
        samples[CHALLENGER]["pr_auc"], samples[BASELINE]["pr_auc"],
        points[CHALLENGER]["pr_auc"], points[BASELINE]["pr_auc"], level,
    )  # fmt: skip
    paired = {"metric": "pr_auc", "a": CHALLENGER, "b": BASELINE, **paired}

    models: dict[str, Any] = {}
    for m, info in tuned.items():
        logger.info("%s: %d-seed refits of %s", split, len(config.stability_seeds), m)
        spw = {"scale_pos_weight": info["scale_pos_weight"]} if "scale_pos_weight" in info else {}
        models[m] = {
            "best_params": info["best_params"],
            **spw,
            "cv_pr_auc_mean": info["cv_pr_auc_mean"],
            "cv_pr_auc_std": info["cv_pr_auc_std"],
            "valid": summarize_ci(points[m], samples[m], level),
            "operating_point": ops[m],
            "seed_stability": seed_stability(
                m, info["best_params"], train, valid, config.stability_seeds, review_cost
            ),
            "mlflow_run_id": info["mlflow_run_id"],
        }

    decision = choose_champion({m: points[m]["pr_auc"] for m in points}, ops, paired)
    runs = latest_runs(metrics_dir / "experiments.csv")
    untuned = runs[(runs["split"] == split) & runs["model"].isin(UNTUNED_REFERENCE)]
    result = {
        "split": split,
        "valid_rows": len(valid),
        "valid_fraud": int(y.sum()),
        "bootstrap": {
            "n_resamples": config.bootstrap.n_resamples,
            "ci_level": level,
            "seed": config.seed,
            "stratified": True,
        },
        "models": models,
        "paired_pr_auc": paired,
        "decision": decision,
        "untuned_valid_pr_auc_reference": dict(
            zip(untuned["model"], untuned["pr_auc"], strict=True)
        ),
        "notes": [
            "Operating points are chosen and reported on the same validation rows (optimistic); "
            "used only for tie-break 1. Phase 8 re-selects thresholds on valid_thr.",
            "Threshold grid is 0.01-0.95 (threshold.py); met_target=false means no grid "
            "threshold reaches the recall target.",
        ],
    }
    out = metrics_dir / f"champion_{split}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %s", out)
    return result


def _ci(d: Mapping[str, float], fmt: str = ".4f") -> str:
    return f"{d['point']:{fmt}} [{d['ci_low']:{fmt}}, {d['ci_high']:{fmt}}]"


def summary_markdown(results: list[dict[str, Any]]) -> str:
    """Human-readable Phase 7 summary of champion_{split}.json files."""
    lines = [
        "# Phase 7 — tuned models, bootstrap CIs, champion decision (VALIDATION only)",
        "",
        "Generated by `python -m fraud_detection.champion` from `champion_{split}.json`.",
        "Tuning: Optuna (TPE), CV on TRAIN only. CIs: 95% percentile stratified bootstrap on",
        "the full validation part. Operating point = min expected cost s.t. recall >= 0.85,",
        "chosen on the same validation rows (optimistic; tie-break input only). Test not used.",
    ]
    for r in results:
        lines += [
            "",
            f"## {r['split']} split (valid rows {r['valid_rows']:,}, fraud {r['valid_fraud']}, "
            f"{r['bootstrap']['n_resamples']} resamples)",
            "",
            "| model | cv PR-AUC (mean ± std) | valid PR-AUC [95% CI] | 5-seed PR-AUC mean ± std "
            "| threshold | recall [CI] | precision [CI] | alerts/1k [CI] | cost € [CI] |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for m, d in r["models"].items():
            v, op, s = d["valid"], d["operating_point"], d["seed_stability"]
            flag = "" if op["met_target"] else " (target not met)"
            lines.append(
                f"| {m}_tuned | {d['cv_pr_auc_mean']:.4f} ± {d['cv_pr_auc_std']:.4f} "
                f"| {_ci(v['pr_auc'])} | {s['mean']:.4f} ± {s['std']:.4f} "
                f"| {op['threshold']:.2f}{flag} | {_ci(v['recall'], '.3f')} "
                f"| {_ci(v['precision'], '.3f')} | {_ci(v['alerts_per_1000'], '.2f')} "
                f"| {_ci(v['expected_cost'], '.0f')} |"
            )
        p, dec = r["paired_pr_auc"], r["decision"]
        lines += [
            "",
            f"Paired bootstrap PR-AUC ({p['a']} - {p['b']}): {p['diff']:+.4f} "
            f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}], prob_a_better = {p['prob_a_better']:.3f}, "
            f"CI includes 0: {p['ci_includes_zero']}.",
            "",
            f"**Champion: {dec['champion']}** — {dec['step']}: {dec['reason']}.",
            "",
            "Best params: "
            + "; ".join(f"{m}: `{json.dumps(d['best_params'])}`" for m, d in r["models"].items()),
            "",
            "Untuned Phase 5/6 valid PR-AUC (context): "
            + ", ".join(f"{m} {v:.4f}" for m, v in r["untuned_valid_pr_auc_reference"].items()),
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Phase 7 champion decision (validation only).")
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    setup_mlflow(config)
    results = [run_champion(s, config) for s in args.splits]
    md = summary_markdown(results)
    (config.paths.metrics_dir / "champion_summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
