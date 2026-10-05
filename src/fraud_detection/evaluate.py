"""Phase 9 — one-shot final evaluation on the TEST split (docs/evaluation.md, "Final test").

This is the ONLY module allowed to load *_test.parquet. It APPLIES the frozen Phase 8
decisions; it never fits, tunes, recalibrates, refits or selects anything:
- frozen policy / thresholds: reports/metrics/policy_{split}.json (authoritative);
- frozen model (with calibrator if any): models/final_model_{split}.joblib (Phase 8);
- pre-registration: reports/metrics/preregistration.json (written before any test read).
Before scoring, it refuses to run if the artifact MD5, the policy JSON, the champion JSON and
the pre-registration do not all agree. It does not load validation data and does not import
any threshold-selection or calibration-fitting code.

Context-only baselines (Phase 5 Dummy, untuned LR, Isolation Forest) are scored for PR-AUC
and ROC-AUC; they cannot change any decision.

Re-running is refused if final_{split}.json exists, unless --force is passed.

    python -m fraud_detection.evaluate --splits time stratified
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # file output only, no window

import matplotlib.pyplot as plt  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from fraud_detection.bootstrap import bootstrap_samples, summarize_ci  # noqa: E402
from fraud_detection.config import Config, load_config  # noqa: E402
from fraud_detection.features import MODEL_FEATURES  # noqa: E402
from fraud_detection.metrics import compute_metrics  # noqa: E402
from fraud_detection.policy import tier_table  # noqa: E402
from fraud_detection.schema import TARGET  # noqa: E402
from fraud_detection.train import SPLIT_NAMES, setup_mlflow  # noqa: E402

logger = logging.getLogger(__name__)

RERUN_WARNING = (
    "REFUSING TO RE-RUN THE TEST EVALUATION.\n"
    "The test split is the one-time exam. Every extra look at test results invites changing "
    "something (model, calibration, threshold) until the test numbers look better, which turns "
    "the test set into a second validation set and makes the reported numbers optimistic and "
    "no longer an honest estimate of performance on unseen data.\n"
    "Existing results: {paths}\n"
    "Pass --force only for a documented reason (e.g. a crash mid-run), and log it in "
    "docs/memory.md."
)
# Fields that must agree between policy_{split}.json and the pre-registration.
FROZEN_FIELDS: tuple[str, ...] = ("mlflow_run_id", "calibration", "t_review", "t_block",
                                  "review_cost")  # fmt: skip


def load_test(split: str, processed_dir: Path) -> pd.DataFrame:
    """The ONLY loader of the test part in the whole package."""
    return pd.read_parquet(processed_dir / f"{split}_test.parquet")


def existing_results(splits: list[str], metrics_dir: Path) -> list[Path]:
    """final_{split}.json files that already exist for the requested splits."""
    return [p for s in splits if (p := metrics_dir / f"final_{s}.json").exists()]


def guard_rerun(splits: list[str], metrics_dir: Path, force: bool) -> None:
    """Refuse (SystemExit 1) if results exist and --force was not passed. Runs before any load."""
    done = existing_results(splits, metrics_dir)
    if done and not force:
        print(RERUN_WARNING.format(paths=", ".join(p.name for p in done)))
        raise SystemExit(1)
    if done:
        logger.warning("--force: overwriting %s; record the reason in docs/memory.md", done)


def file_md5(path: Path) -> str:
    """Hex MD5 of a file (integrity check for the frozen artifact)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def verify_frozen(
    split: str, policy: dict[str, Any], champion: dict[str, Any], prereg: dict[str, Any]
) -> None:
    """Raise ValueError unless policy, champion and pre-registration agree for this split."""
    frozen = prereg["splits"][split]
    problems = []
    if policy["model"] != frozen["champion"]:
        problems.append(f"policy model {policy['model']} != pre-registered {frozen['champion']}")
    for field in FROZEN_FIELDS:
        a, b = policy[field], frozen[field]
        same = a == b if (a is None or b is None or isinstance(a, str)) else math.isclose(a, b)
        if not same:
            problems.append(f"{field}: policy {a!r} != pre-registered {b!r}")
    if champion["decision"]["champion"] != policy["model"]:
        problems.append("champion_{split}.json and policy_{split}.json name different models")
    run = champion["models"][policy["model"]]["mlflow_run_id"]
    if run != policy["mlflow_run_id"]:
        problems.append("champion and policy MLflow run ids differ")
    if problems:
        raise ValueError(f"{split}: frozen configuration mismatch: " + "; ".join(problems))


def check_artifact(path: Path, expected_md5: str) -> None:
    """Raise ValueError if the frozen artifact was changed after pre-registration."""
    actual = file_md5(path)
    if actual != expected_md5:
        raise ValueError(f"{path.name}: md5 {actual} != pre-registered {expected_md5}")


def reference_costs(y: pd.Series, amounts: pd.Series, review_cost: float) -> dict[str, Any]:
    """No-model (flag nothing) and flag-everything costs, via compute_metrics."""
    n = len(y)
    nothing = compute_metrics(y, np.zeros(n), 1.0, amounts, review_cost)  # 0 >= 1: no alerts
    everything = compute_metrics(y, np.ones(n), 0.0, amounts, review_cost)  # all alerts
    return {
        "no_model": {"expected_cost": nothing["expected_cost"], "alerts": 0,
                     "definition": "approve everything / flag nothing: all fraud Amounts lost"},
        "flag_everything": {"expected_cost": everything["expected_cost"], "alerts": n,
                            "definition": "alert on every row: review_cost x rows, none missed"},
    }  # fmt: skip


def score_frozen(
    y: pd.Series,
    proba: np.ndarray,
    amounts: pd.Series,
    t_review: float,
    t_block: float | None,
    review_cost: float,
) -> dict[str, Any]:
    """Apply the frozen thresholds as given (never recomputed)."""
    at_block = (
        compute_metrics(y, proba, t_block, amounts, review_cost) if t_block is not None else None
    )
    return {
        "at_t_review": compute_metrics(y, proba, t_review, amounts, review_cost),
        "at_t_block": at_block,
        "tiers": tier_table(y, proba, t_review, t_block).to_dict(orient="records"),
    }


def plot_confusion(scores: dict[str, Any], split: str, path: Path) -> None:
    """Confusion matrices at t_review and (if it exists) t_block."""
    points = [("t_review", scores["at_t_review"])]
    if scores["at_t_block"] is not None:
        points.append(("t_block", scores["at_t_block"]))
    fig, axes = plt.subplots(1, len(points), figsize=(5 * len(points), 4.5), squeeze=False)
    for ax, (name, m) in zip(axes[0], points, strict=True):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["pred legit", "pred fraud"])
        ax.set_yticks([0, 1], ["true legit", "true fraud"])
        ax.set_title(f"{split} TEST — {name} = {m['threshold']:.2f}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr(y: pd.Series, proba: np.ndarray, scores: dict[str, Any], split: str,
            path: Path) -> None:  # fmt: skip
    """Test PR curve with the frozen operating points marked."""
    precision, recall, _ = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ap = scores["at_t_review"]["pr_auc"]
    ax.step(recall, precision, where="post", label=f"PR curve (AP={ap:.3f})")
    for name, marker in (("t_review", "o"), ("t_block", "s")):
        m = scores[f"at_{name}"]
        if m is not None:
            label = (
                f"{name} {m['threshold']:.2f} (R {m['recall']:.3f}, P {m['precision']:.3f})"
            )
            ax.plot(m["recall"], m["precision"], marker, ms=10, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision-recall — {split} TEST (n fraud = {int(np.sum(y))})")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_split(split: str, config: Config) -> dict[str, Any]:
    """Verify the frozen configuration, then score TEST once for one split."""
    metrics_dir, figures_dir = config.paths.metrics_dir, config.paths.figures_dir
    prereg = json.loads((metrics_dir / "preregistration.json").read_text(encoding="utf-8"))
    policy = json.loads((metrics_dir / f"policy_{split}.json").read_text(encoding="utf-8"))
    champion = json.loads((metrics_dir / f"champion_{split}.json").read_text(encoding="utf-8"))
    frozen = prereg["splits"][split]

    # All integrity checks happen BEFORE the test parquet is opened.
    verify_frozen(split, policy, champion, prereg)
    artifact = config.paths.models_dir / f"final_model_{split}.joblib"
    check_artifact(artifact, frozen["artifact_md5"])
    model = joblib.load(artifact)
    baselines = {
        name: mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        for name, run_id in prereg["context_baselines_phase5"][split].items()
    }
    logger.info("%s: frozen config verified (md5 %s); loading TEST", split, frozen["artifact_md5"])

    test = load_test(split, config.paths.processed_dir)
    X, y, amounts = test[MODEL_FEATURES], test[TARGET], test["Amount"]
    t_review, t_block = policy["t_review"], policy["t_block"]  # authoritative, not recomputed
    review_cost = policy["review_cost"]
    proba = model.predict_proba(X)[:, 1]
    scores = score_frozen(y, proba, amounts, t_review, t_block, review_cost)
    base_proba = {name: m.predict_proba(X)[:, 1] for name, m in baselines.items()}

    boot = prereg["bootstrap"]
    probas = {"t_review": proba, **{f"baseline_{k}": v for k, v in base_proba.items()}}
    thresholds = {"t_review": t_review, **{f"baseline_{k}": 0.5 for k in base_proba}}
    if t_block is not None:
        probas["t_block"], thresholds["t_block"] = proba, t_block
    samples = bootstrap_samples(
        y, probas, thresholds, amounts, review_cost, boot["n_resamples"], boot["seed"]
    )
    level = boot["ci_level"]
    ci_review = summarize_ci(scores["at_t_review"], samples["t_review"], level)
    ci_block = None
    if t_block is not None:
        ci_block = summarize_ci(scores["at_t_block"], samples["t_block"], level)
    baseline_out = {}
    for name, p in base_proba.items():
        m = compute_metrics(y, p, 0.5, amounts, review_cost)
        ci = summarize_ci(m, samples[f"baseline_{name}"][["pr_auc"]], level)["pr_auc"]
        baseline_out[name] = {
            "mlflow_run_id": prereg["context_baselines_phase5"][split][name],
            "pr_auc": m["pr_auc"], "pr_auc_ci": [ci["ci_low"], ci["ci_high"]],
            "roc_auc": m["roc_auc"],
        }  # fmt: skip

    plot_confusion(scores, split, figures_dir / f"final_confusion_{split}.png")
    plot_pr(y, proba, scores, split, figures_dir / f"final_pr_curve_{split}.png")

    result = {
        "split": split,
        "headline": split == prereg["headline_split"],
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "phase8_commit": prereg["phase8_commit"],
        "artifact": frozen["artifact"],
        "artifact_md5_verified": frozen["artifact_md5"],
        "frozen_policy": {k: policy[k] for k in ("model", "mlflow_run_id", "calibration",
                                                  "t_review", "t_block", "review_cost")},
        "validation_t_review_recall": frozen["valid_t_review_recall"],
        "validation_fallback": frozen["fallback"],
        "test_rows": len(test),
        "test_fraud": int(y.sum()),
        "at_t_review": scores["at_t_review"],
        "at_t_review_ci": ci_review,
        "at_t_block": scores["at_t_block"],
        "at_t_block_ci": ci_block,
        "tiers_test": scores["tiers"],
        "reference_costs": reference_costs(y, amounts, review_cost),
        "context_baselines": baseline_out,
        "bootstrap": boot,
    }  # fmt: skip
    out = metrics_dir / f"final_{split}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %s", out)
    return result


def _ci(d: dict[str, float], fmt: str) -> str:
    return f"{d['point']:{fmt}} [{d['ci_low']:{fmt}}, {d['ci_high']:{fmt}}]"


def summary_markdown(results: list[dict[str, Any]]) -> str:
    """final_summary.md: one table for both splits + a placeholder paragraph for KING."""
    lines = [
        "# Final TEST evaluation (Phase 9, run once)",
        "",
        "Frozen Phase 8 models and thresholds applied to the TEST split (pre-registered in",
        "docs/memory.md and reports/metrics/preregistration.json). Primary metric: PR-AUC.",
        "Time split = headline. 95% stratified bootstrap CIs (1,000 resamples, seed 42).",
        "",
        "| split | model | calibration | t_review | PR-AUC [CI] | recall [CI] | precision [CI] "
        "| alerts/1k [CI] | cost € [CI] | no-model € | flag-all € |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ci, fp, ref = r["at_t_review_ci"], r["frozen_policy"], r["reference_costs"]
        lines.append(
            f"| {r['split']}{' (headline)' if r['headline'] else ''} | {fp['model']} "
            f"| {fp['calibration']} | {fp['t_review']:.2f} | {_ci(ci['pr_auc'], '.4f')} "
            f"| {_ci(ci['recall'], '.3f')} | {_ci(ci['precision'], '.3f')} "
            f"| {_ci(ci['alerts_per_1000'], '.2f')} | {_ci(ci['expected_cost'], ',.0f')} "
            f"| {ref['no_model']['expected_cost']:,.0f} "
            f"| {ref['flag_everything']['expected_cost']:,.0f} |"
        )
    lines += [
        "", "## t_block (HIGH / HOLD)", "",
        "| split | t_block | precision [CI] | recall [CI] | TP | FP |", "|---|---|---|---|---|---|",
    ]  # fmt: skip
    for r in results:
        b, ci = r["at_t_block"], r["at_t_block_ci"]
        if b is None:
            lines.append(f"| {r['split']} | none (does not exist) | — | — | — | — |")
        else:
            lines.append(
                f"| {r['split']} | {b['threshold']:.2f} | {_ci(ci['precision'], '.3f')} "
                f"| {_ci(ci['recall'], '.3f')} | {b['tp']} | {b['fp']} |"
            )
    lines += ["", "## Context-only baselines (Phase 5, frozen; cannot change any decision)", "",
              "| split | baseline | PR-AUC [CI] | ROC-AUC |", "|---|---|---|---|"]  # fmt: skip
    for r in results:
        for name, b in r["context_baselines"].items():
            lo, hi = b["pr_auc_ci"]
            lines.append(
                f"| {r['split']} | {name} | {b['pr_auc']:.4f} [{lo:.4f}, {hi:.4f}] "
                f"| {b['roc_auc']:.4f} |"
            )
    lines += [
        "",
        "## Stratified vs time — why the results differ",
        "",
        "_TODO (KING): write this paragraph._",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Phase 9: one-shot TEST evaluation.")
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=["time", "stratified"])
    parser.add_argument("--force", action="store_true", help="overwrite existing final results")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    guard_rerun(args.splits, config.paths.metrics_dir, args.force)  # before ANY loading
    setup_mlflow(config)
    results = [evaluate_split(s, config) for s in args.splits]
    md = summary_markdown(results)
    (config.paths.metrics_dir / "final_summary.md").write_text(md + "\n", encoding="utf-8")
    print(md.encode("ascii", "replace").decode("ascii"))  # Windows console safe


if __name__ == "__main__":
    main()
