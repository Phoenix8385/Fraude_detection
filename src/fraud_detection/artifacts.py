"""Phase 10 — the versioned production artifact and the drift reference profile.

models/model.joblib     byte-identical copy of the frozen Phase 8 headline model
                        (models/final_model_time.joblib: tuned LR + isotonic). Never re-dumped,
                        retrained or modified; its MD5 must equal the pre-registered one.
models/model_meta.json  version, thresholds (from policy_time.json), features, schema_hash,
                        library versions, git commit, final test metrics (final_time.json).
models/reference_profile.json
                        quantile bin edges + reference proportions for Amount and the model
                        score, computed on valid_thr (time split). No V1..V28 profiles and no
                        raw rows: aggregates only. Used by drift.py (PSI) in Phase 17.

    python -m fraud_detection.artifacts
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from fraud_detection.config import PROJECT_ROOT, Config, load_config
from fraud_detection.features import MODEL_FEATURES
from fraud_detection.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_VERSION = "logreg-iso-v1.0.0"
HEADLINE_SPLIT = "time"
ARTIFACT_NAME = "model.joblib"
META_NAME = "model_meta.json"
PROFILE_NAME = "reference_profile.json"
PROFILE_BINS = 10  # quantile (decile) bins; tied edges are merged
LIBRARIES: tuple[str, ...] = ("sklearn", "numpy", "pandas", "joblib", "xgboost", "imblearn")


def file_md5(path: Path) -> str:
    """Hex MD5 of a file."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def schema_hash() -> str:
    """SHA-256 of the raw input columns (schema.py) and the ordered model feature list.

    Any change to the request contract or feature order changes this hash.
    """
    contract = {"input_columns": FEATURE_COLUMNS, "input_dtype": "float64",
                "model_features": MODEL_FEATURES}  # fmt: skip
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def library_versions() -> dict[str, str]:
    """Versions of the libraries the pickled model depends on."""
    import importlib

    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        versions[name] = importlib.import_module(name).__version__
    return versions


def git_commit(root: Path = PROJECT_ROOT) -> str | None:
    """Current HEAD commit, or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True)  # fmt: skip
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def reference_profile(values: ArrayLike, n_bins: int = PROFILE_BINS) -> dict[str, Any]:
    """Quantile bin edges + the share of reference rows per bin.

    Inner edges are the reference quantiles; duplicate edges (ties, e.g. isotonic scores)
    are merged, so a tied distribution may have fewer than n_bins bins. The outer bins are
    open (-inf, e1) and [ek, +inf) so any future value falls in some bin.
    """
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        raise ValueError("reference profile needs at least one value")
    inner = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1))[1:-1])
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    counts = np.histogram(x, bins=edges)[0]
    return {
        "edges": [None if np.isinf(e) else float(e) for e in edges],  # None = -inf / +inf
        "proportions": (counts / counts.sum()).tolist(),
        "n_bins": len(counts),
        "n_rows": int(len(x)),
    }


def build_meta(config: Config, source: Path, source_md5: str) -> dict[str, Any]:
    """model_meta.json content, read from the frozen Phase 7-9 outputs (nothing recomputed)."""
    metrics_dir = config.paths.metrics_dir

    def read(name: str) -> dict[str, Any]:
        return json.loads((metrics_dir / name).read_text(encoding="utf-8"))

    policy, prereg = read(f"policy_{HEADLINE_SPLIT}.json"), read("preregistration.json")
    final, champion = read(f"final_{HEADLINE_SPLIT}.json"), read(f"champion_{HEADLINE_SPLIT}.json")
    at, ci = final["at_t_review"], final["at_t_review_ci"]
    return {
        "model_version": MODEL_VERSION,
        "model_name": f"{policy['model']}_tuned",
        "headline_split": HEADLINE_SPLIT,
        "mlflow_run_id": policy["mlflow_run_id"],
        "best_params": champion["models"][policy["model"]]["best_params"],
        "calibrated": policy["calibration"] != "none",
        "calibration": policy["calibration"],
        "thresholds": {"review": policy["t_review"], "block": policy["t_block"]},
        "review_cost": policy["review_cost"],
        "t_review_validation_fallback": policy["t_review_detail"]["fallback"],
        "features": MODEL_FEATURES,
        "input_columns": FEATURE_COLUMNS,
        "schema_hash": schema_hash(),
        "source_artifact": source.name,
        "artifact_md5": source_md5,
        "phase8_commit": prereg["phase8_commit"],
        "trained": "Phase 7 MLflow run (see mlflow_run_id); calibrated in Phase 8 (phase8_commit)",
        "final_test_metrics": {
            "source": f"reports/metrics/final_{HEADLINE_SPLIT}.json",
            "test_rows": final["test_rows"], "test_fraud": final["test_fraud"],
            "pr_auc": ci["pr_auc"], "recall": ci["recall"], "precision": ci["precision"],
            "alerts_per_1000": ci["alerts_per_1000"], "expected_cost": ci["expected_cost"],
            "roc_auc": at["roc_auc"],
        },  # fmt: skip
        "explanations": "coef x scaled feature, log-odds of the UNCALIBRATED LR; not causes",
        "reference_profile": PROFILE_NAME,
        "library_versions": library_versions(),
        "git_commit": git_commit(),
        "packaged_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def save_artifact(source: Path, models_dir: Path, meta: dict[str, Any]) -> Path:
    """Copy the frozen model byte-for-byte to models/model.joblib and write the metadata."""
    dest = models_dir / ARTIFACT_NAME
    shutil.copyfile(source, dest)
    if file_md5(dest) != meta["artifact_md5"]:
        raise RuntimeError(f"{dest.name} is not byte-identical to {source.name}")
    (models_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dest


def load_artifact(models_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load model.joblib + metadata, verifying integrity before unpickling.

    Raises on a changed file (MD5), feature list or schema; warns on library version drift
    (a pickled sklearn model may behave differently under another version).
    """
    import joblib

    meta = json.loads((models_dir / META_NAME).read_text(encoding="utf-8"))
    path = models_dir / ARTIFACT_NAME
    if file_md5(path) != meta["artifact_md5"]:
        raise ValueError(f"{path.name} MD5 does not match model_meta.json")
    if meta["features"] != MODEL_FEATURES:
        raise ValueError("model_meta.json feature list differs from MODEL_FEATURES")
    if meta["schema_hash"] != schema_hash():
        raise ValueError("schema_hash differs: the input contract changed since packaging")
    current = library_versions()
    for lib, version in meta["library_versions"].items():
        if current.get(lib) != version:
            warnings.warn(f"{lib} {current.get(lib)} != packaged {version}", stacklevel=2)
    return joblib.load(path), meta


def main() -> None:
    """CLI: package the frozen headline model and build the reference profile."""
    from fraud_detection.explain import valid_thr_rows

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    models_dir = config.paths.models_dir
    prereg = json.loads((config.paths.metrics_dir / "preregistration.json").read_text("utf-8"))
    source = PROJECT_ROOT / prereg["splits"][HEADLINE_SPLIT]["artifact"]
    expected = prereg["splits"][HEADLINE_SPLIT]["artifact_md5"]
    if file_md5(source) != expected:
        raise RuntimeError(f"{source.name} MD5 differs from the pre-registered {expected}")

    meta = build_meta(config, source, expected)
    save_artifact(source, models_dir, meta)
    model, meta = load_artifact(models_dir)

    thr = valid_thr_rows(HEADLINE_SPLIT, config)  # validation data only
    score = model.predict_proba(thr[MODEL_FEATURES])[:, 1]  # served (calibrated) probability
    profile = {
        "model_version": MODEL_VERSION,
        "source": f"valid_thr ({HEADLINE_SPLIT} split), {len(thr)} rows",
        "binning": f"{PROFILE_BINS} quantile bins on the reference; tied edges merged",
        "features": {"Amount": reference_profile(thr["Amount"]),
                     "score": reference_profile(score)},  # fmt: skip
    }
    (models_dir / PROFILE_NAME).write_text(json.dumps(profile, indent=2) + "\n", "utf-8")
    print(f"{MODEL_VERSION}: {ARTIFACT_NAME} (md5 {meta['artifact_md5']}), {META_NAME}, "
          f"{PROFILE_NAME} (Amount {profile['features']['Amount']['n_bins']} bins, "
          f"score {profile['features']['score']['n_bins']} bins)")  # fmt: skip


if __name__ == "__main__":
    main()
