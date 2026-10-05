"""Tests for fraud_detection.artifacts — temp dirs and a synthetic model only."""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_detection.artifacts import (
    ARTIFACT_NAME,
    META_NAME,
    MODEL_VERSION,
    file_md5,
    library_versions,
    load_artifact,
    reference_profile,
    save_artifact,
    schema_hash,
)
from fraud_detection.features import MODEL_FEATURES


@pytest.fixture
def packaged(tmp_path: Path) -> tuple[Path, Path]:
    """A synthetic 'frozen' model file and a models dir packaged from it."""
    source = tmp_path / "final_model_time.joblib"
    model = LogisticRegression().fit(np.array([[0.0], [1.0]]), [0, 1])
    joblib.dump(model, source)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    meta = {"model_version": MODEL_VERSION, "artifact_md5": file_md5(source),
            "features": MODEL_FEATURES, "schema_hash": schema_hash(),
            "library_versions": library_versions()}  # fmt: skip
    save_artifact(source, models_dir, meta)
    return source, models_dir


def test_artifact_is_byte_identical_copy(packaged) -> None:  # type: ignore[no-untyped-def]
    source, models_dir = packaged
    assert (models_dir / ARTIFACT_NAME).read_bytes() == source.read_bytes()
    meta = json.loads((models_dir / META_NAME).read_text(encoding="utf-8"))
    assert meta["artifact_md5"] == file_md5(source)


def test_save_refuses_when_copy_md5_differs(tmp_path: Path) -> None:
    source = tmp_path / "m.joblib"
    source.write_bytes(b"model")
    with pytest.raises(RuntimeError, match="byte-identical"):
        save_artifact(source, tmp_path, {"artifact_md5": "0" * 32})


def test_load_artifact_round_trip(packaged) -> None:  # type: ignore[no-untyped-def]
    _, models_dir = packaged
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # same environment: no version warnings
        model, meta = load_artifact(models_dir)
    assert meta["model_version"] == "logreg-iso-v1.0.0"
    assert model.predict_proba([[0.5]]).shape == (1, 2)


def test_load_artifact_detects_tampering(packaged) -> None:  # type: ignore[no-untyped-def]
    _, models_dir = packaged
    (models_dir / ARTIFACT_NAME).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="MD5"):
        load_artifact(models_dir)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [("features", MODEL_FEATURES[::-1], "feature list"), ("schema_hash", "x", "schema_hash")],
)
def test_load_artifact_rejects_contract_changes(packaged, field, value, match) -> None:  # type: ignore[no-untyped-def]
    _, models_dir = packaged
    meta_path = models_dir / META_NAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps({**meta, field: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_artifact(models_dir)


def test_load_artifact_warns_on_library_drift(packaged) -> None:  # type: ignore[no-untyped-def]
    _, models_dir = packaged
    meta_path = models_dir / META_NAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["library_versions"]["sklearn"] = "0.0.1"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.warns(UserWarning, match="sklearn"):
        load_artifact(models_dir)


def test_schema_hash_is_stable_and_sensitive_to_feature_order(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fraud_detection.artifacts as art

    h = schema_hash()
    assert h == schema_hash() and len(h) == 64
    monkeypatch.setattr(art, "MODEL_FEATURES", MODEL_FEATURES[::-1])
    assert art.schema_hash() != h


def test_reference_profile_quantile_bins() -> None:
    p = reference_profile(np.arange(1000), n_bins=10)
    assert p["n_bins"] == 10 and p["n_rows"] == 1000
    assert p["edges"][0] is None and p["edges"][-1] is None  # open outer bins
    inner = p["edges"][1:-1]
    assert inner == sorted(inner) and len(inner) == 9
    assert sum(p["proportions"]) == pytest.approx(1.0)
    assert all(q == pytest.approx(0.1) for q in p["proportions"])


def test_reference_profile_merges_tied_edges() -> None:
    # 90% of scores tied at 0 (like isotonic output): duplicate quantile edges collapse
    values = np.r_[np.zeros(900), np.linspace(0.1, 1.0, 100)]
    p = reference_profile(values, n_bins=10)
    assert p["n_bins"] < 10
    assert sum(p["proportions"]) == pytest.approx(1.0)
    assert len(p["edges"]) == p["n_bins"] + 1
    with pytest.raises(ValueError):
        reference_profile([])
