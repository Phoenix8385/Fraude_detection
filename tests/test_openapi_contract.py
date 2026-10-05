"""OpenAPI contract snapshot (docs/evaluation.md "Additional tests"; rules.md rule 19).

The live schema must equal tests/contracts/openapi.json. Any accidental change to routes,
request/response fields, types, limits or error models fails here. For an INTENDED change:
python scripts/export_openapi.py, review the diff, commit it with the API change (and use a
new /vN prefix if it breaks clients).
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from api.main import create_app
from api.settings import Settings

SNAPSHOT = Path(__file__).parent / "contracts" / "openapi.json"


def live_schema() -> dict[str, Any]:
    return create_app(Settings(models_dir=Path("nonexistent-models-dir"))).openapi()


def contract_diff(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Human-readable list of differences (first few levels are enough to locate a change)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = [f"removed {path}.{k}" for k in expected.keys() - actual.keys()]
        out += [f"added {path}.{k}" for k in actual.keys() - expected.keys()]
        for k in expected.keys() & actual.keys():
            out += contract_diff(expected[k], actual[k], f"{path}.{k}")
        return out
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return [d for i, (e, a) in enumerate(zip(expected, actual, strict=True))
                for d in contract_diff(e, a, f"{path}[{i}]")]  # fmt: skip
    return [] if expected == actual else [f"changed {path}: {expected!r} -> {actual!r}"]


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_live_schema_matches_committed_snapshot(snapshot) -> None:  # type: ignore[no-untyped-def]
    diff = contract_diff(snapshot, live_schema())
    assert not diff, (
        "OpenAPI contract changed (run scripts/export_openapi.py only if intended):\n"
        + "\n".join(sorted(diff)[:40])
    )


def test_schema_is_deterministic() -> None:
    assert json.dumps(live_schema(), sort_keys=True) == json.dumps(live_schema(), sort_keys=True)


@pytest.mark.parametrize(
    "breaking_change",
    [
        lambda s: s["paths"].pop("/v1/explain"),  # route removed
        lambda s: s["components"]["schemas"]["PredictionResponse"]["properties"].pop(
            "risk_tier"),  # response field removed  # fmt: skip
        lambda s: s["components"]["schemas"]["PredictionResponse"]["required"].remove(
            "recommended_action"),  # field no longer required  # fmt: skip
        lambda s: s["components"]["schemas"]["BatchRequest"]["properties"]["transactions"]
        .update(maxItems=1000),  # batch limit changed
        lambda s: s["components"]["schemas"]["Transaction"].update(
            additionalProperties=True),  # extra fields would be accepted  # fmt: skip
    ],
    ids=["route_removed", "field_removed", "field_optional", "batch_limit", "extra_allowed"],
)
def test_snapshot_comparison_catches_breaking_changes(snapshot, breaking_change) -> None:  # type: ignore[no-untyped-def]
    changed = copy.deepcopy(snapshot)
    breaking_change(changed)
    assert contract_diff(snapshot, changed)


def test_snapshot_pins_the_core_contract(snapshot) -> None:  # type: ignore[no-untyped-def]
    """Spot-check the facts the rest of the system depends on, so the snapshot is not empty."""
    schemas = snapshot["components"]["schemas"]
    assert set(snapshot["paths"]) == {"/health", "/ready", "/model-info", "/v1/predict",
                                      "/v1/predict/batch", "/v1/explain"}  # fmt: skip
    assert schemas["BatchRequest"]["properties"]["transactions"]["maxItems"] == 500
    assert schemas["Transaction"]["additionalProperties"] is False
    assert set(schemas["PredictionResponse"]["required"]) == {
        "transaction_id", "fraud_probability", "risk_tier", "recommended_action", "is_flagged",
        "thresholds", "model_version", "latency_ms"}  # fmt: skip
    assert schemas["PredictionResponse"]["properties"]["risk_tier"]["enum"] == [
        "HIGH", "MEDIUM", "LOW"]  # fmt: skip
    for code in ("401", "422", "429", "503"):
        assert code in snapshot["paths"]["/v1/predict"]["post"]["responses"]
