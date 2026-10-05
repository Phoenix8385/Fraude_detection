"""Export the API's OpenAPI schema to the committed contract snapshot.

tests/test_openapi_contract.py fails whenever the live schema differs from
tests/contracts/openapi.json. If a change is INTENDED, regenerate the snapshot with this script,
review the diff, and commit it together with the API change (docs/rules.md rule 19: breaking
changes need a new /vN prefix).

The schema does not depend on settings, environment, the model or time, so the output is
deterministic (keys sorted, 2-space indent, trailing newline).

    python scripts/export_openapi.py            # write the snapshot
    python scripts/export_openapi.py --check    # exit 1 if the snapshot is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # project root: import api

from api.main import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402

SNAPSHOT = ROOT / "tests" / "contracts" / "openapi.json"


def current_schema() -> str:
    """Canonical JSON text of the live schema (no model is loaded to build it)."""
    app = create_app(Settings(models_dir=ROOT / "nonexistent-models-dir"))
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export / check the OpenAPI snapshot.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = current_schema()
    if args.check:
        # compare parsed JSON: git may convert line endings of the committed file
        same = SNAPSHOT.exists() and json.loads(SNAPSHOT.read_text("utf-8")) == json.loads(text)
        print("snapshot up to date" if same else "snapshot OUT OF DATE")
        raise SystemExit(0 if same else 1)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {SNAPSHOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
