"""The TEST split is loaded only by evaluate.py (Phase 9) and written only by splits.py.

Static check over src/: any string literal naming the test part ("test", "*test.parquet")
or any direct parquet read is only allowed in the modules below. Everything else must go
through train.load_part, which refuses part="test".
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "fraud_detection"

ALLOWED_TEST_LITERAL = {"splits.py", "evaluate.py"}
# train.load_part is the guarded loader: it names "test" only to refuse it.
ALLOWED_FUNCTIONS = {("train.py", "load_part")}
ALLOWED_PARQUET_READ = {"evaluate.py", "train.py"}


def _names_test_split(value: str) -> bool:
    return value == "test" or value.endswith("test.parquet")


def _nodes_with_function(node: ast.AST, func: str = "<module>"):  # noqa: ANN202
    """Yield (node, name of the innermost enclosing function) for every node in the tree."""
    for child in ast.iter_child_nodes(node):
        name = child.name if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else func
        yield child, name
        yield from _nodes_with_function(child, name)


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node, func in _nodes_with_function(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            allowed = path.name in ALLOWED_TEST_LITERAL or (path.name, func) in ALLOWED_FUNCTIONS
            if _names_test_split(node.value) and not allowed:
                found.append(f"{path.name}:{node.lineno} '{node.value}' in {func}")
        if isinstance(node, ast.Attribute) and node.attr == "read_parquet":
            if path.name not in ALLOWED_PARQUET_READ:
                found.append(f"{path.name}:{node.lineno} read_parquet")
    return found


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_module_does_not_touch_test_split(path: Path) -> None:
    assert _violations(path) == []


def test_checker_catches_a_leak(tmp_path: Path) -> None:
    leak = tmp_path / "leaky.py"
    leak.write_text(
        'import pandas as pd\ndef f(d):\n    return pd.read_parquet(d / "time_test.parquet")\n',
        encoding="utf-8",
    )
    assert len(_violations(leak)) == 2


def test_load_part_refuses_test(tmp_path: Path) -> None:
    from fraud_detection.train import load_part

    with pytest.raises(PermissionError):
        load_part("time", "test", tmp_path)
