"""'Every feature live': each public class/function in src/ is reachable from some web tool."""

import ast
from pathlib import Path

from webtools import core

ROOT = Path(__file__).resolve().parents[1]

# Public names that are deliberately not exposed as their own feature, with the reason.
INTERNAL = {
    "src.utils.numerical.safe_divide": "numerical helper used inside models",
    "src.utils.numerical.rolling_window": "numerical helper used inside models",
    "src.utils.numerical.exponential_weighted_mean": "numerical helper used inside models",
    "src.utils.numerical.numerical_gradient": "numerical helper used inside models",
    "src.utils.numerical.numerical_hessian": "numerical helper used inside models",
}


def public_symbols():
    out = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        mod = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                out.append(f"{mod}.{node.name}")
    return out


def test_every_public_symbol_is_covered_by_a_tool():
    covers = set()
    for spec in core.tool_specs():
        for c in spec["covers"]:
            covers.add(c)
            parts = c.split(".")
            for i in range(len(parts) - 1, 1, -1):  # a covered method also covers its class
                covers.add(".".join(parts[:i]))
    missing = [s for s in public_symbols() if s not in covers and s not in INTERNAL]
    assert not missing, "Public symbols without a web tool:\n  " + "\n  ".join(missing)


def test_every_script_has_a_runner():
    scripts = ["main.py"] + sorted(p.name for p in ROOT.glob("generate_*.py"))
    covers = {c for spec in core.tool_specs() for c in spec["covers"]}
    missing = [s for s in scripts if f"script:{s}" not in covers]
    assert not missing, f"Scripts without a browser runner: {missing}"
