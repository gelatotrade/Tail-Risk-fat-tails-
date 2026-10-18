"""Every web tool runs with its default parameters on real and synthetic data, within budget."""

import json
import time

import pytest

from webtools import core

SPECS = core.tool_specs()
BUDGET = {"fast": 6.0, "medium": 25.0, "slow": 120.0}  # native seconds; Pyodide is ~2-4x slower


@pytest.mark.parametrize("spec", SPECS, ids=[s["id"] for s in SPECS])
def test_tool_runs_with_defaults(spec, sp500, synthetic):
    datasets = [sp500, synthetic] if spec["needs"] else [None]
    for ds in datasets:
        t0 = time.perf_counter()
        res = core.run_tool(spec["id"], {}, ds)
        elapsed = time.perf_counter() - t0
        assert res["ok"], f"{spec['id']} failed on {getattr(ds, 'name', 'no data')}: {res.get('error')}\n{res.get('traceback', '')}"
        payload = json.dumps(core.to_json(res), allow_nan=False)
        assert len(payload) < 6_000_000, f"{spec['id']} payload too large: {len(payload) / 1e6:.1f} MB"
        assert res["metrics"] or res["charts"] or res["figures"] or res["tables"], f"{spec['id']} shows nothing"
        assert elapsed < BUDGET[spec["cost"]], f"{spec['id']} took {elapsed:.1f}s (budget {BUDGET[spec['cost']]}s)"


@pytest.mark.parametrize("spec", [s for s in SPECS if s["params"]], ids=[s["id"] for s in SPECS if s["params"]])
def test_tool_survives_parameter_extremes(spec, synthetic):
    """Min and max of every numeric parameter must not crash (they may return a user-facing error)."""
    lo = {p["name"]: p["min"] for p in spec["params"] if p["type"] in ("int", "float")}
    hi = {p["name"]: p["max"] for p in spec["params"] if p["type"] in ("int", "float")}
    if spec["cost"] == "slow":
        pytest.skip("slow tool: extremes covered by defaults test")
    for params in (lo, hi):
        res = core.run_tool(spec["id"], params, synthetic if spec["needs"] else None)
        assert res["ok"] or res["kind"] == "user", f"{spec['id']} {params}: {res.get('error')}\n{res.get('traceback', '')}"
