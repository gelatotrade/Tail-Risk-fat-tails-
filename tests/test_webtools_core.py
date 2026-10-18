"""Contract tests for the browser bridge: specs, parameter coercion, data handling, serialisation."""

import json
import math

import numpy as np
import pytest

from webtools import core, data
from webtools.core import Dataset, ToolError, coerce_params, validate_spec


def test_catalogue_is_valid_and_unique():
    cat = core.catalogue()
    ids = [t["id"] for t in cat["tools"]]
    assert ids, "no tools registered"
    assert len(ids) == len(set(ids))
    for t in cat["tools"]:
        validate_spec(t, t["id"].replace("-", "_"))
        assert t["summary"].strip() and t["title"].strip()
        assert t["covers"], f"{t['id']} covers nothing"
    json.dumps(cat, allow_nan=False)


SPEC = {
    "id": "x", "title": "x", "group": "data", "summary": "x", "needs": [], "covers": ["a"], "cost": "fast",
    "params": [
        {"name": "n", "type": "int", "default": 10, "min": 1, "max": 100, "label": "n"},
        {"name": "c", "type": "float", "default": 0.99, "min": 0.9, "max": 0.999, "label": "c"},
        {"name": "m", "type": "select", "default": "hill", "label": "m",
         "options": [{"value": "hill", "label": "Hill"}, {"value": 2, "label": "zwei"}]},
        {"name": "b", "type": "bool", "default": False, "label": "b"},
    ],
}


def test_coerce_params_clamps_and_types():
    p = coerce_params(SPEC, {"n": "1e6", "c": "nan", "m": "2", "b": "true", "junk": 1})
    assert p == {"n": 100, "c": 0.99, "m": 2, "b": True}
    assert coerce_params(SPEC, {"n": -5, "m": "evil"})["n"] == 1
    assert coerce_params(SPEC, {"m": "evil"})["m"] == "hill"
    assert coerce_params(SPEC, None) == {"n": 10, "c": 0.99, "m": "hill", "b": False}


def test_validate_spec_rejects_bad_defaults():
    bad = json.loads(json.dumps(SPEC))
    bad["params"][0]["default"] = 1000
    with pytest.raises(ValueError):
        validate_spec(bad)


def test_to_json_is_strict():
    obj = {"a": np.float64("nan"), "b": np.array([1.0, np.inf, 2.0]), "c": np.int64(3), "d": [np.bool_(True)],
           "e": np.array([[1.0, np.nan]])}
    out = core.to_json(obj)
    assert out == {"a": None, "b": [1.0, None, 2.0], "c": 3, "d": [True], "e": [[1.0, None]]}
    json.dumps(out, allow_nan=False)


def test_prices_become_aligned_log_returns():
    prices = 100 * np.exp(np.cumsum(np.full(100, 0.01)))
    dates = [f"2024-01-{i:02d}" if i < 32 else f"2024-02-{i - 31:02d}" for i in range(1, 101)]
    vix = np.arange(100, dtype=float)
    ds = data.from_arrays("t", prices=prices, dates=dates, vix=vix)
    assert ds.n == 99
    assert np.allclose(ds.returns, 0.01)
    assert ds.dates[0] == dates[1] and ds.vix[0] == 1.0  # a return belongs to the later price


def test_percent_returns_and_gaps():
    r = np.full(80, 1.0)  # +1 % per day
    r[5] = np.nan
    ds = data.from_arrays("t", returns=r, returns_unit="percent")
    assert ds.n == 79
    assert math.isclose(ds.returns[0], math.log(1.01))


def test_too_short_or_implausible_series_is_a_user_error():
    with pytest.raises(ToolError):
        data.from_arrays("t", returns=np.zeros(10))
    with pytest.raises(ToolError):
        data.from_arrays("t", returns=np.full(100, 250.0))  # prices mistaken for returns


def test_vix_with_gaps_is_dropped_not_misaligned():
    prices = np.linspace(100, 120, 200)
    vix = np.full(200, 20.0)
    vix[:60] = np.nan  # VIX3M-style late start: too many gaps -> series dropped entirely
    ds = data.from_arrays("t", prices=prices, vix=vix)
    assert ds.vix is None and ds.n == 199


def test_dataset_require_message_is_actionable():
    with pytest.raises(ToolError, match=r"does not contain: VIX\. Choose a dataset"):
        Dataset(returns=np.zeros(100)).require("returns", "vix")


def test_worker_api_round_trip(synthetic):
    from webtools import worker_api

    res = json.loads(worker_api.set_dataset("k", json.dumps({"source": "synthetic", "model": "garch", "n": 300, "seed": 1})))
    assert res["ok"] and res["summary"]["n"] == 300
    first = core.tool_specs()[0]["id"]
    out = json.loads(worker_api.run(first, "{}"))
    assert "ok" in out
    bad = json.loads(worker_api.set_dataset("k2", json.dumps({"source": "nope"})))
    assert not bad["ok"] and bad["kind"] == "user"


def test_unknown_tool_is_reported_not_raised():
    assert core.run_tool("does-not-exist")["ok"] is False


def test_show_if_must_reference_an_existing_param():
    spec = json.loads(json.dumps(SPEC))
    spec["params"][3]["show_if"] = {"m": ["hill"]}
    validate_spec(spec)
    spec["params"][3]["show_if"] = {"nope": ["hill"]}
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_table_formats_accept_digits_and_reject_unknown_codes():
    out = core.Result()
    out.table("t", ["a", "b"], [["x", 0.5]], formats=["text", "pct:0"])
    with pytest.raises(AssertionError):
        out.table("t", ["a"], [[1]], formats=["percent"])
