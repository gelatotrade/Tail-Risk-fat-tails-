"""Every web tool runs with its default parameters on real and synthetic data, within budget."""

import ast
import io
import json
import re
import time
import tokenize
from pathlib import Path

import pytest

from webtools import core

SPECS = core.tool_specs()
BUDGET = {"fast": 6.0, "medium": 25.0, "slow": 120.0}  # native seconds; Pyodide is ~2-4x slower

# The site is in English (webtools/README.md, "Language and style"): no German words, no German
# number formats such as "2,5 %" or "99 %".
GERMAN = re.compile(r"[äöüÄÖÜß]|\b(und|oder|nicht|mit|für|wird|werden|ist|sind|der|die|das|des|dem|den|ein|eine|"
                    r"einer|auf|bei|nach|über|zwischen|Tage|Jahre|Renditen?|Verlust|Schwelle|Datensatz|Werkzeug|"
                    r"Kennzahl|Hinweis|Zeitraum|Mrd|Perzentil|Fehler|Kurse?|\w*[Dd]aten)\b|\d[\s\u00a0\u202f]%|\d,\d{1,2}\s?%")


def visible_text(x):
    """Every string a user can see in a SPEC or result, without image payloads."""
    if isinstance(x, str):
        if not (len(x) > 400 and " " not in x[:400]):  # base64 figures and GIFs
            yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from visible_text(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from visible_text(v)


def assert_english(obj, where):
    found = sorted({m.group(0) for s in visible_text(obj) for m in GERMAN.finditer(s)})
    assert not found, f"{where}: German text or number format {found}"


@pytest.mark.parametrize("spec", SPECS, ids=[s["id"] for s in SPECS])
def test_spec_is_english(spec):
    assert_english(spec, spec["id"])


def _string_pieces(source):
    """(first line, last line, text at the start, text at the end) of every string literal."""
    pieces, fstring = [], None
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type == tokenize.STRING:
            try:
                text = ast.literal_eval(t.string)
            except ValueError:  # f-string on Python < 3.12
                text = re.sub(r"^[^\"']*(\"\"\"|'\'\'|\"|')", "", t.string)[:-1]
            text = text if isinstance(text, str) else ""
            pieces.append((t.start[0], t.end[0], text, text))
        elif t.type == getattr(tokenize, "FSTRING_START", None):
            fstring = [t.start[0], None, None, ""]
        elif t.type == getattr(tokenize, "FSTRING_MIDDLE", None) and fstring:
            fstring[2] = t.string if fstring[2] is None else fstring[2]
            fstring[3] = t.string
        elif t.type == getattr(tokenize, "FSTRING_END", None) and fstring:
            pieces.append((fstring[0], t.end[0], fstring[2] or "", fstring[3]))
            fstring = None
        elif t.type == tokenize.OP and fstring and t.string == "}":
            fstring[3] = "}"  # an f-string ending in a replacement field
        elif t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT) and not fstring:
            pieces.append(None)  # anything else breaks a run of adjacent literals
    return pieces


def test_no_words_glued_by_string_concatenation():
    """Strings continued on the next line need a space: "... of days" "lie above" would read "dayslie"."""
    glued = []
    for f in sorted((Path(__file__).resolve().parents[1] / "webtools").rglob("*.py")):
        pieces = _string_pieces(f.read_text(encoding="utf-8"))
        for a, b in zip(pieces, pieces[1:]):
            if a and b and b[0] > a[1] and re.search(r"[\w.,;:)%]$", a[3]) and not a[3].endswith("\\n") \
                    and re.match(r"[\w(]", b[2]):
                glued.append(f"{f.name}:{a[1]}: ...{a[3][-25:]}|{b[2][:25]}...")
    assert not glued, "missing space between concatenated strings:\n" + "\n".join(glued)


def test_site_shell_is_english(market):
    site = Path(__file__).resolve().parents[1] / "site"
    for f in [site / "index.html", *sorted((site / "assets" / "js").glob("*.js")), site / "data" / "gallery.json"]:
        assert_english(f.read_text(encoding="utf-8"), f.relative_to(site))
    assert_english([[d["name"], d["blurb"]] for d in market["datasets"]], "market.json")


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
        assert_english(core.to_json(res), f"{spec['id']} on {getattr(ds, 'name', 'no data')}")
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
