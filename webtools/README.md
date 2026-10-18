# webtools – the framework as web tools

`webtools/` exposes every feature of `src/` (and every script in the repository root) as an
interactive tool on the GitHub Pages site. The Python code runs **unmodified in the browser**
(Pyodide 314.0.7 · CPython 3.14 · NumPy 2.4 · SciPy 1.18 · Matplotlib 3.10) inside a Web Worker.
Locally and in CI the same code runs on CPython.

```
site/                 static front end (ES modules, no build tool)
webtools/core.py      contract: SPEC validation, parameter coercion, result builder, JSON
webtools/charts.py    declarative chart specs (rendered in the browser with Plotly)
webtools/data.py      datasets: market data, uploads, synthetic series, crisis scenario
webtools/worker_api.py  the functions the Web Worker calls
webtools/tools/*.py   one module per tool
scripts/build_site.py builds _site/ (bundle, catalogue, gallery)
scripts/fetch_market_data.py  FRED → site/data/market.json
```

## Writing a tool

File `webtools/tools/<id_with_underscores>.py`:

```python
from .. import charts as C
from ..data import index_axis

SPEC = {
    "id": "risk-metrics",               # = file name with '-' instead of '_'
    "title": "VaR and Expected Shortfall",
    "group": "risk",                    # data | risk | physics | regime | warning | surface | dashboard | scripts
    "order": 20,                        # position within the group (optional)
    "summary": "One sentence on what the user learns.",
    "description": "Paragraphs separated by a blank line. **bold**, *italic* and `code` are allowed.",
    "needs": ["returns"],               # [] = simulation; otherwise returns / prices / vix / vix3m / dates
    "params": [
        {"name": "confidence", "type": "float", "default": 0.99, "min": 0.9, "max": 0.999,
         "step": 0.001, "label": "Confidence level", "help": "optional"},
        {"name": "method", "type": "select", "default": "hill", "label": "Estimator",
         "options": [{"value": "hill", "label": "Hill"}, {"value": "pickands", "label": "Pickands"}]},
        {"name": "show_fit", "type": "bool", "default": True, "label": "Show fit",
         "show_if": {"method": ["hill"]}},     # optional: only visible when method == "hill"
    ],
    "covers": ["src.models.risk_metrics.TailRiskMetrics.var_historical", "..."],
    "cost": "fast",                     # fast (< 2 s, live update) | medium | slow (> 20 s)
    "preload": "import pandas",         # optional: packages Pyodide must load first
}

def run(params, data, out):
    r = data.returns                    # np.ndarray, daily log returns (decimal)
    out.metric("99% VaR", 0.023, "pct", hint="…", status=None, badge="")  # badge: custom badge text
    out.table("Title", ["Column", "Value"], [["a", 0.1]], formats=["text", "pct:1"])  # code[:decimals]
    out.chart(C.xy("Title", [C.trace("Series", x, y)], x=C.axis("x"), y=C.axis("y", fmt="pct")))
    out.figure(fig, "Caption")          # Matplotlib figure from src/ → PNG
    out.note("Note", "info")            # info | warning | success
    out.set_verdict("serious", "Short assessment", "One sentence of reasoning.")
```

`params` arrive typed and clamped to `[min, max]`. `data` is a `webtools.core.Dataset`
(`returns`, `prices`, `vix`, `vix3m`, `dates`, `n`, `name`, `meta`). `data.require(...)` is checked
against `needs` before `run` is called.

### Rules

* **Call the original code**, do not re-implement it. Web tools are thin adapters. Bugs in the
  original are fixed in `src/`, not worked around in the adapter.
* `covers` lists full paths (`src.package.module.Symbol[.method]`; scripts as `script:<file>.py`).
  `tests/test_feature_coverage.py` enforces that every public symbol in `src/` is covered by at
  least one tool.
* Runtime with default parameters (CPython): fast < 2 s, medium < 8 s, slow < 40 s. Pyodide is
  2–4× slower. Larger simulations belong behind parameters, not in the defaults.
* No file writes, no `plt.show()`, no network. Randomness always via a `seed` parameter.
* `ToolError("…")` for errors the user can fix (an actionable message in the site's language).
* **Charts**: one y-axis per chart (never dual axes). Colours by role: `color=1..8` for series in a
  fixed order, `"muted"` for references (e.g. the Gaussian curve), status colours
  `good/warning/serious/critical` only when the colour *means* a risk state. Percentage axes use
  `fmt="pct"` (values as decimal fractions). Long series are thinned automatically, preserving
  minima and maxima. Fans and confidence bands with `C.envelope(name, x, lower, upper)`, reference
  lines with `C.hline` / `C.vline` (labels avoid each other automatically).
* `set_verdict` only when the tool delivers a genuine assessment.
* Parameters that only apply to one view or switch get `"show_if": {"param": [values]}`; they are
  shown only when the condition holds (no "only for …" remarks in help texts).
* Descriptions, notes, chart subtitles, table notes, metric hints and verdicts may contain **bold**,
  *italic* and `code`.
* Matplotlib figures from `src/visualization` are welcome (`out.figure`); interactive Plotly specs
  (`out.chart`) are the primary presentation.

## Language and style

The site is written in **Business English with British spelling** (analyse, modelling, colour,
behaviour, optimise, centre, programme). Write for risk managers, treasury and controlling teams:

* **Concise and factual.** Short sentences, active voice, no hype, no exclamation marks.
  Address the reader as "you" where an instruction is needed; otherwise neutral statements.
* **Titles** are short noun phrases in sentence case ("Extreme value theory", "Rolling risk
  metrics"); no questions, no trailing full stops. **Summaries** are one sentence.
* **Parameter labels** are noun phrases in sentence case ("Confidence level", "Window (trading
  days)"); units in the label or in `unit`.
* **Numbers**: decimal point, thousands comma (1,234.56); percentages without a space (2.5%);
  dates as `15 Oct 2026` in prose (`webtools.data.fmt_date`). Formatting of values is done by the
  front end — pass raw numbers. Units follow the same typography: `unit="%"` and `"σ"` attach to
  the number, `"$bn"` wraps it ($4.2bn), word units get the singular for exactly one (1 trading day).
* **Labels**: confidence level first ("99% VaR · historical", "97.5% Expected Shortfall"); "vs"
  without a full stop; "and" rather than "&" in titles; single quotes in prose (‘Original scripts’).
  Negative numbers in text use the minus sign (−2.5%), as the front end does.
* Keep established terms in English finance usage (see glossary); do not invent synonyms.

| German | English (use this) |
|---|---|
| Werkzeug | tool |
| Datensatz | dataset |
| Rendite / Tagesrendite | return / daily return |
| Kurs / Schlusskurs | price / closing price |
| Verlust / Tagesverlust | loss / one-day loss |
| Kennzahlen | metrics / key figures |
| Fette Ränder / Tails | fat tails |
| Rand / Randbereich | tail |
| Normalverteilung / Gauß | normal distribution / Gaussian |
| Tail-Index | tail index |
| Schwelle | threshold |
| Konfidenzniveau | confidence level |
| Zeitraum / Fenster | period / window |
| Handelstage | trading days |
| annualisiert / p. a. | annualised / p.a. |
| Volatilität | volatility |
| Einschätzung / Urteil | assessment |
| Frühwarnsystem / Frühwarnsignal | early-warning system / early-warning signal |
| Laufzeitstruktur (VIX) | term structure |
| Krise / Crash / Einbruch | crisis / crash / sell-off |
| Erholung | recovery |
| Wiederkehrperiode / -niveau | return period / return level |
| Überschreitung | exceedance |
| Stichprobe / Ziehung | sample / draw |
| Simulation / Pfad | simulation / path |
| Phasenübergang | phase transition |
| Regime / Regimewechsel | regime / regime switch |
| Absicherung | hedge / hedging |
| Hinweis | note |
| Normal / Erhöht / Hoch / Kritisch (risk levels) | Normal / Elevated / High / Critical |

## Testing and building locally

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # every tool on real S&P 500 data
python scripts/fetch_market_data.py       # refresh the FRED snapshot
python scripts/build_site.py              # → _site/
python -m http.server -d _site 8000       # http://localhost:8000
```
