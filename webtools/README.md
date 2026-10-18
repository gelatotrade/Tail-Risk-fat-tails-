# webtools – das Framework als Web-Werkzeuge

`webtools/` macht jedes Feature aus `src/` (und jedes Skript im Repo-Wurzelverzeichnis) als
interaktives Werkzeug auf der GitHub-Pages-Seite nutzbar. Der Python-Code läuft dort
**unverändert im Browser** (Pyodide 314.0.7 · CPython 3.14 · NumPy 2.4 · SciPy 1.18 ·
Matplotlib 3.10) in einem Web Worker. Lokal und in CI läuft derselbe Code in CPython.

```
site/                 statisches Frontend (ES-Module, kein Build-Tool)
webtools/core.py      Vertrag: SPEC-Validierung, Parameter-Coercion, Result-Builder, JSON
webtools/charts.py    deklarative Diagramm-Specs (vom Browser mit Plotly gerendert)
webtools/data.py      Datensätze: Marktdaten, Upload, synthetisch, Krisenszenario
webtools/worker_api.py  die Funktionen, die der Web Worker aufruft
webtools/tools/*.py   ein Modul pro Werkzeug
scripts/build_site.py baut _site/ (Bundle, Katalog, Galerie)
scripts/fetch_market_data.py  FRED → site/data/market.json
```

## Ein Werkzeug schreiben

Datei `webtools/tools/<id_mit_unterstrichen>.py`:

```python
from .. import charts as C
from ..data import index_axis

SPEC = {
    "id": "risk-metrics",               # = Dateiname mit '-' statt '_'
    "title": "VaR & Expected Shortfall",
    "group": "risk",                    # data | risk | physics | regime | warning | surface | dashboard | scripts
    "order": 20,                        # Reihenfolge innerhalb der Gruppe (optional)
    "summary": "Ein Satz, was man lernt.",
    "description": "Absätze durch Leerzeile getrennt. **fett** und `code` erlaubt.",
    "needs": ["returns"],               # [] = Simulation; sonst returns / prices / vix / vix3m / dates
    "params": [
        {"name": "confidence", "type": "float", "default": 0.99, "min": 0.9, "max": 0.999,
         "step": 0.001, "label": "Konfidenzniveau", "help": "optional"},
        {"name": "method", "type": "select", "default": "hill", "label": "Schätzer",
         "options": [{"value": "hill", "label": "Hill"}, {"value": "pickands", "label": "Pickands"}]},
        {"name": "show_fit", "type": "bool", "default": True, "label": "Fit einblenden"},
    ],
    "covers": ["src.models.risk_metrics.TailRiskMetrics.var_historical", "..."],
    "cost": "fast",                     # fast (< 2 s, Live-Update) | medium | slow (> 20 s)
    "preload": "import pandas",         # optional: Pakete, die Pyodide vorher laden muss
}

def run(params, data, out):
    r = data.returns                    # np.ndarray, tägliche Log-Renditen (dezimal)
    out.metric("VaR 99 %", 0.023, "pct", hint="…", status=None)
    out.table("Titel", ["Spalte", "Wert"], [["a", 0.1]], formats=["text", "pct"])
    out.chart(C.xy("Titel", [C.trace("Reihe", x, y)], x=C.axis("x"), y=C.axis("y", fmt="pct")))
    out.figure(fig, "Bildunterschrift")  # Matplotlib-Figur aus src/ → PNG
    out.note("Hinweis", "info")          # info | warning | success
    out.set_verdict("serious", "Kurz-Einschätzung", "Ein Satz Begründung.")
```

`params` kommen bereits typisiert und auf `[min, max]` begrenzt an. `data` ist ein
`webtools.core.Dataset` (`returns`, `prices`, `vix`, `vix3m`, `dates`, `n`, `name`, `meta`).
`data.require(...)` wird vor `run` anhand von `needs` geprüft.

### Regeln

* **Oberfläche auf Deutsch**, Du-Form, Fachbegriffe wie VaR, Expected Shortfall, GEX bleiben.
* **Den Original-Code aufrufen**, nicht nachbauen. Webtools sind dünne Adapter. Fehler im
  Original werden in `src/` behoben, nicht im Adapter umschifft.
* `covers` listet vollständige Pfade (`src.paket.modul.Symbol[.methode]`; Skripte als
  `script:<datei>.py`). `tests/test_feature_coverage.py` erzwingt, dass jedes öffentliche Symbol
  in `src/` von mindestens einem Werkzeug abgedeckt ist.
* Laufzeit mit Standardparametern (CPython): fast < 2 s, medium < 8 s, slow < 40 s. Pyodide ist
  2–4× langsamer. Größere Simulationen gehören hinter Parameter, nicht in die Defaults.
* Keine Dateien schreiben, kein `plt.show()`, kein Netzwerk. Zufall immer über einen `seed`-Parameter.
* `ToolError("…")` für Fehler, die der Nutzer beheben kann (deutsche, handlungsleitende Meldung).
* **Diagramme**: eine y-Achse pro Diagramm (nie Doppelachsen). Farben nach Rolle:
  `color=1..8` für Reihen in fester Reihenfolge, `"muted"` für Referenzen (z. B. die Gauß-Kurve),
  Statusfarben `good/warning/serious/critical` nur, wenn die Farbe einen Risikozustand *bedeutet*.
  Prozentachsen mit `fmt="pct"` (Werte als Dezimalbruch). Lange Reihen werden automatisch
  min/max-erhaltend ausgedünnt.
* `set_verdict` nur, wenn das Werkzeug eine echte Einschätzung liefert.
* Matplotlib-Figuren aus `src/visualization` sind willkommen (`out.figure`), interaktive
  Plotly-Specs (`out.chart`) sind die Hauptdarstellung.

## Lokal testen und bauen

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # alle Werkzeuge mit echten S&P-500-Daten
python scripts/fetch_market_data.py       # FRED-Snapshot aktualisieren
python scripts/build_site.py              # → _site/
python -m http.server -d _site 8000       # http://localhost:8000
```
