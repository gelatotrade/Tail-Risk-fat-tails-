"""main.py --mode analyze on the active dataset (the CLI's own analysis path)."""

import numpy as np

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-main-analyze",
    "title": "main.py – Analyse deiner Daten",
    "group": "scripts",
    "order": 11,
    "summary": "Die Analyse-Funktion der Kommandozeile auf deinem Datensatz: vollständiges Dashboard plus Textbericht.",
    "description": (
        "Entspricht `python main.py --mode analyze --data renditen.csv`: Der aktive Datensatz wird als CSV mit "
        "einer Spalte `returns` in das Arbeitsverzeichnis des Skripts gelegt, dann läuft das Original-Skript.\n\n"
        "Ergebnis ist das große Tail-Risk-Dashboard des Frameworks und der Zusammenfassungsbericht – ideal, um "
        "eigene Excel-Daten in einem Schritt komplett durchleuchten zu lassen."
    ),
    "needs": ["returns"],
    "params": [QUALITY_PARAM],
    "covers": ["script:main.py"],
    "cost": "slow",
    "preload": "import pandas",
}


def run(params, data, out):
    csv = "returns\n" + "\n".join(f"{x:.10g}" for x in np.asarray(data.returns, dtype=float))
    script_tool("main.py", params, out, argv=["--mode", "analyze", "--data", "returns.csv", "--output", "outputs"],
                files={"returns.csv": csv})
