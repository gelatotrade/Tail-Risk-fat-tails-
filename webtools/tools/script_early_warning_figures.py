"""Frühwarnsystem: alle Abbildungen – runs `generate_early_warning_figures.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-early-warning-figures",
    "title": "Frühwarnsystem: alle Abbildungen",
    "group": "scripts",
    "order": 42,
    "summary": "Erzeugt die Abbildungen des Frühwarnsystems: Indikatoren, Warnmatrix, Zeitachse, 3D-Phasenraum, Dashboard.",
    "description": (
        "Führt `generate_early_warning_figures.py` aus und erzeugt alle Grafiken zum Frühwarnsystem aus dem README: "
        "die fünf Indikatoren, die Warnmatrix, die Signal-Zeitachse vor den Crashs, den 3D-Phasenraum und das "
        "Demo-Dashboard."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_early_warning_figures.py"],
    "cost": "slow",
}


def run(params, data, out):
    script_tool("generate_early_warning_figures.py", params, out)
