"""Abbildungen der Dokumentation – runs `generate_readme_figures.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-readme-figures",
    "title": "Abbildungen der Dokumentation",
    "group": "scripts",
    "order": 20,
    "summary": "Erzeugt die fünf Grundlagen-Grafiken des README: fette Ränder, Physik-Mapping, Tail-Index, VaR-Vergleich, Frühwarnsignale.",
    "description": (
        "Führt `generate_readme_figures.py` aus. Die Grafiken entstehen aus synthetischen Daten mit festem Seed und "
        "sind deshalb reproduzierbar – das Ergebnis sollte den Bildern im Repository entsprechen.\n\n"
        "Gut geeignet, um zu prüfen, dass das ganze Framework im Browser identisch rechnet."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_readme_figures.py"],
    "cost": "medium",
}


def run(params, data, out):
    script_tool("generate_readme_figures.py", params, out)
