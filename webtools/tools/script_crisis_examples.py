"""Krisenbeispiele 2020, 2022, 2025 – runs `generate_crisis_examples.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-crisis-examples",
    "title": "Krisenbeispiele 2020, 2022, 2025",
    "group": "scripts",
    "order": 40,
    "summary": "Erzeugt Analysen und Dashboards der drei großen Krisen aus den im Skript hinterlegten S&P-500-Daten.",
    "description": (
        "Führt `generate_crisis_examples.py` aus: COVID-Crash 2020, Bärenmarkt 2022 und Zoll-Crash 2025 mit "
        "Renditen, Volatilität, Tail-Index und vollständigen Dashboards sowie dem Krisenvergleich.\n\n"
        "Die Kursverläufe sind im Skript hinterlegt; für eine Analyse auf aktuellen FRED-Daten nutze die "
        "Datensatz-Auswahl mit den Krisen-Zeitfenstern in den anderen Werkzeugen."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_crisis_examples.py"],
    "cost": "slow",
}


def run(params, data, out):
    script_tool("generate_crisis_examples.py", params, out)
