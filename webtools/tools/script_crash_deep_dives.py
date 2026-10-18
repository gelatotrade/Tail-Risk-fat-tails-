"""Crash-Tiefenanalysen mit Datumsachsen – runs `generate_crash_deep_dives.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-crash-deep-dives",
    "title": "Crash-Tiefenanalysen mit Datumsachsen",
    "group": "scripts",
    "order": 41,
    "summary": "Die vier Detailanalysen (2018, 2020, 2022, 2025) mit echten Kalenderdaten und allen Frühwarnindikatoren.",
    "description": (
        "Führt `generate_crash_deep_dives.py` aus: Volmageddon 2018, COVID-19 2020, Bärenmarkt 2022 und "
        "Zoll-Crash 2025 – jeweils Kursverlauf, Indikatoren und Komposit-Score auf einer Zeitachse, dazu der "
        "Vergleich aller vier Crashs."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_crash_deep_dives.py"],
    "cost": "slow",
    "preload": "import pandas",
}


def run(params, data, out):
    script_tool("generate_crash_deep_dives.py", params, out)
