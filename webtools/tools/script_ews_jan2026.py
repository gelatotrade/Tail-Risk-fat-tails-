"""Frühwarnanalyse Januar 2026 – runs `generate_ews_jan2026.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-ews-jan2026",
    "title": "Frühwarnanalyse Januar 2026",
    "group": "scripts",
    "order": 43,
    "summary": "Die Frühwarnanalyse zum Jahresbeginn 2026 mit allen fünf Indikatoren und dem "
               "Komposit-Score.",
    "description": (
        "Führt `generate_ews_jan2026.py` aus: Stand der Frühwarnindikatoren am 3. Januar 2026 im Vergleich zu den "
        "Krisen – die Grundlage des Abschnitts „Current Market Status“ im README."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_ews_jan2026.py"],
    "cost": "medium",
}


def run(params, data, out):
    script_tool("generate_ews_jan2026.py", params, out)
