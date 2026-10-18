"""3D-Tail-Risk-Fläche (Original-Grafik) – runs `generate_tail_risk_surface.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-tail-risk-surface",
    "title": "3D-Tail-Risk-Fläche (Original-Grafik)",
    "group": "scripts",
    "order": 30,
    "summary": "Die Hauptgrafik des Frameworks: Crash-Wahrscheinlichkeit über Volatilität und Tail-Index mit Krisenpfaden.",
    "description": (
        "Führt `generate_tail_risk_surface.py` aus und erzeugt die große 3D-Risikofläche mit Draufsicht, "
        "Konturen der Crash-Wahrscheinlichkeit, den Krisenverläufen von COVID-19, 2022 und 2025 sowie der Marktlage.\n\n"
        "Eine interaktive, drehbare Variante findest du unter „3D-Risikoflächen“."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_tail_risk_surface.py"],
    "cost": "medium",
}


def run(params, data, out):
    script_tool("generate_tail_risk_surface.py", params, out)
