"""Animierte 3D-Risikofläche (GIF) – runs `generate_tail_risk_surface_animation.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-surface-animation",
    "title": "Animierte 3D-Risikofläche (GIF)",
    "group": "scripts",
    "order": 31,
    "summary": "Rendert die Animation der Risikofläche Bild für Bild als GIF – direkt im Browser.",
    "description": (
        "Führt `generate_tail_risk_surface_animation.py` aus: Die Crash-Schwelle wandert von 3 % auf 10 %, ein "
        "synthetischer Marktpfad läuft vom ruhigen Regime in die Krise und zurück, die Kamera dreht sich.\n\n"
        "Sehr rechenintensiv, weil jedes Einzelbild gerendert wird. In Vorschauqualität entstehen die Bilder "
        "mit geringerer Auflösung und deutlich schneller."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_tail_risk_surface_animation.py"],
    "cost": "slow",
}


def run(params, data, out):
    script_tool("generate_tail_risk_surface_animation.py", params, out)
