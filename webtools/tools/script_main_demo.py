"""main.py --mode demo: the framework's complete demonstration, unmodified, in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-main-demo",
    "title": "main.py – komplette Demonstration",
    "group": "scripts",
    "order": 10,
    "summary": "Führt den Einstiegspunkt des Frameworks aus: Verteilungen fitten, Risikokennzahlen, EVT, Phasenraum und Dashboards.",
    "description": (
        "Das ist exakt `python main.py --mode demo` – der Rundgang durch das ganze Framework mit synthetischen "
        "Daten: Lévy-, Tsallis- und Hill-Schätzung, VaR nach vier Methoden, Extremwerttheorie, Frühwarnsignale, "
        "3D-Phasenräume und zwei vollständige Dashboards (normaler Markt und Krisenszenario).\n\n"
        "Die Konsolenausgabe des Skripts erscheint unten, die erzeugten Grafiken und der Textbericht ebenfalls. "
        "Rechenintensiv: Im Browser dauert ein Durchlauf rund eine bis drei Minuten."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:main.py"],
    "cost": "slow",
}


def run(params, data, out):
    script_tool("main.py", params, out, argv=["--mode", "demo", "--output", "outputs"])
