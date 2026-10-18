"""main.py --mode demo: the framework's complete demonstration, unmodified, in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-main-demo",
    "title": "main.py – full demonstration",
    "group": "scripts",
    "order": 10,
    "summary": "Runs the framework's entry point: distribution fitting, risk metrics, EVT, phase space and dashboards.",
    "description": (
        "This is exactly `python main.py --mode demo` – a tour of the entire framework on synthetic data: Lévy, "
        "Tsallis and Hill estimation, VaR by four methods, extreme value theory, early-warning signals, 3D phase "
        "spaces and two complete dashboards (normal market and crisis scenario).\n\n"
        "The script's console output appears below, together with the figures and the text report it produces. "
        "Compute-intensive: a run takes about one to three minutes in the browser."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:main.py"],
    "cost": "slow",
}


def run(params, data, out):
    script_tool("main.py", params, out, argv=["--mode", "demo", "--output", "outputs"])
