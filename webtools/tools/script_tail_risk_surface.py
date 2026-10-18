"""3D tail-risk surface (original figure) – runs `generate_tail_risk_surface.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-tail-risk-surface",
    "title": "3D tail-risk surface (original figure)",
    "group": "scripts",
    "order": 30,
    "summary": "The framework's flagship chart: crash probability across volatility and tail index, with crisis paths.",
    "description": (
        "Runs `generate_tail_risk_surface.py` and produces the large 3D risk surface with a top view, contours of "
        "the crash probability, the crisis paths of COVID-19, 2022 and 2025, and the current market "
        "position.\n\n"
        "For an interactive, rotatable version with your dataset's trajectory, use the ‘3D tail-risk surface with "
        "market trajectory’ tool."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_tail_risk_surface.py"],
    "cost": "medium",
}


def run(params, data, out):
    script_tool("generate_tail_risk_surface.py", params, out)
