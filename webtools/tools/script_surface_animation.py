"""Animated 3D risk surface (GIF) – runs `generate_tail_risk_surface_animation.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-surface-animation",
    "title": "Animated 3D risk surface (GIF)",
    "group": "scripts",
    "order": 31,
    "summary": "Renders the risk-surface animation frame by frame as a GIF – directly in the browser.",
    "description": (
        "Runs `generate_tail_risk_surface_animation.py`: the crash threshold moves from 3% to 10%, a synthetic "
        "market path runs from a calm regime into a crisis and back, and the camera rotates.\n\n"
        "Very compute-intensive, because every single frame is rendered. In preview quality the frames are "
        "produced at a lower resolution and considerably faster."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_tail_risk_surface_animation.py"],
    "cost": "slow",
}


def run(params, data, out):
    script_tool("generate_tail_risk_surface_animation.py", params, out)
