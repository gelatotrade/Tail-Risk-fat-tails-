"""Early-warning system: all figures – runs `generate_early_warning_figures.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-early-warning-figures",
    "title": "Early-warning system: all figures",
    "group": "scripts",
    "order": 42,
    "summary": "Generates the figures of the early-warning system: indicators, warning matrix, timeline, 3D phase "
               "space, dashboard.",
    "description": (
        "Runs `generate_early_warning_figures.py` and produces all early-warning system figures from the README: "
        "the five indicators, the warning matrix, the signal timeline ahead of the crashes, the 3D phase space and "
        "the demo dashboard."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_early_warning_figures.py"],
    "cost": "slow",
}


def run(params, data, out):
    script_tool("generate_early_warning_figures.py", params, out)
