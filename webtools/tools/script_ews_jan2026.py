"""Early-warning analysis January 2026 – runs `generate_ews_jan2026.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-ews-jan2026",
    "title": "Early-warning analysis, January 2026",
    "group": "scripts",
    "order": 43,
    "summary": "The early-warning analysis at the start of 2026 with all five indicators and the "
               "composite score.",
    "description": (
        "Runs `generate_ews_jan2026.py`: the state of the early-warning indicators on 3 Jan 2026 compared with "
        "past crises – the basis of the ‘Current Market Status’ section in the README."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_ews_jan2026.py"],
    "cost": "medium",
}


def run(params, data, out):
    script_tool("generate_ews_jan2026.py", params, out)
