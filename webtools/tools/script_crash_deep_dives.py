"""Crash deep dives with date axes – runs `generate_crash_deep_dives.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-crash-deep-dives",
    "title": "Crash deep dives with date axes",
    "group": "scripts",
    "order": 41,
    "summary": "The four detailed analyses (2018, 2020, 2022, 2025) with real calendar dates and all early-warning "
               "indicators.",
    "description": (
        "Runs `generate_crash_deep_dives.py`: Volmageddon 2018, the 2020 COVID crash, the 2022 bear market and the 2025 "
        "tariff crash – each with price path, indicators and composite score on a single time axis, plus a "
        "comparison of all four crashes."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_crash_deep_dives.py"],
    "cost": "slow",
    "preload": "import pandas",
}


def run(params, data, out):
    script_tool("generate_crash_deep_dives.py", params, out)
