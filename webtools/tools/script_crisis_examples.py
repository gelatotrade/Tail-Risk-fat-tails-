"""Crisis examples 2020, 2022, 2025 – runs `generate_crisis_examples.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-crisis-examples",
    "title": "Crisis examples 2020, 2022, 2025",
    "group": "scripts",
    "order": 40,
    "summary": "Generates analyses and dashboards of the three major crises from the S&P 500 data stored in the script.",
    "description": (
        "Runs `generate_crisis_examples.py`: the 2020 COVID crash, the 2022 bear market and the 2025 tariff crash, "
        "with returns, volatility, tail index and complete dashboards, plus a crisis comparison.\n\n"
        "The price paths are stored in the script; to analyse current FRED data, choose one of the crisis periods "
        "in the dataset selection of the other tools."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_crisis_examples.py"],
    "cost": "slow",
}


def run(params, data, out):
    script_tool("generate_crisis_examples.py", params, out)
