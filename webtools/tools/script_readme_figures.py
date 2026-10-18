"""Documentation figures – runs `generate_readme_figures.py` unmodified in the browser."""

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-readme-figures",
    "title": "Documentation figures",
    "group": "scripts",
    "order": 20,
    "summary": "Generates the README's five core figures: fat tails, physics mapping, tail index, VaR comparison, "
               "early-warning signals.",
    "description": (
        "Runs `generate_readme_figures.py`. The figures are generated from synthetic data with a fixed seed and "
        "are therefore reproducible – the result should match the images in the repository.\n\n"
        "Useful for verifying that the entire framework produces identical results in the browser."
    ),
    "needs": [],
    "params": [QUALITY_PARAM],
    "covers": ["script:generate_readme_figures.py"],
    "cost": "medium",
}


def run(params, data, out):
    script_tool("generate_readme_figures.py", params, out)
