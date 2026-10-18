"""main.py --mode analyze on the active dataset (the CLI's own analysis path)."""

import numpy as np

from ..scriptrunner import QUALITY_PARAM, script_tool

SPEC = {
    "id": "script-main-analyze",
    "title": "main.py – analysis of your data",
    "group": "scripts",
    "order": 11,
    "summary": "Runs the command-line analysis mode on your dataset: full dashboard plus text report.",
    "description": (
        "Equivalent to `python main.py --mode analyze --data returns.csv`: the active dataset is saved to the script's "
        "working directory as a CSV file with a single `returns` column, then the original script "
        "runs.\n\n"
        "The result is the framework's full tail-risk dashboard and the summary report – a convenient way to "
        "analyse your own Excel data comprehensively in a single step."
    ),
    "needs": ["returns"],
    "params": [QUALITY_PARAM],
    "covers": ["script:main.py"],
    "cost": "slow",
    "preload": "import pandas",
}


def run(params, data, out):
    csv = "returns\n" + "\n".join(f"{x:.10g}" for x in np.asarray(data.returns, dtype=float))
    script_tool("main.py", params, out, argv=["--mode", "analyze", "--data", "returns.csv", "--output", "outputs"],
                files={"returns.csv": csv})
