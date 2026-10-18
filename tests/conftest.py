import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def market():
    return json.loads((ROOT / "site" / "data" / "market.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sp500(market):
    """The real S&P 500 snapshot the website ships (prices + VIX + VIX3M)."""
    from webtools import data

    ds = next(d for d in market["datasets"] if d["id"] == "sp500")
    vix3m = [np.nan if v is None else v for v in ds["vix3m"]]
    return data.from_arrays("S&P 500", prices=ds["prices"], dates=ds["dates"], vix=ds["vix"], vix3m=vix3m)


@pytest.fixture(scope="session")
def synthetic():
    from webtools import data

    return data.synthetic("levy_jump", 1500, 7)
