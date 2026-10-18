"""Regression tests: the historical crash facts in CrashComparisonChart match the S&P 500 and VIX closes.

Reference values are daily closes (FRED series SP500 and VIXCLS), written out here so the test does not
depend on the rolling ten-year market snapshot in site/data/market.json.
"""

import os

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from src.visualization.early_warning_dashboard import CrashComparisonChart  # noqa: E402

# crash -> (S&P 500 close before, S&P 500 close after, max VIX close in the episode)
#   2018: single day 2018-02-02 -> 2018-02-05; VIX 17.31 -> 37.32 (+115.6 %, stored as a % change)
#   2020: high 2020-02-19 -> low 2020-03-23; VIX close 82.69 on 2020-03-16
#   2022: high 2022-01-03 -> low 2022-10-12; VIX close high 36.45 on 2022-03-07
#   2025: high 2025-02-19 -> low 2025-04-08; VIX close high 52.33 on 2025-04-08
REFERENCE = {
    "2018 Volmageddon": (2762.13, 2648.94, 100 * (37.32 / 17.31 - 1)),
    "2020 COVID Crash": (3386.15, 2237.40, 82.69),
    "2022 Bear Market": (4796.56, 3577.03, 36.45),
    "2025 Tariff Crash": (6144.15, 4982.77, 52.33),
}


@pytest.mark.parametrize("crash", sorted(REFERENCE))
def test_drop_matches_sp500_closes(crash):
    before, after, _ = REFERENCE[crash]
    drop = CrashComparisonChart().crash_events[crash]["drop"]
    assert drop == pytest.approx(100 * (after / before - 1), abs=0.1)


@pytest.mark.parametrize("crash", sorted(REFERENCE))
def test_vix_matches_vix_closes(crash):
    vix = CrashComparisonChart().crash_events[crash]["vix_spike"]
    assert vix == pytest.approx(REFERENCE[crash][2], abs=0.1)


def test_tariff_crash_no_longer_overstated():
    # the former value (-23.0 %) overstated the 2025 drawdown by four percentage points
    assert CrashComparisonChart().crash_events["2025 Tariff Crash"]["drop"] > -20.0
