"""SyntheticMarketGenerator: optional common market correlation; default output unchanged."""

import numpy as np
import pytest

from src.utils.data_loader import SyntheticMarketGenerator


def test_default_output_is_unchanged_by_the_new_option():
    a = SyntheticMarketGenerator(n_assets=6, seed=11).generate(300)
    b = SyntheticMarketGenerator(n_assets=6, seed=11, base_correlation=0.0).generate(300)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("rho", [0.3, 0.6])
def test_base_correlation_raises_pair_correlations(rho):
    m = SyntheticMarketGenerator(n_assets=20, seed=3, base_correlation=rho).generate(4000)
    c = m["correlations"]
    iu = np.triu_indices(20, 1)
    assert np.allclose(np.diag(c), 1.0)
    assert np.all(np.linalg.eigvalsh(c) > 0)          # still a valid correlation matrix
    assert c[iu].mean() == pytest.approx(rho, abs=0.08)
    emp = np.corrcoef(m["returns"].T)[iu].mean()
    assert emp == pytest.approx(c[iu].mean(), abs=0.08)  # the returns really carry it


def test_base_correlation_keeps_the_random_draws():
    a = SyntheticMarketGenerator(n_assets=5, seed=2).generate(200)
    b = SyntheticMarketGenerator(n_assets=5, seed=2, base_correlation=0.4).generate(200)
    np.testing.assert_array_equal(a["volatilities"], b["volatilities"])
    np.testing.assert_array_equal(a["drifts"], b["drifts"])


@pytest.mark.parametrize("rho", [-0.1, 1.0])
def test_invalid_base_correlation(rho):
    with pytest.raises(ValueError):
        SyntheticMarketGenerator(n_assets=3, seed=1, base_correlation=rho)
