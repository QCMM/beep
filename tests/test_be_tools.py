"""Tests for beep/core/be_tools.py."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from beep.core.be_tools import gauss, gauss_fitting


# ---------------------------------------------------------------------------
# gauss (pure math)
# ---------------------------------------------------------------------------

def test_gauss_peak_at_mu():
    assert gauss(5.0, 2.0, 5.0, 1.0) == 2.0


def test_gauss_symmetry():
    mu, A, sigma = 0.0, 1.0, 1.0
    assert abs(gauss(mu - 1, A, mu, sigma) - gauss(mu + 1, A, mu, sigma)) < 1e-15


def test_gauss_known_value():
    assert gauss(0, 1, 0, 1) == 1.0


def test_gauss_array_input():
    x = np.array([0.0, 1.0, 2.0])
    result = gauss(x, 1.0, 0.0, 1.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


# ---------------------------------------------------------------------------
# gauss_fitting (bootstrap Gaussian fit)
# ---------------------------------------------------------------------------

def test_gauss_fitting_returns_three_params(test_logger):
    np.random.seed(42)
    true_mu = -5.0
    data = np.random.normal(loc=true_mu, scale=1.0, size=2000)
    p0 = [200, true_mu, 1.0]
    vbest = gauss_fitting(nbins=30, data=data, p0=p0, logger=test_logger, nboot=500)
    assert isinstance(vbest, list)
    assert len(vbest) == 3
    # All parameters should be finite numbers
    assert all(np.isfinite(v) for v in vbest)
