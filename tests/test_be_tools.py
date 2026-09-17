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


def test_apply_lin_models_std_over_methods_only():
    """Regression: StdDev_all_dft in the linear-model frame was computed after
    the Mean column was inserted, so it ran over methods + mean. With an
    identity model (m=1, n=0) the corrected columns equal the inputs and the
    std must equal the pandas std (ddof=1) over the three method columns."""
    import pandas as pd
    from beep.core.be_tools import apply_lin_models

    be_methods = ["wb97x-v", "m06-hf", "wpbe-d3bj"]
    basis = "def2-tzvp"
    entries = ["s1", "s2"]
    df_be = pd.DataFrame({
        "wb97x-v/def2-tzvp":   [-10.0, -3.0],
        "m06-hf/def2-tzvp":    [-12.0, -5.0],
        "wpbe-d3bj/def2-tzvp": [-17.0, -4.0],
    }, index=entries)
    df_be["Mean_Eb_all_dft"] = df_be.mean(axis=1)
    out = apply_lin_models(
        df_be, df_be, {"Mean": [1.0, 0.0, 1.0]}, be_methods, basis, "CO",
        be_range=(-0.1, -25.0), generate_plots=False,
    )
    lin_cols = [f"{bm}/{basis}_lin_ZPVE" for bm in be_methods]
    assert set(lin_cols) <= set(out.columns)
    pd.testing.assert_series_equal(
        out["StdDev_all_dft"], out[lin_cols].std(axis=1, ddof=1), check_names=False)
    assert out.loc["s1", "StdDev_all_dft"] == pytest.approx(
        np.std([-10.0, -12.0, -17.0], ddof=1))
