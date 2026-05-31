"""Tests for beep/core/trajectory_metrics.py."""
import numpy as np
import pandas as pd
import pytest

from beep.core.trajectory_metrics import (
    per_step_deltas,
    summarize_method_metrics,
    combined_zscore_ranking,
    HARTREE_TO_MEV,
    HARTREE_PER_BOHR_TO_MEV_PER_A,
)


# ---------------------------------------------------------------------------
# Conversion constants
# ---------------------------------------------------------------------------

def test_hartree_to_meV_value():
    # 1 hartree = 27.2113962 eV = 27211.4 meV (CODATA 2018, 6 sig figs)
    assert abs(HARTREE_TO_MEV - 27211.386) < 0.01


def test_hartree_per_bohr_to_meV_per_A_value():
    # 1 hartree/bohr ≈ 51422.07 meV/Å
    assert abs(HARTREE_PER_BOHR_TO_MEV_PER_A - 51422.07) < 0.1


# ---------------------------------------------------------------------------
# per_step_deltas
# ---------------------------------------------------------------------------

def test_per_step_deltas_zero_diff_is_zero():
    """Identical ref and DFT inputs → all deltas zero."""
    e = np.array([-100.0, -100.5, -101.0])
    g = np.zeros((3, 2, 3))
    d = per_step_deltas(e, g, e, g, n_atoms=2)
    assert np.all(d["delta_e_per_atom_meV"] == 0.0)
    assert np.all(d["delta_force_meV_per_A"] == 0.0)


def test_per_step_deltas_energy_per_atom_units():
    """0.001 hartree total-energy difference on a 2-atom system → ~13.6 meV/atom."""
    ref_e = np.array([-100.0, -100.5])
    dft_e = np.array([-100.001, -100.501])   # 0.001 hartree lower
    g = np.zeros((2, 2, 3))
    d = per_step_deltas(ref_e, g, dft_e, g, n_atoms=2)
    expected = -0.001 * HARTREE_TO_MEV / 2.0
    assert d["delta_e_per_atom_meV"].shape == (2,)
    assert abs(d["delta_e_per_atom_meV"][0] - expected) < 1e-6
    assert abs(d["delta_e_per_atom_meV"][1] - expected) < 1e-6


def test_per_step_deltas_force_units():
    """1e-4 hartree/bohr gradient difference → ~5.14 meV/Å per component."""
    e = np.zeros(1)
    ref_g = np.zeros((1, 1, 3))
    dft_g = np.zeros((1, 1, 3))
    dft_g[0, 0, 0] = 1e-4   # x-component only
    d = per_step_deltas(e, ref_g, e, dft_g, n_atoms=1)
    expected = 1e-4 * HARTREE_PER_BOHR_TO_MEV_PER_A
    assert d["delta_force_meV_per_A"].shape == (1, 1, 3)
    assert abs(d["delta_force_meV_per_A"][0, 0, 0] - expected) < 1e-6
    assert d["delta_force_meV_per_A"][0, 0, 1] == 0.0
    assert d["delta_force_meV_per_A"][0, 0, 2] == 0.0


def test_per_step_deltas_shape_mismatch_raises():
    with pytest.raises(ValueError, match="ref_energies shape"):
        per_step_deltas(
            np.array([1.0, 2.0]), np.zeros((2, 1, 3)),
            np.array([1.0]),       np.zeros((1, 1, 3)),
            n_atoms=1,
        )


def test_per_step_deltas_gradient_shape_mismatch_raises():
    with pytest.raises(ValueError, match="ref_gradients shape"):
        per_step_deltas(
            np.array([1.0]), np.zeros((1, 2, 3)),
            np.array([1.0]), np.zeros((1, 3, 3)),
            n_atoms=2,
        )


def test_per_step_deltas_invalid_n_atoms_raises():
    with pytest.raises(ValueError, match="n_atoms"):
        per_step_deltas(
            np.array([1.0]), np.zeros((1, 1, 3)),
            np.array([1.0]), np.zeros((1, 1, 3)),
            n_atoms=0,
        )


# ---------------------------------------------------------------------------
# summarize_method_metrics
# ---------------------------------------------------------------------------

def test_summarize_empty_returns_nans():
    s = summarize_method_metrics([])
    assert np.isnan(s["rmsd_energy"])
    assert np.isnan(s["rmsd_force"])
    assert s["n_energy_points"] == 0
    assert s["n_force_components"] == 0


def test_summarize_single_system_constant_delta():
    """A trajectory with constant +5 meV/atom delta → RMSD = 5."""
    deltas = {
        "delta_e_per_atom_meV": np.array([5.0, 5.0, 5.0]),
        "delta_force_meV_per_A": np.zeros((3, 2, 3)),
    }
    s = summarize_method_metrics([deltas])
    assert s["rmsd_energy"] == 5.0
    assert s["rmsd_force"] == 0.0
    assert s["n_energy_points"] == 3
    assert s["n_force_components"] == 18   # 3 steps × 2 atoms × 3 components


def test_summarize_signed_deltas_use_squared_under_root():
    """+5 and -5 should give RMSD = 5 (signs irrelevant under squaring)."""
    deltas = {
        "delta_e_per_atom_meV": np.array([+5.0, -5.0]),
        "delta_force_meV_per_A": np.zeros((2, 1, 3)),
    }
    s = summarize_method_metrics([deltas])
    assert s["rmsd_energy"] == 5.0


def test_summarize_concatenates_systems():
    """Deltas from two systems with different lengths get concatenated."""
    a = {
        "delta_e_per_atom_meV": np.array([2.0, 4.0]),
        "delta_force_meV_per_A": np.full((2, 1, 3), 1.0),
    }
    b = {
        "delta_e_per_atom_meV": np.array([6.0, 8.0, 10.0]),
        "delta_force_meV_per_A": np.full((3, 2, 3), 2.0),
    }
    s = summarize_method_metrics([a, b])
    assert s["n_energy_points"] == 5
    # Force components: 2*1*3 + 3*2*3 = 6 + 18 = 24
    assert s["n_force_components"] == 24
    # RMSD energy = sqrt(mean([4, 16, 36, 64, 100])) = sqrt(44)
    assert abs(s["rmsd_energy"] - np.sqrt(44.0)) < 1e-10


def test_summarize_force_rmsd_over_all_components():
    """Force RMSD is taken over every Cartesian component (n_steps × n_atoms × 3)."""
    f = np.array([[[1.0, 0.0, 0.0]], [[0.0, 3.0, 0.0]]])   # shape (2, 1, 3)
    deltas = {
        "delta_e_per_atom_meV": np.zeros(2),
        "delta_force_meV_per_A": f,
    }
    s = summarize_method_metrics([deltas])
    # RMSD = sqrt(mean([1, 0, 0, 0, 9, 0])) = sqrt(10/6)
    assert abs(s["rmsd_force"] - np.sqrt(10.0 / 6.0)) < 1e-10


# ---------------------------------------------------------------------------
# combined_zscore_ranking  (weight-driven; works for any metric subset)
# ---------------------------------------------------------------------------

def _three_method_df():
    """Standard 2-metric setup matching the geom_benchmark workflow."""
    return pd.DataFrame(
        {
            "rmsd_eq":    [0.04, 0.03, 0.05],
            "rmsd_force": [25.0, 20.0, 30.0],
        },
        index=["B3LYP-D3BJ", "MPWB1K-D3BJ", "M06-2X"],
    )


def _equal_weights():
    return {"rmsd_eq": 1.0, "rmsd_force": 1.0}


def test_combined_zscore_winner():
    """MPWB1K-D3BJ is best on both dims → must come out top."""
    ranking = combined_zscore_ranking(_three_method_df(), _equal_weights())
    assert ranking.index[0] == "MPWB1K-D3BJ"


def test_combined_zscore_columns():
    ranking = combined_zscore_ranking(_three_method_df(), _equal_weights())
    for col in ("z_rmsd_eq", "z_rmsd_force", "combined_score"):
        assert col in ranking.columns


def test_combined_zscore_sorted_ascending():
    ranking = combined_zscore_ranking(_three_method_df(), _equal_weights())
    scores = ranking["combined_score"].to_numpy()
    assert (np.diff(scores) >= 0).all(), "combined_score should be ascending"


def test_combined_zscore_zero_variance_dim_contributes_zero():
    """A metric where all methods are equal → that z-column is all 0."""
    df = pd.DataFrame(
        {
            "rmsd_eq":    [0.04, 0.04, 0.04],   # flat
            "rmsd_force": [25.0, 20.0, 30.0],
        },
        index=["A", "B", "C"],
    )
    ranking = combined_zscore_ranking(df, _equal_weights())
    assert (ranking["z_rmsd_eq"] == 0.0).all()


def test_combined_zscore_weights_change_ranking():
    """Upweighting force enough should change the winner if a method is
    bad on force but good on geometry."""
    df = pd.DataFrame(
        {
            "rmsd_eq":    [0.03, 0.04],
            "rmsd_force": [40.0, 20.0],
        },
        index=["good_geom_bad_force", "decent_balanced"],
    )
    equal = combined_zscore_ranking(df, _equal_weights())
    # Equal weights: both are mirror-symmetric on the two dims so score = 0
    # for both. Tie-break by sort stability — make the test robust by
    # checking the winner under heavier force weight.
    assert "good_geom_bad_force" in equal.index
    force_heavy = combined_zscore_ranking(
        df, {"rmsd_eq": 1.0, "rmsd_force": 10.0},
    )
    assert force_heavy.index[0] == "decent_balanced"


def test_combined_zscore_missing_metric_column_raises():
    df = pd.DataFrame(
        {"rmsd_eq": [0.04, 0.03]},
        index=["A", "B"],
    )
    with pytest.raises(ValueError, match="rmsd_force"):
        combined_zscore_ranking(df, _equal_weights())


def test_combined_zscore_empty_weights_raises():
    with pytest.raises(ValueError, match="non-empty"):
        combined_zscore_ranking(_three_method_df(), {})


def test_combined_zscore_single_method_zero_score():
    """One method → std=0 → all z-scores 0 → combined_score 0."""
    df = pd.DataFrame(
        {"rmsd_eq": [0.04], "rmsd_force": [25.0]},
        index=["A"],
    )
    ranking = combined_zscore_ranking(df, _equal_weights())
    assert ranking.loc["A", "combined_score"] == 0.0


def test_combined_zscore_handles_arbitrary_metric_set():
    """Weight-driven: any metric subset present in both df and weights works."""
    df = pd.DataFrame(
        {
            "alpha": [1.0, 2.0, 3.0],
            "beta":  [5.0, 4.0, 3.0],
            "gamma": [9.0, 9.0, 9.0],   # zero-variance, ignored
        },
        index=["x", "y", "z"],
    )
    ranking = combined_zscore_ranking(
        df, {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
    )
    for col in ("z_alpha", "z_beta", "z_gamma", "combined_score"):
        assert col in ranking.columns
    # gamma is zero-variance → contributes 0
    assert (ranking["z_gamma"] == 0.0).all()
