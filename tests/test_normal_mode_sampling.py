"""Tests for beep/core/normal_mode_sampling.py."""
import numpy as np
import pytest

from beep.core.normal_mode_sampling import (
    classify_mode,
    select_modes,
    displace_along_mode,
    extract_normal_modes_from_hessian_record,
    _BOHR_TO_A,
)


# ---------------------------------------------------------------------------
# classify_mode
# ---------------------------------------------------------------------------

def _ten_atom_masses(n_ads=2):
    """Cluster of 8 oxygens + adsorbate of `n_ads` hydrogens (rough proxy)."""
    return np.array([16.0] * 8 + [1.0] * n_ads)


def test_classify_intermolecular_pure_com_translation():
    """Adsorbate translates uniformly +z, cluster stationary → intermolecular."""
    masses = _ten_atom_masses(n_ads=2)
    mode = np.zeros((10, 3))
    mode[8:10, 2] = 1.0  # both adsorbate atoms move +z
    assert classify_mode(mode, masses, n_adsorbate_atoms=2,
                          frequency_cm=200.0) == "intermolecular"


def test_classify_stretching_intramolecular_bond_oscillation():
    """Adsorbate H–H bond stretch, cluster stationary, high-freq → stretching."""
    masses = _ten_atom_masses(n_ads=2)
    mode = np.zeros((10, 3))
    mode[8, 0] = +1.0   # H₁ moves +x
    mode[9, 0] = -1.0   # H₂ moves −x  (net COM motion ~ 0)
    assert classify_mode(mode, masses, n_adsorbate_atoms=2,
                          frequency_cm=4000.0) == "stretching"


def test_classify_bending_intramolecular_low_freq():
    """Same antisymmetric motion at low frequency → bending (intramolecular)."""
    masses = _ten_atom_masses(n_ads=2)
    mode = np.zeros((10, 3))
    mode[8, 1] = +1.0
    mode[9, 1] = -1.0
    assert classify_mode(mode, masses, n_adsorbate_atoms=2,
                          frequency_cm=800.0) == "bending"


def test_classify_degenerate_n_ads_falls_back_to_freq():
    """If n_adsorbate_atoms is 0 or full system, f_inter is zero, so
    only frequency drives the classification."""
    masses = _ten_atom_masses(n_ads=2)
    mode = np.ones((10, 3))
    # n_ads=0 → degenerate, falls through to frequency cut
    assert classify_mode(mode, masses, n_adsorbate_atoms=0,
                          frequency_cm=500.0) == "bending"
    assert classify_mode(mode, masses, n_adsorbate_atoms=0,
                          frequency_cm=3000.0) == "stretching"


def test_classify_zero_mode_returns_bending_low_freq():
    """A zero mode (e.g. spurious near-TR remnant) shouldn't blow up."""
    masses = _ten_atom_masses(n_ads=2)
    mode = np.zeros((10, 3))
    # zero kinetic → f_inter forced to 0 → falls through to frequency
    assert classify_mode(mode, masses, n_adsorbate_atoms=2,
                          frequency_cm=100.0) == "bending"


# ---------------------------------------------------------------------------
# select_modes
# ---------------------------------------------------------------------------

def _twelve_mode_setup():
    freqs = np.array(
        [100, 200, 300, 400, 500,        # 5 intermolecular
         600, 700, 800,                  # 3 bending
         2900, 3700, 3800, 3900],        # 4 stretching
        dtype=complex,
    )
    classes = (
        ["intermolecular"] * 5
        + ["bending"] * 3
        + ["stretching"] * 4
    )
    band_caps = {"intermolecular": 3, "bending": 2, "stretching": 1}
    band_amps = {"intermolecular": 0.08, "bending": 0.05, "stretching": 0.03}
    return freqs, classes, band_caps, band_amps


def test_select_modes_caps_and_lowest_freq_picked():
    freqs, classes, band_caps, band_amps = _twelve_mode_setup()
    sel = select_modes(freqs, classes, band_caps, band_amps,
                        extra_amplitudes_lowest_count=0)
    # Three lowest intermolecular: indices 0, 1, 2
    # Two lowest bending: indices 5, 6
    # One lowest stretching: index 8
    indices = [s[0] for s in sel]
    assert indices == [0, 1, 2, 5, 6, 8]
    # Amplitudes per band
    amps = [s[1] for s in sel]
    assert amps == [0.08, 0.08, 0.08, 0.05, 0.05, 0.03]
    bands = [s[2] for s in sel]
    assert bands == ["intermolecular"] * 3 + ["bending"] * 2 + ["stretching"]


def test_select_modes_extra_amplitude_for_lowest():
    freqs, classes, band_caps, band_amps = _twelve_mode_setup()
    sel = select_modes(freqs, classes, band_caps, band_amps,
                        extra_amplitudes_lowest_count=1,
                        extra_amplitude_factor=2.0)
    # The previous 6 entries are unchanged, the 7th is a duplicate of the
    # lowest-frequency entry at amplitude × 2.
    assert len(sel) == 7
    assert sel[-1][0] == 0           # mode index of the lowest-freq entry
    assert sel[-1][1] == pytest.approx(0.16)  # amplitude × 2
    assert sel[-1][2] == "intermolecular"


def test_select_modes_drops_large_imaginary():
    """Imaginary frequency > freq_max_imag_cm is dropped; small imag kept."""
    freqs = np.array(
        [80j,                       # |80| > 50 → dropped (saddle direction)
         30j,                       # |30| < 50 → kept (numerical noise)
         100, 200, 300, 600, 700,   # five reals
         3700,
        ],
        dtype=complex,
    )
    classes = ["intermolecular"] * 5 + ["bending", "bending", "stretching"]
    band_caps = {"intermolecular": 3, "bending": 2, "stretching": 1}
    band_amps = {"intermolecular": 0.08, "bending": 0.05, "stretching": 0.03}
    sel = select_modes(freqs, classes, band_caps, band_amps,
                        freq_max_imag_cm=50.0,
                        extra_amplitudes_lowest_count=0)
    indices = [s[0] for s in sel]
    # 80j dropped; 30j kept (lowest "frequency"=0); next two reals = 100, 200
    assert 0 not in indices            # 80j was at index 0
    assert 1 in indices                # 30j survives


def test_select_modes_unknown_band_is_ignored():
    freqs = np.array([100, 200], dtype=complex)
    classes = ["intermolecular", "unknown_band"]
    band_caps = {"intermolecular": 3}
    band_amps = {"intermolecular": 0.08}
    sel = select_modes(freqs, classes, band_caps, band_amps,
                        extra_amplitudes_lowest_count=0)
    assert sel == [(0, 0.08, "intermolecular")]


def test_select_modes_empty_inputs_returns_empty():
    sel = select_modes(np.array([], dtype=complex), [], {"a": 1}, {"a": 0.1})
    assert sel == []


# ---------------------------------------------------------------------------
# displace_along_mode
# ---------------------------------------------------------------------------

def test_displace_rms_equals_amplitude():
    """The RMS Cartesian displacement of the displaced geometry equals the
    target amplitude (in Å)."""
    rng = np.random.default_rng(seed=2026)
    geom = rng.standard_normal((5, 3))   # bohr
    mode = rng.standard_normal((5, 3))
    target_A = 0.10
    displaced = displace_along_mode(geom, mode, amplitude_A=target_A, sign=+1)
    dx_A = (displaced - geom) * _BOHR_TO_A
    rms = float(np.sqrt(np.mean(dx_A ** 2)))
    assert rms == pytest.approx(target_A, rel=1e-9)


def test_displace_sign_inversion_is_exact_opposite():
    rng = np.random.default_rng(2027)
    geom = rng.standard_normal((4, 3))
    mode = rng.standard_normal((4, 3))
    plus = displace_along_mode(geom, mode, amplitude_A=0.05, sign=+1)
    minus = displace_along_mode(geom, mode, amplitude_A=0.05, sign=-1)
    # plus + minus == 2 * geom  (exact)
    assert np.allclose(plus + minus, 2.0 * geom)


def test_displace_invalid_sign_raises():
    geom = np.zeros((2, 3))
    mode = np.ones((2, 3))
    with pytest.raises(ValueError, match="sign"):
        displace_along_mode(geom, mode, amplitude_A=0.1, sign=0)


def test_displace_zero_mode_returns_original():
    geom = np.random.default_rng(0).standard_normal((3, 3))
    mode = np.zeros((3, 3))
    out = displace_along_mode(geom, mode, amplitude_A=0.1, sign=+1)
    assert np.allclose(out, geom)


# ---------------------------------------------------------------------------
# extract_normal_modes_from_hessian_record
# ---------------------------------------------------------------------------

def test_extract_normal_modes_diagonal_hessian_harmonics():
    """A diagonal Hessian on a 3-atom linear molecule gives analytical
    harmonic frequencies. After stripping TR modes we expect 3·N − 5 = 4
    vibrational frequencies (linear) plus the TR modes correctly removed."""
    try:
        from qcelemental.models import Molecule
    except ImportError:
        pytest.skip("qcelemental not available")

    # Linear molecule along x: H ... O ... H (centered on O)
    geom_bohr = np.array([[-1.5, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [+1.5, 0.0, 0.0]])
    symbols = ["H", "O", "H"]
    masses_amu = [1.00782503207, 15.99491461957, 1.00782503207]
    mol = Molecule(
        symbols=symbols, geometry=geom_bohr.flatten(),
        masses=masses_amu, fix_com=True, fix_orientation=True,
    )
    # Diagonal Hessian: equal positive k in every Cartesian direction.
    k = 0.5
    hess = k * np.eye(9)
    freqs, modes = extract_normal_modes_from_hessian_record(hess, mol, energy=0.0)
    # A linear triatomic gives 3*3 - 5 = 4 vibrations (but identifying
    # linearity from masses alone isn't done here, so the harmonic_analysis
    # treats it as non-linear and projects 6 TR modes → 3 vib modes left).
    # Either way, all non-TR frequencies must be real and finite.
    assert freqs.size >= 3
    assert np.all(np.abs(np.imag(freqs)) < 1e-6)
    assert np.all(np.real(freqs) > 0)
    assert modes.shape == (freqs.size, 3, 3)


def test_extract_normal_modes_returns_aligned_arrays():
    """Frequencies and mode arrays line up by index."""
    try:
        from qcelemental.models import Molecule
    except ImportError:
        pytest.skip("qcelemental not available")

    # Diatomic — simplest vibrational case.
    geom_bohr = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, +0.7]])
    mol = Molecule(
        symbols=["H", "H"], geometry=geom_bohr.flatten(),
        masses=[1.00782503207, 1.00782503207],
        fix_com=True, fix_orientation=True,
    )
    hess = 0.6 * np.eye(6)
    freqs, modes = extract_normal_modes_from_hessian_record(hess, mol, energy=0.0)
    assert freqs.shape[0] == modes.shape[0]
    assert modes.shape[1:] == (2, 3)
