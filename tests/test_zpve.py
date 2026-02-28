"""Tests for beep/core/zpve.py.

suppress_stdout is pure Python and always testable.
_vibanal_wfn requires a working psi4 installation; tests are skipped
when psi4 is broken (e.g. segfault on import).
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from qcelemental.models.molecule import Molecule

from beep.core.zpve import suppress_stdout

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# suppress_stdout
# ---------------------------------------------------------------------------

def test_suppress_stdout_captures_output(capsys):
    """Decorated function should produce no stdout output."""
    @suppress_stdout
    def noisy():
        print("this should not appear")
        return 42

    result = noisy()
    captured = capsys.readouterr()
    assert result == 42
    assert "this should not appear" not in captured.out


def test_suppress_stdout_restores_on_exception():
    """stdout should be restored even if the decorated function raises."""
    @suppress_stdout
    def raises():
        print("noise")
        raise ValueError("boom")

    original = sys.stdout
    with pytest.raises(ValueError, match="boom"):
        raises()
    assert sys.stdout is original


def test_suppress_stdout_preserves_return():
    """Return values pass through the decorator unchanged."""
    @suppress_stdout
    def returns_dict():
        print("noise")
        return {"a": 1, "b": [2, 3]}

    result = returns_dict()
    assert result == {"a": 1, "b": [2, 3]}


def test_suppress_stdout_wraps_name():
    """Decorator preserves function name via @wraps."""
    @suppress_stdout
    def my_func():
        pass

    assert my_func.__name__ == "my_func"


# ---------------------------------------------------------------------------
# _vibanal_wfn — requires working psi4
# ---------------------------------------------------------------------------

def _psi4_importable():
    """Check if psi4 can be imported."""
    try:
        import psi4
        return True
    except ImportError:
        return False


psi4_available = _psi4_importable()


@pytest.fixture(scope="session")
def hessian_data():
    """Load pre-computed hessian data from the QCFractal server."""
    hess = np.load(str(DATA_DIR / "hessian_11atom.npy"))
    mol = Molecule.from_file(str(DATA_DIR / "hessian_11atom.xyz"))
    with open(str(DATA_DIR / "hessian_11atom_energy.json")) as f:
        meta = json.load(f)
    return hess, mol, meta["energy"]


def test_hessian_fixture_shapes(hessian_data):
    """Verify the saved hessian data has correct dimensions."""
    hess, mol, energy = hessian_data
    natoms = len(mol.symbols)
    assert natoms == 11
    assert hess.shape == (3 * natoms, 3 * natoms)
    # Hessian should be symmetric
    np.testing.assert_allclose(hess, hess.T, atol=1e-10)
    assert isinstance(energy, float)
    assert energy < 0  # negative total energy


def test_hessian_fixture_molecule(hessian_data):
    """Verify the molecule fixture has expected composition."""
    _, mol, _ = hessian_data
    symbols = list(mol.symbols)
    assert symbols.count("C") == 2
    assert symbols.count("O") == 3
    assert symbols.count("H") == 6


@pytest.mark.skipif(not psi4_available, reason="psi4 not available")
def test_vibanal_wfn_runs(hessian_data):
    """Integration test: _vibanal_wfn produces frequencies and thermo data.

    Uses real hessian data from the QCFractal server (b3lyp/6-31g, 11 atoms).
    """
    import psi4
    psi4.set_memory("500 MB")
    psi4.set_output_file("/tmp/psi4_zpve_test.log")

    from beep.core.zpve import _vibanal_wfn

    hess, mol, energy = hessian_data
    vibinfo, therminfo = _vibanal_wfn(hess=hess, molecule=mol, energy=energy)

    # Should have vibrational frequencies
    assert "omega" in vibinfo
    freqs = vibinfo["omega"].data
    ndof = 3 * len(mol.symbols)
    assert len(freqs) == ndof  # all 3N modes returned (incl TR near-zero)

    # Count real vibrational modes (> 10 cm-1)
    n_vib_expected = ndof - 6  # nonlinear molecule: 3N-6 = 27
    real_freqs = [f.real for f in freqs if abs(f.real) > 10]
    # Some modes may be near-zero if structure isn't at a true minimum
    assert len(real_freqs) >= n_vib_expected - 5
    assert len(real_freqs) <= n_vib_expected

    # Thermodynamic corrections should be present and physical
    assert "ZPE_corr" in therminfo
    assert "ZPE_tot" in therminfo
    zpve = therminfo["ZPE_corr"].data
    assert zpve > 0  # ZPVE is always positive

    # Verify against known values from our manual run
    assert abs(zpve - 0.083933) < 0.001  # ZPVE ~ 0.0839 Eh
    assert therminfo["ZPE_tot"].data < 0  # total energy with ZPE is negative


@pytest.mark.skipif(not psi4_available, reason="psi4 not available")
def test_vibanal_wfn_frequencies_physical(hessian_data):
    """Vibrational frequencies should be in a physically reasonable range."""
    import psi4
    psi4.set_memory("500 MB")
    psi4.set_output_file("/tmp/psi4_zpve_test2.log")

    from beep.core.zpve import _vibanal_wfn

    hess, mol, energy = hessian_data
    vibinfo, _ = _vibanal_wfn(hess=hess, molecule=mol, energy=energy)

    freqs = vibinfo["omega"].data
    real_freqs = sorted([f.real for f in freqs if abs(f.real) > 10])

    # OH stretch should be ~3800 cm-1 (this molecule has O-H bonds)
    assert max(real_freqs) > 3000  # at least one high-frequency mode
    assert max(real_freqs) < 5000  # but not unphysically high

    # Lowest real freq should be a low-energy skeletal mode
    assert min(real_freqs) > 50
    assert min(real_freqs) < 1000


@pytest.mark.skipif(not psi4_available, reason="psi4 not available")
def test_vibanal_wfn_thermo_consistency(hessian_data):
    """Thermodynamic corrections should satisfy H = E + PV, G = H - TS."""
    import psi4
    psi4.set_memory("500 MB")
    psi4.set_output_file("/tmp/psi4_zpve_test3.log")

    from beep.core.zpve import _vibanal_wfn

    hess, mol, energy = hessian_data
    _, therminfo = _vibanal_wfn(hess=hess, molecule=mol, energy=energy)

    # H_corr = E_corr + kT (at 298.15 K, kT ~ 0.000944 Eh)
    kt = 0.000944
    h_minus_e = therminfo["H_corr"].data - therminfo["E_corr"].data
    assert abs(h_minus_e - kt) < 0.0001

    # G < H (entropy is positive → TS > 0)
    assert therminfo["G_corr"].data < therminfo["H_corr"].data

    # ZPE < E_corr (thermal energy includes ZPE + thermal population)
    assert therminfo["ZPE_corr"].data < therminfo["E_corr"].data
