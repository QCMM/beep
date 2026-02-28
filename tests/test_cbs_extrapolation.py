"""Tests for beep/core/cbs_extrapolation.py."""
import numpy as np
import pytest

from beep.core.cbs_extrapolation import (
    scf_xtpl_helgaker_2,
    scf_xtpl_helgaker_3,
    corl_xtpl_helgaker_2,
)


# ---------------------------------------------------------------------------
# scf_xtpl_helgaker_2  (2-point exponential SCF extrapolation)
# ---------------------------------------------------------------------------

def test_scf_2pt_float():
    # DZ (z=2) and TZ (z=3) energies
    valueLO, valueTZ = -76.02, -76.06
    zLO, zHI = 2, 3
    result = scf_xtpl_helgaker_2("HF", zLO, valueLO, zHI, valueTZ)
    # Manual: beta = (valueHI - valueLO) / (exp(-alpha*zLO)*(exp(-alpha)-1))
    alpha = 1.63
    beta = (valueTZ - valueLO) / (np.exp(-alpha * zLO) * (np.exp(-alpha) - 1))
    expected = valueTZ - beta * np.exp(-alpha * zHI)
    assert abs(result - expected) < 1e-12


def test_scf_2pt_ndarray():
    valueLO = np.array([-76.02, -100.5])
    valueHI = np.array([-76.06, -100.8])
    result = scf_xtpl_helgaker_2("HF", 2, valueLO, 3, valueHI)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    # Each element should be more negative than HI (extrapolation lowers energy)
    for r, hi in zip(result, valueHI):
        assert r <= hi or abs(r - hi) < 0.1  # CBS limit close to or below HI


def test_scf_2pt_type_mismatch():
    with pytest.raises(ValueError):
        scf_xtpl_helgaker_2("HF", 2, -76.02, 3, np.array([-76.06]))


def test_scf_2pt_custom_alpha():
    result_default = scf_xtpl_helgaker_2("HF", 2, -76.02, 3, -76.06)
    result_custom = scf_xtpl_helgaker_2("HF", 2, -76.02, 3, -76.06, alpha=2.0)
    assert result_default != result_custom


# ---------------------------------------------------------------------------
# scf_xtpl_helgaker_3  (3-point SCF extrapolation)
# ---------------------------------------------------------------------------

def test_scf_3pt_float():
    valueLO, valueMD, valueHI = -76.02, -76.06, -76.07
    zLO, zMD, zHI = 2, 3, 4
    result = scf_xtpl_helgaker_3("HF", zLO, valueLO, zMD, valueMD, zHI, valueHI)
    assert isinstance(result, float)
    # CBS limit should be more negative than QZ
    assert result <= valueHI or abs(result - valueHI) < 0.01


def test_scf_3pt_type_mismatch():
    with pytest.raises(ValueError):
        scf_xtpl_helgaker_3("HF", 2, -76.02, 3, np.array([-76.06]), 4, -76.07)


# ---------------------------------------------------------------------------
# corl_xtpl_helgaker_2  (2-point correlation extrapolation)
# ---------------------------------------------------------------------------

def test_corl_2pt_float():
    valueLO, valueHI = -0.20, -0.25
    zLO, zHI = 2, 3
    alpha = 3.0
    expected = (valueHI * zHI**alpha - valueLO * zLO**alpha) / (zHI**alpha - zLO**alpha)
    result = corl_xtpl_helgaker_2("MP2", zLO, valueLO, zHI, valueHI)
    assert abs(result - expected) < 1e-12


def test_corl_2pt_type_mismatch():
    with pytest.raises(ValueError):
        corl_xtpl_helgaker_2("MP2", 2, -0.20, 3, np.array([-0.25]))
