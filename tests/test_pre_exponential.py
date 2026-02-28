"""Tests for beep.core.pre_exponential — pure pre-exponential factor functions."""
import math
import numpy as np
import pytest
import qcelemental as qcel

from beep.core.pre_exponential import (
    get_mass,
    parse_coordinates,
    align_to_z_axis,
    get_moments_of_inertia,
    pre_exponential_factor,
)


# Simple H2 molecule along z-axis
H2_XYZ = "H  0.0  0.0  0.0\nH  0.0  0.0  0.74"

# Water molecule
H2O_XYZ = "O  0.000  0.000  0.117\nH  0.000  0.756 -0.469\nH  0.000 -0.756 -0.469"

# CO molecule
CO_XYZ = "C  0.0  0.0  0.0\nO  0.0  0.0  1.128"


class TestGetMass:
    def test_h2_mass(self):
        mass = get_mass(H2_XYZ)
        # H2 mass ~ 2 * 1.008 amu converted to kg
        expected = 2 * qcel.periodictable.to_mass("H") / (qcel.constants.get("na") * 1000)
        assert abs(mass - expected) < 1e-30

    def test_water_mass(self):
        mass = get_mass(H2O_XYZ)
        expected_amu = qcel.periodictable.to_mass("O") + 2 * qcel.periodictable.to_mass("H")
        expected_kg = expected_amu / (qcel.constants.get("na") * 1000)
        assert abs(mass - expected_kg) < 1e-28

    def test_co_mass(self):
        mass = get_mass(CO_XYZ)
        expected_amu = qcel.periodictable.to_mass("C") + qcel.periodictable.to_mass("O")
        expected_kg = expected_amu / (qcel.constants.get("na") * 1000)
        assert abs(mass - expected_kg) < 1e-28


class TestParseCoordinates:
    def test_h2(self):
        symbols, coords = parse_coordinates(H2_XYZ)
        assert symbols == ["H", "H"]
        assert coords.shape == (2, 3)
        np.testing.assert_allclose(coords[1], [0.0, 0.0, 0.74])

    def test_water(self):
        symbols, coords = parse_coordinates(H2O_XYZ)
        assert symbols == ["O", "H", "H"]
        assert coords.shape == (3, 3)


class TestAlignToZAxis:
    def test_output_shape(self):
        symbols, coords = parse_coordinates(H2O_XYZ)
        aligned = align_to_z_axis(symbols, coords)
        assert aligned.shape == coords.shape

    def test_center_of_mass_at_origin(self):
        symbols, coords = parse_coordinates(H2O_XYZ)
        aligned = align_to_z_axis(symbols, coords)
        masses = np.array([qcel.periodictable.to_mass(s) for s in symbols])
        com = np.sum(masses[:, np.newaxis] * aligned, axis=0) / np.sum(masses)
        np.testing.assert_allclose(com, [0, 0, 0], atol=1e-10)


class TestGetMomentsOfInertia:
    def test_h2_linear(self):
        symbols, coords = parse_coordinates(H2_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        # Linear molecule: Ia ~ 0, Ib == Ic
        assert Ia < 1e-50  # essentially zero for linear molecule along axis
        np.testing.assert_allclose(Ib, Ic, rtol=1e-10)

    def test_water_nonlinear(self):
        symbols, coords = parse_coordinates(H2O_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        # All three should be positive for non-linear molecule
        assert Ia > 0
        assert Ib > 0
        assert Ic > 0
        # Should be sorted: Ia <= Ib <= Ic
        assert Ia <= Ib <= Ic

    def test_co_linear(self):
        symbols, coords = parse_coordinates(CO_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        assert Ia < 1e-50
        np.testing.assert_allclose(Ib, Ic, rtol=1e-10)


class TestPreExponentialFactor:
    def test_returns_list(self):
        mass = get_mass(H2O_XYZ)
        symbols, coords = parse_coordinates(H2O_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        T_list = [100, 200, 300]
        result = pre_exponential_factor(mass, T_list, sigma=2, Ia=Ia, Ib=Ib, Ic=Ic, A=1e-19)
        assert len(result) == 3
        # All values should be positive
        assert all(v > 0 for v in result)

    def test_monotonically_increasing_with_temperature(self):
        mass = get_mass(H2O_XYZ)
        symbols, coords = parse_coordinates(H2O_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        T_list = [50, 100, 200, 300, 500]
        result = pre_exponential_factor(mass, T_list, sigma=2, Ia=Ia, Ib=Ib, Ic=Ic, A=1e-19)
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    def test_linear_molecule_branch(self):
        """Linear molecule has Ia=0, triggering the linear branch."""
        mass = get_mass(H2_XYZ)
        symbols, coords = parse_coordinates(H2_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        T_list = [100, 200]
        result = pre_exponential_factor(mass, T_list, sigma=2, Ia=0, Ib=Ib, Ic=Ic, A=1e-19)
        assert len(result) == 2
        assert all(v > 0 for v in result)

    def test_single_temperature(self):
        mass = get_mass(CO_XYZ)
        symbols, coords = parse_coordinates(CO_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        result = pre_exponential_factor(mass, [300], sigma=1, Ia=Ia, Ib=Ib, Ic=Ic, A=1e-19)
        assert len(result) == 1
        assert result[0] > 0
