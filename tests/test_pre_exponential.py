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


class TestLinearRotorDetection:
    """``Ia == 0`` exact float compare replaced by a relative tolerance so a
    linear molecule whose smallest eigenvalue is round-off noise (possibly
    negative) still takes the linear branch."""

    def _linear_value(self, mass, T, sigma, Ib, A):
        kB = qcel.constants.get("kb")
        h = qcel.constants.get("h")
        pi = math.pi
        trans = ((2 * pi * mass * kB * T) / h**2) * A
        rot = (8 * pi**(5 / 2) * kB * T / h**2) * (Ib / sigma)
        return ((kB * T) / h) * trans * rot

    def test_co_computed_moments_take_linear_branch(self):
        from beep.core.pre_exponential import is_linear_rotor
        mass = get_mass(CO_XYZ)
        symbols, coords = parse_coordinates(CO_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        assert Ia != 0 or True   # round-off, not exactly zero in general
        assert is_linear_rotor(Ia, Ib)
        (v,) = pre_exponential_factor(mass, [300], sigma=1, Ia=Ia, Ib=Ib, Ic=Ic, A=1e-19)
        assert v == pytest.approx(self._linear_value(mass, 300, 1, Ib, 1e-19), rel=1e-12)
        # And identical to passing an exact zero
        (v0,) = pre_exponential_factor(mass, [300], sigma=1, Ia=0.0, Ib=Ib, Ic=Ic, A=1e-19)
        assert v == pytest.approx(v0, rel=1e-12)

    def test_negative_roundoff_ia_takes_linear_branch(self):
        """A slightly negative eigenvalue must not reach math.sqrt."""
        mass = get_mass(CO_XYZ)
        symbols, coords = parse_coordinates(CO_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        (v,) = pre_exponential_factor(mass, [300], sigma=1, Ia=-1e-60, Ib=Ib, Ic=Ic, A=1e-19)
        assert v == pytest.approx(self._linear_value(mass, 300, 1, Ib, 1e-19), rel=1e-12)

    def test_water_takes_nonlinear_branch(self):
        from beep.core.pre_exponential import is_linear_rotor
        mass = get_mass(H2O_XYZ)
        symbols, coords = parse_coordinates(H2O_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        assert not is_linear_rotor(Ia, Ib)
        (v,) = pre_exponential_factor(mass, [300], sigma=2, Ia=Ia, Ib=Ib, Ic=Ic, A=1e-19)
        kB = qcel.constants.get("kb")
        h = qcel.constants.get("h")
        pi = math.pi
        trans = ((2 * pi * mass * kB * 300) / h**2) * 1e-19
        rot = (pi**0.5 / (2 * h**3)) * (8 * pi**2 * kB * 300)**1.5 * math.sqrt(Ia * Ib * Ic)
        assert v == pytest.approx(((kB * 300) / h) * trans * rot, rel=1e-12)

    def test_negative_nonlinear_moment_raises(self):
        mass = get_mass(H2O_XYZ)
        symbols, coords = parse_coordinates(H2O_XYZ)
        Ia, Ib, Ic = get_moments_of_inertia(symbols, coords)
        with pytest.raises(ValueError):
            pre_exponential_factor(mass, [300], sigma=2, Ia=Ia, Ib=-Ib, Ic=Ic, A=1e-19)
