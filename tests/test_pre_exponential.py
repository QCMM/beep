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
        rot = (8 * pi**2 * kB * T / h**2) * (Ib / sigma)  # classical linear rotor
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


class TestRotationalPartitionFunctionValues:
    """Pin the rotational partition functions to independent reference values.

    The linear rotor must reproduce the high-temperature limit of the quantum
    rotor, q_rot = kB T / (sigma h c B). For CO, B = 1.9313 cm^-1 gives
    q_rot(298.15 K) = 107.3. Minissale et al. 2022 Eq. 20 (an extra sqrt(pi),
    the form BEEP used before 0.16) would give 190, their Table 4 337.
    """

    kB = qcel.constants.get("kb")
    h = qcel.constants.get("h")
    c = qcel.constants.get("c")
    amu_A2 = qcel.constants.get("atomic mass constant") * 1e-20

    def _rot_part(self, m, T, sigma, Ia, Ib, Ic, A):
        nu = pre_exponential_factor(m, [T], sigma, Ia, Ib, Ic, A)[0]
        translational = (2 * math.pi * m * self.kB * T / self.h**2) * A
        return nu / ((self.kB * T / self.h) * translational)

    def test_linear_co_matches_spectroscopic_high_t_limit(self):
        T, B_cm = 298.15, 1.9313
        q_ref = self.kB * T / (self.h * self.c * 100.0 * B_cm)  # 107.3
        I = self.h / (8 * math.pi**2 * self.c * 100.0 * B_cm)  # kg m^2 from B
        q = self._rot_part(28.0 * qcel.constants.get("atomic mass constant"),
                           T, 1, 0.0, I, I, 1e-19)
        assert q == pytest.approx(q_ref, rel=1e-6)
        assert q == pytest.approx(107.3, rel=1e-3)
        # and explicitly not the sqrt(pi)/pi variants of Minissale 2022
        assert not q == pytest.approx(q_ref * math.sqrt(math.pi), rel=1e-2)
        assert not q == pytest.approx(q_ref * math.pi, rel=1e-2)

    def test_nonlinear_matches_minissale_table4(self):
        # CH4: Ix=Iy=Iz=3.17 amu A^2, sigma=12, Tpeak=47 K -> q_rot,3D = 2.25
        # H2O: 1.83/1.21/0.62 amu A^2, sigma=2, Tpeak=155 K -> 16.77
        for (Ia, Ib, Ic, sigma, T, q_tab) in [
            (3.17, 3.17, 3.17, 12, 47.0, 2.25),
            (1.83, 1.21, 0.62, 2, 155.0, 16.77),
        ]:
            q = self._rot_part(18.0 * qcel.constants.get("atomic mass constant"), T, sigma,
                               Ia * self.amu_A2, Ib * self.amu_A2, Ic * self.amu_A2, 1e-19)
            assert q == pytest.approx(q_tab, rel=5e-3)

    def test_default_surface_area_is_1e_minus_19(self):
        from beep.models.pre_exp import PreExpConfig
        cfg = PreExpConfig(workflow="pre_exp")
        assert cfg.molecule_surface_area == pytest.approx(1e-19)
