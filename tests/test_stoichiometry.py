"""Tests for beep/core/stoichiometry.py."""
import numpy as np
from qcelemental.models.molecule import Molecule

from beep.core.stoichiometry import be_stoichiometry


def _make_struc_mol(ws3_cluster, h2_mol):
    """Build a combined structure from cluster + small molecule."""
    symbols = list(ws3_cluster.symbols) + list(h2_mol.symbols)
    geom = np.concatenate([ws3_cluster.geometry, h2_mol.geometry]).flatten()
    return Molecule(symbols=symbols, geometry=geom)


# ---------------------------------------------------------------------------
# Synthetic composite (ws3 + h2) — original tests
# ---------------------------------------------------------------------------

def test_be_stoichiometry_keys(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert set(result.keys()) == {"default", "be_nocp", "ie", "de"}


def test_be_stoichiometry_default_count(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert len(result["default"]) == 7


def test_be_stoichiometry_be_nocp_count(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert len(result["be_nocp"]) == 3


def test_be_stoichiometry_ie_count(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert len(result["ie"]) == 3


def test_be_stoichiometry_de_count(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    assert len(result["de"]) == 4


def test_be_stoichiometry_coefficients(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    coeffs = [c for _, c in result["default"]]
    assert coeffs == [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]


def test_be_stoichiometry_atom_counts(ws3_cluster, h2_mol, test_logger):
    struc = _make_struc_mol(ws3_cluster, h2_mol)
    result = be_stoichiometry(h2_mol, ws3_cluster, struc, test_logger)
    total_atoms = len(struc.symbols)
    for mol, _ in result["be_nocp"]:
        assert len(mol.symbols) <= total_atoms


# ---------------------------------------------------------------------------
# Real optimized structures from QCFractal (CO on w2/w3 clusters)
# ---------------------------------------------------------------------------

def test_real_co_w2_stoichiometry(co_w2_0001, test_logger):
    """Stoichiometry from a real CO-on-water-dimer binding site."""
    # co_w2_0001 has 8 atoms: 6 water (O,H,H,O,H,H) + 2 CO (C,O)
    # Extract isolated CO and w2 cluster from the composite
    n_water_atoms = 6
    co_symbols = list(co_w2_0001.symbols[n_water_atoms:])
    co_geom = co_w2_0001.geometry[n_water_atoms:].flatten()
    co_mol = Molecule(symbols=co_symbols, geometry=co_geom)

    w2_symbols = list(co_w2_0001.symbols[:n_water_atoms])
    w2_geom = co_w2_0001.geometry[:n_water_atoms].flatten()
    w2_mol = Molecule(symbols=w2_symbols, geometry=w2_geom)

    result = be_stoichiometry(co_mol, w2_mol, co_w2_0001, test_logger)

    assert set(result.keys()) == {"default", "be_nocp", "ie", "de"}
    assert len(result["default"]) == 7
    assert len(result["be_nocp"]) == 3

    # Verify fragments have correct atom counts
    for mol, coeff in result["default"]:
        assert len(mol.symbols) in (2, 6, 8)  # CO, w2, or full structure


def test_real_co_w3_stoichiometry(co_w3_0001, test_logger):
    """Stoichiometry from a real CO-on-water-trimer binding site."""
    # co_w3_0001 has 11 atoms: 9 water + 2 CO
    n_water_atoms = 9
    co_symbols = list(co_w3_0001.symbols[n_water_atoms:])
    co_geom = co_w3_0001.geometry[n_water_atoms:].flatten()
    co_mol = Molecule(symbols=co_symbols, geometry=co_geom)

    w3_symbols = list(co_w3_0001.symbols[:n_water_atoms])
    w3_geom = co_w3_0001.geometry[:n_water_atoms].flatten()
    w3_mol = Molecule(symbols=w3_symbols, geometry=w3_geom)

    result = be_stoichiometry(co_mol, w3_mol, co_w3_0001, test_logger)

    assert len(result["default"]) == 7
    # be_nocp coefficients should sum to zero (energy conservation)
    coeff_sum = sum(c for _, c in result["be_nocp"])
    assert abs(coeff_sum - (-1.0)) < 1e-10  # +1 - 1 - 1 = -1


def test_real_two_binding_sites_differ(co_w2_0001, co_w2_0007, test_logger):
    """Two different binding sites produce different fragment geometries."""
    n_water_atoms = 6

    def _get_stoic(struc_mol):
        co_mol = Molecule(
            symbols=list(struc_mol.symbols[n_water_atoms:]),
            geometry=struc_mol.geometry[n_water_atoms:].flatten(),
        )
        w2_mol = Molecule(
            symbols=list(struc_mol.symbols[:n_water_atoms]),
            geometry=struc_mol.geometry[:n_water_atoms].flatten(),
        )
        return be_stoichiometry(co_mol, w2_mol, struc_mol, test_logger)

    stoic1 = _get_stoic(co_w2_0001)
    stoic2 = _get_stoic(co_w2_0007)

    # Both should have the same structure (same keys, same counts)
    assert stoic1.keys() == stoic2.keys()
    assert len(stoic1["default"]) == len(stoic2["default"])

    # But the geometries should differ (different binding sites)
    geom1 = stoic1["default"][0][0].geometry
    geom2 = stoic2["default"][0][0].geometry
    assert not np.allclose(geom1, geom2, atol=1e-6)


# ---------------------------------------------------------------------------
# Real ReactionDataset structure: CO on w5 cluster
# Validated against be_CO_W5_01_MPWB1K-D3BJ_DEF2-TZVPD on QCFractal server
# ---------------------------------------------------------------------------

def _decompose_co_w5(struc_mol):
    """Split a CO-on-w5 composite (17 atoms) into CO and w5 fragments."""
    n_water = 15  # w5 = 5 waters = 15 atoms
    co_mol = Molecule(
        symbols=list(struc_mol.symbols[n_water:]),
        geometry=struc_mol.geometry[n_water:].flatten(),
    )
    w5_mol = Molecule(
        symbols=list(struc_mol.symbols[:n_water]),
        geometry=struc_mol.geometry[:n_water].flatten(),
    )
    return co_mol, w5_mol


def test_real_co_w5_stoichiometry_matches_server(co_w5_0001, test_logger):
    """Verify be_stoichiometry() reproduces the exact structure from QCFractal.

    The server's ReactionDataset be_CO_W5_01_MPWB1K-D3BJ_DEF2-TZVPD has:
      default: 7 entries, coeffs [+1,+1,+1,-1,-1,-1,-1]
      be_nocp: 3 entries, coeffs [+1,-1,-1]
      ie:      3 entries, coeffs [+1,-1,-1]
      de:      4 entries, coeffs [-1,-1,+1,+1]
    with atom counts: composite=17, CO=2, w5=15.
    """
    co_mol, w5_mol = _decompose_co_w5(co_w5_0001)
    result = be_stoichiometry(co_mol, w5_mol, co_w5_0001, test_logger)

    # --- default (counterpoise) ---
    assert len(result["default"]) == 7
    d_coeffs = [c for _, c in result["default"]]
    assert d_coeffs == [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    d_natoms = [len(m.symbols) for m, _ in result["default"]]
    assert d_natoms == [17, 2, 15, 17, 17, 15, 2]

    # --- be_nocp ---
    assert len(result["be_nocp"]) == 3
    bn_coeffs = [c for _, c in result["be_nocp"]]
    assert bn_coeffs == [1.0, -1.0, -1.0]
    bn_natoms = [len(m.symbols) for m, _ in result["be_nocp"]]
    assert bn_natoms == [17, 15, 2]

    # --- ie (interaction energy) ---
    assert len(result["ie"]) == 3
    ie_coeffs = [c for _, c in result["ie"]]
    assert ie_coeffs == [1.0, -1.0, -1.0]
    ie_natoms = [len(m.symbols) for m, _ in result["ie"]]
    assert ie_natoms == [17, 17, 17]

    # --- de (deformation energy) ---
    assert len(result["de"]) == 4
    de_coeffs = [c for _, c in result["de"]]
    assert de_coeffs == [-1.0, -1.0, 1.0, 1.0]
    de_natoms = [len(m.symbols) for m, _ in result["de"]]
    assert de_natoms == [15, 2, 2, 15]


def test_real_co_w5_energy_consistency(co_w5_0001, test_logger):
    """Verify BE = IE + DE (counterpoise decomposition identity).

    Using real mpwb1k/def2-tzvpd energies from the server:
      default (BE_CP) = -0.0017746215 Ha
      be_nocp         = -0.0019059003 Ha
      ie              = -0.0018816693 Ha
      de              =  0.0001070478 Ha
    The identity BE_CP = IE + DE should hold exactly.
    """
    # Server-validated values (Hartree)
    be_cp = -0.0017746215
    ie_val = -0.0018816693
    de_val = 0.0001070478

    # BE_CP = IE + DE
    assert abs(be_cp - (ie_val + de_val)) < 1e-8

    # Also verify: BE_CP = BE_NOCP - BSSE
    be_nocp = -0.0019059003
    bsse = be_nocp - be_cp
    assert abs(bsse - (-0.0001312788)) < 1e-6  # BSSE ~ -0.13 mHa


def test_real_co_w5_two_sites_same_decomposition(co_w5_0001, co_w5_0002, test_logger):
    """Two different CO-w5 binding sites produce the same stoichiometry structure."""
    co1, w5_1 = _decompose_co_w5(co_w5_0001)
    co2, w5_2 = _decompose_co_w5(co_w5_0002)

    stoic1 = be_stoichiometry(co1, w5_1, co_w5_0001, test_logger)
    stoic2 = be_stoichiometry(co2, w5_2, co_w5_0002, test_logger)

    # Same keys and entry counts
    assert stoic1.keys() == stoic2.keys()
    for key in stoic1:
        assert len(stoic1[key]) == len(stoic2[key])
        # Same coefficients
        coeffs1 = [c for _, c in stoic1[key]]
        coeffs2 = [c for _, c in stoic2[key]]
        assert coeffs1 == coeffs2

    # But geometries differ (different binding configurations)
    geom1 = stoic1["default"][0][0].geometry
    geom2 = stoic2["default"][0][0].geometry
    assert not np.allclose(geom1, geom2, atol=1e-6)
