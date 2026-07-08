"""Tests for pure SAPT molecule-fragment utilities."""
from types import SimpleNamespace

import numpy as np
import pytest
from qcelemental.models import Molecule

from beep.core.sapt import (
    HARTREE_TO_KCAL_MOL,
    build_sapt_molecule,
    extract_sapt_components,
    water_count_from_cluster,
)


def _combined_structure(cluster, adsorbate_symbols):
    adsorbate_geometry = np.arange(
        len(adsorbate_symbols) * 3,
        dtype=float,
    ).reshape(-1, 3) + 100.0
    return Molecule(
        symbols=list(cluster.symbols) + list(adsorbate_symbols),
        geometry=np.vstack([cluster.geometry, adsorbate_geometry]),
        fix_com=True,
        fix_orientation=True,
    )


@pytest.mark.parametrize(
    ("cluster_name", "expected"),
    [("W5_01", 5), ("w22_14", 22)],
)
def test_water_count_from_cluster(cluster_name, expected):
    assert water_count_from_cluster(cluster_name) == expected


def test_build_sapt_molecule_for_w5_so2(ws5_cluster):
    structure = _combined_structure(ws5_cluster, ["S", "O", "O"])
    fragmented = build_sapt_molecule(structure, "W5_01")

    assert [len(fragment) for fragment in fragmented.fragments] == [15, 3]
    assert list(fragmented.symbols[fragmented.fragments[1]]) == ["S", "O", "O"]
    assert fragmented.fragment_charges == [0.0, 0.0]
    assert fragmented.fragment_multiplicities == [1, 1]
    assert np.allclose(fragmented.geometry, structure.geometry)


def test_build_sapt_molecule_for_w22_diol(w22_cluster):
    diol_symbols = ["C", "C", "O", "O", "H", "H", "H", "H", "H", "H"]
    structure = _combined_structure(w22_cluster, diol_symbols)
    fragmented = build_sapt_molecule(structure, "W22_01")

    assert [len(fragment) for fragment in fragmented.fragments] == [66, 10]
    assert list(fragmented.symbols[fragmented.fragments[1]]) == diol_symbols


def test_build_sapt_molecule_preserves_fragment_metadata(ws5_cluster):
    structure = _combined_structure(ws5_cluster, ["O", "H"])
    fragmented = build_sapt_molecule(
        structure,
        "W5_01",
        molecule_charge=0,
        molecule_multiplicity=2,
    )

    assert fragmented.fragment_charges == [0.0, 0.0]
    assert fragmented.fragment_multiplicities == [1, 2]
    assert fragmented.molecular_multiplicity == 2


def test_build_sapt_molecule_rejects_invalid_surface_composition(ws5_cluster):
    symbols = list(ws5_cluster.symbols)
    symbols[0] = "C"
    structure = Molecule(
        symbols=symbols + ["S", "O", "O"],
        geometry=np.vstack(
            [
                ws5_cluster.geometry,
                np.arange(9, dtype=float).reshape(3, 3) + 100.0,
            ]
        ),
        fix_com=True,
        fix_orientation=True,
    )

    with pytest.raises(ValueError, match="Water validation failed"):
        build_sapt_molecule(structure, "W5_01")


def test_build_sapt_molecule_rejects_missing_adsorbate(ws5_cluster):
    with pytest.raises(ValueError, match="at least one adsorbate atom"):
        build_sapt_molecule(ws5_cluster, "W5_01")


def test_water_count_rejects_invalid_cluster_name():
    with pytest.raises(ValueError, match="Invalid water-cluster name"):
        water_count_from_cluster("ice_cluster")


def _sapt_qcvars(case=str.lower):
    values = {
        "SAPT0 ELST ENERGY": -0.010,
        "SAPT0 EXCH ENERGY": 0.006,
        "SAPT0 IND ENERGY": -0.003,
        "SAPT0 DISP ENERGY": -0.004,
        "SAPT0 TOTAL ENERGY": -0.011,
    }
    return {case(key): value for key, value in values.items()}


def test_extract_sapt_components_from_qcfractal_record():
    record = SimpleNamespace(extras={"qcvars": _sapt_qcvars()})
    result = extract_sapt_components(record)

    assert result["electrostatics_kcal_mol"] == pytest.approx(
        -0.010 * HARTREE_TO_KCAL_MOL
    )
    assert result["exchange_kcal_mol"] == pytest.approx(
        0.006 * HARTREE_TO_KCAL_MOL
    )
    assert result["induction_kcal_mol"] == pytest.approx(
        -0.003 * HARTREE_TO_KCAL_MOL
    )
    assert result["dispersion_kcal_mol"] == pytest.approx(
        -0.004 * HARTREE_TO_KCAL_MOL
    )
    assert result["total_sapt_kcal_mol"] == pytest.approx(
        -0.011 * HARTREE_TO_KCAL_MOL
    )


def test_extract_sapt_components_accepts_uppercase_serialized_record():
    source = {"extras": {"qcvars": _sapt_qcvars(str.upper)}}
    result = extract_sapt_components(source, method="SAPT0")

    assert result["total_sapt_kcal_mol"] == pytest.approx(
        -0.011 * HARTREE_TO_KCAL_MOL
    )


def test_extract_sapt_components_accepts_generic_sapt_qcvars():
    qcvars = {
        key.replace("sapt0", "sapt"): value
        for key, value in _sapt_qcvars().items()
    }
    result = extract_sapt_components(qcvars)

    assert result["total_sapt_kcal_mol"] == pytest.approx(
        -0.011 * HARTREE_TO_KCAL_MOL
    )


def test_extract_sapt_components_rejects_missing_component():
    qcvars = _sapt_qcvars()
    del qcvars["sapt0 disp energy"]

    with pytest.raises(KeyError, match="sapt0 disp energy"):
        extract_sapt_components(qcvars)
