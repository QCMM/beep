"""IE mode of the periodic BE workflows: fragment construction, IE/BE/DE arithmetic,
config defaults. Pure-python (no QCFractal / MACE)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import qcelemental as qcel

from beep.core.periodic_sampler import ANG2BOHR, adsorbate_fragment, strip_adsorbate
from beep.models.be_assemble_periodic import BeAssemblePeriodicConfig
from beep.models.be_comp_periodic import BeCompPeriodicConfig
from beep.workflows.be_assemble_periodic import HARTREE2KCAL, interaction_rows

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CELL = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 30.0]]
PBC = [True, True, False]


def _complex(ads_xyz_ang):
    slab = [("O", [1.0, 1.0, 5.0]), ("H", [1.8, 1.0, 5.5]), ("H", [0.2, 1.0, 5.5])]
    ads = list(zip(["C", "O"], ads_xyz_ang))
    atoms = slab + ads
    geom = np.array([xyz for _, xyz in atoms], dtype=float) * ANG2BOHR
    return qcel.models.Molecule(
        symbols=[s for s, _ in atoms], geometry=geom.ravel(),
        fix_com=True, fix_orientation=True,
    )


def test_fragments_partition_the_complex():
    mol = _complex([[5.0, 5.0, 8.0], [5.0, 5.0, 9.13]])
    slab = strip_adsorbate(mol, 3)
    ads = adsorbate_fragment(mol, 3, CELL, PBC)
    assert list(slab.symbols) + list(ads.symbols) == list(mol.symbols)
    np.testing.assert_allclose(
        np.vstack([slab.geometry, ads.geometry]), np.asarray(mol.geometry).reshape(-1, 3)
    )


def test_straddling_adsorbate_is_made_whole():
    # C at x = 9.9 A, O wrapped to x = 0.3 A (bond crosses the x boundary)
    mol = _complex([[9.9, 5.0, 8.0], [0.3, 5.0, 8.0]])
    ads = adsorbate_fragment(mol, 3, CELL, PBC)
    d = np.linalg.norm(ads.geometry[1] - ads.geometry[0]) / ANG2BOHR
    assert d == pytest.approx(0.4, abs=1e-8)
    # without the cell the bond stays broken
    raw = adsorbate_fragment(mol, 3)
    assert np.linalg.norm(raw.geometry[1] - raw.geometry[0]) / ANG2BOHR == pytest.approx(9.6)


def test_adsorbate_charge_and_multiplicity_override():
    mol = _complex([[5.0, 5.0, 8.0], [5.0, 5.0, 9.13]])
    ads = adsorbate_fragment(mol, 3, CELL, PBC, molecular_charge=0, molecular_multiplicity=1)
    assert ads.molecular_charge == 0 and ads.molecular_multiplicity == 1


def test_ie_arithmetic():
    rows = interaction_rows({"s1": -10.0}, {"s1": -9.0}, {"s1": -0.99})
    assert len(rows) == 1 and len(rows[0]) == 5
    assert rows[0][4] == pytest.approx(-0.01 * HARTREE2KCAL)


def test_be_equals_ie_plus_deformation():
    ec, es, ea = {"s1": -10.0, "s2": -10.0}, {"s1": -9.0, "s2": -9.0}, {"s1": -0.99, "s2": -0.99}
    rows = interaction_rows(ec, es, ea, e_surface={"s1": -9.002}, e_gas=-0.991)
    assert [r[0] for r in rows] == ["s1"]  # s2 has no relaxed surface
    _, _, _, _, ie, be, de_slab, de_ads, de = rows[0]
    assert be == pytest.approx((-10.0 + 9.002 + 0.991) * HARTREE2KCAL)
    assert de_slab == pytest.approx(0.002 * HARTREE2KCAL)
    assert de_ads == pytest.approx(0.001 * HARTREE2KCAL)
    assert be == pytest.approx(ie + de)


def test_quantity_defaults_to_be():
    comp = BeCompPeriodicConfig(**json.loads((EXAMPLES / "be_comp_periodic.json").read_text()))
    asm = BeAssemblePeriodicConfig(**json.loads((EXAMPLES / "be_assemble_periodic.json").read_text()))
    assert comp.quantity == "be" and asm.quantity == "be"
    assert comp.ie_site_filter == "unique"


def test_quantity_rejects_unknown_value():
    cfg = json.loads((EXAMPLES / "be_comp_periodic.json").read_text())
    cfg["quantity"] = "de"
    with pytest.raises(Exception):
        BeCompPeriodicConfig(**cfg)
