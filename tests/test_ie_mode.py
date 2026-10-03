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


def test_gas_adsorbate_without_spec_uses_entry_geometry():
    """A small-molecule collection without a spec for the electronic LOT (e.g. 'volatiles' on
    another server) must fall back to the entry geometry, not raise."""
    import logging
    from types import SimpleNamespace
    from beep.workflows.be_comp_periodic import gas_adsorbate
    co = qcel.models.Molecule(symbols=["C", "O"], geometry=[0, 0, 0, 0, 0, 2.13])

    class DS:
        specifications = {"b3lyp-d4_def2-tzvpd": object()}
        def get_entry(self, name):
            return SimpleNamespace(initial_molecule=co)
        def get_record(self, *a, **k):
            raise AssertionError("must not ask for a record of a missing spec")

    lot = SimpleNamespace(lot_name="lmft-co-b3lyp-v1-s1", display="lmft-co-b3lyp-v1-s1 (mace)")
    assert gas_adsorbate(DS(), "CO", lot, logging.getLogger("t")) is co


# ---------------------------------------------------------------------------
# Periodic BE / IE / DE ReactionDatasets
# ---------------------------------------------------------------------------

def _surface_and_gas():
    # a relaxed bare surface: the slab moved slightly from its geometry in the complex
    surf = strip_adsorbate(_complex([[5.0, 5.0, 8.0], [5.0, 5.0, 9.13]]), 3)
    surf = surf.copy(update={"geometry": np.asarray(surf.geometry) + 0.01})
    gas = qcel.models.Molecule(symbols=["C", "O"], geometry=[0, 0, 0, 0, 0, 2.13],
                               molecular_charge=0, molecular_multiplicity=1)
    return surf, gas


def test_periodic_stoichiometry_ie_only_without_references():
    from beep.core.stoichiometry import periodic_stoichiometry
    mol = _complex([[5.0, 5.0, 8.0], [5.0, 5.0, 9.13]])
    st = periodic_stoichiometry(mol, 3)
    assert set(st) == {"ie"}
    assert [c for _, c in st["ie"]] == [1.0, -1.0, -1.0]
    cpx, slab, ads = (m for m, _ in st["ie"])
    assert cpx is mol
    assert list(slab.symbols) == ["O", "H", "H"] and list(ads.symbols) == ["C", "O"]
    # fragments keep the in-complex coordinates
    np.testing.assert_allclose(np.vstack([slab.geometry, ads.geometry]),
                               np.asarray(mol.geometry).reshape(-1, 3))


def test_periodic_stoichiometry_be_equals_ie_plus_de():
    from beep.core.stoichiometry import periodic_stoichiometry
    mol = _complex([[5.0, 5.0, 8.0], [5.0, 5.0, 9.13]])
    surf, gas = _surface_and_gas()
    st = periodic_stoichiometry(mol, 3, surface_mol=surf, gas_mol=gas)
    assert set(st) == {"be", "ie", "de"}
    assert st["be"][1][0] is surf and st["be"][2][0] is gas
    assert [c for _, c in st["de"]] == [1.0, 1.0, -1.0, -1.0]

    # any per-molecule energy: sum_be = sum_ie + sum_de
    def energy(m):
        return float(np.sum(np.asarray(m.geometry) ** 2)) + 0.1 * len(m.symbols)

    tot = {k: sum(c * energy(m) for m, c in v) for k, v in st.items()}
    assert tot["be"] == pytest.approx(tot["ie"] + tot["de"], abs=1e-10)


def test_split_components():
    from beep.workflows.be_assemble_periodic import split_components
    ie = [(-1.0, 2, -113.0), (1.0, 5, -190.01), (-1.0, 3, -77.0)]
    assert split_components(ie, "ie") == {"complex": -190.01, "slab_frozen": -77.0, "ads_frozen": -113.0}
    assert split_components(ie, "be") == {"complex": -190.01, "surface": -77.0, "gas": -113.0}
    de = [(1.0, 3, -76.9), (1.0, 2, -112.9), (-1.0, 3, -77.0), (-1.0, 2, -113.0)]
    assert split_components(de, "de") == {"slab_frozen": -76.9, "ads_frozen": -112.9,
                                          "surface": -77.0, "gas": -113.0}
    assert split_components(ie[:2], "ie") is None
    assert split_components(de, "ie") is None
    assert split_components([(1.0, 5, 0.0), (-1.0, 2, 0.0), (-1.0, 2, 0.0)], "ie") is None


def test_reaction_energies_sum_specs_and_total():
    from types import SimpleNamespace as NS
    from beep.workflows.be_assemble_periodic import reaction_energies

    def comp(coef, n, e):
        return NS(coefficient=coef, molecule=NS(symbols=["X"] * n),
                  singlepoint_record=NS(status="complete", properties={"return_energy": e}))

    recs = {
        "elec": NS(status="complete", components=[comp(1, 5, -10.0), comp(-1, 3, -6.0), comp(-1, 2, -3.9)]),
        "disp": NS(status="complete", components=[comp(1, 5, -0.30), comp(-1, 3, -0.20), comp(-1, 2, -0.01)]),
    }

    class DS:
        def iterate_records(self, entry_names, specification_names, include):
            for n in entry_names:
                for sp in specification_names:
                    if not (n == "b" and sp == "disp"):
                        yield n, sp, recs[sp]

    import logging
    out = reaction_energies(DS(), "ie", ["a", "b"], "elec", "disp", logging.getLogger("t"))
    assert set(out) == {"a"}  # 'b' lacks its dispersion record
    assert out["a"]["complex"] == pytest.approx(-10.30)
    assert out["a"]["total"] == pytest.approx(-10.30 + 6.20 + 3.91)


def test_interaction_rows_per_site_gas():
    rows = interaction_rows({"s": -10.0}, {"s": -6.0}, {"s": -3.9}, {"s": -6.1}, {"s": -3.95})
    n, ec, es, ea, ie, be, de_slab, de_ads, de = rows[0]
    assert be == pytest.approx(ie + de)
    assert be == pytest.approx((-10.0 + 6.1 + 3.95) * HARTREE2KCAL)


def test_guard_reused_reactions():
    from types import SimpleNamespace as NS
    from beep.workflows.be_comp_periodic import _guard_reused_reactions
    from beep.core.stoichiometry import periodic_stoichiometry
    mol = _complex([[5.0, 5.0, 8.0], [5.0, 5.0, 9.13]])
    st = periodic_stoichiometry(mol, 3)["ie"]

    class DS:
        name = "ds"
        def __init__(self, stored):
            self.stored = stored
            self.entry_names = ["s"]
        def iterate_entries(self, entry_names):
            yield NS(name="s", stoichiometries=[NS(coefficient=c, molecule=m) for m, c in self.stored])

    _guard_reused_reactions(DS(st), {"s": st}, CELL, PBC)  # same entry: fine
    moved = periodic_stoichiometry(_complex([[5.0, 5.0, 8.5], [5.0, 5.0, 9.63]]), 3)["ie"]
    with pytest.raises(ValueError):
        _guard_reused_reactions(DS(st), {"s": moved}, CELL, PBC)
    with pytest.raises(ValueError):
        _guard_reused_reactions(DS(st[:2]), {"s": st}, CELL, PBC)


def test_reaction_component_energies():
    from types import SimpleNamespace as NS
    from beep.adapters.qcfractal_adapter import reaction_component_energies

    def comp(coef, n, e, status="complete"):
        sp = NS(status=status, properties={"return_energy": e})
        return NS(coefficient=coef, molecule=NS(symbols=["X"] * n), singlepoint_record=sp)

    rec = NS(status="complete", components=[comp(1, 5, -1.5), comp(-1, 3, -1.0), comp(-1, 2, -0.4)])
    assert reaction_component_energies(rec) == [(1.0, 5, -1.5), (-1.0, 3, -1.0), (-1.0, 2, -0.4)]
    assert reaction_component_energies(None) is None
    rec.components[2] = comp(-1, 2, -0.4, status="error")
    assert reaction_component_energies(rec) is None


def test_unmoved_surface_merges_de_components():
    """A bare-surface optimization that converged at its first step returns the frozen slab
    itself: the DE's +1/-1 slab terms are one molecule, merged away; BE = IE + DE still holds."""
    from beep.core.stoichiometry import periodic_stoichiometry
    mol = _complex([[5.0, 5.0, 8.0], [5.0, 5.0, 9.13]])
    _, gas = _surface_and_gas()
    surf = strip_adsorbate(mol, 3)  # 'relaxed' surface == frozen slab
    st = periodic_stoichiometry(mol, 3, surface_mol=surf, gas_mol=gas)
    assert len(st["de"]) == 2 and sorted(c for _, c in st["de"]) == [-1.0, 1.0]
    assert {len(m.symbols) for m, _ in st["de"]} == {2}
    assert len({m.get_hash() for m, _ in st["de"]}) == 2
    assert len(st["be"]) == 3 and len(st["ie"]) == 3

    def energy(m):
        return float(np.sum(np.asarray(m.geometry) ** 2)) + 0.1 * len(m.symbols)

    tot = {k: sum(c * energy(m) for m, c in v) for k, v in st.items()}
    assert tot["be"] == pytest.approx(tot["ie"] + tot["de"], abs=1e-10)


def test_reaction_energies_reads_merged_de():
    from types import SimpleNamespace as NS
    from beep.workflows.be_assemble_periodic import reaction_energies

    def comp(coef, n, e):
        return NS(coefficient=coef, molecule=NS(symbols=["X"] * n),
                  singlepoint_record=NS(status="complete", properties={"return_energy": e}))

    recs = {
        "elec": NS(status="complete", components=[comp(1, 2, -3.90), comp(-1, 2, -3.95)]),
        "disp": NS(status="complete", components=[comp(1, 2, -0.01), comp(-1, 2, -0.02)]),
    }

    class DS:
        def iterate_records(self, entry_names, specification_names, include):
            for n in entry_names:
                for sp in specification_names:
                    yield n, sp, recs[sp]

    import logging
    out = reaction_energies(DS(), "de", ["s"], "elec", "disp", logging.getLogger("t"))
    assert out["s"]["total"] == pytest.approx(0.05 + 0.01)
    # be/ie still require their full shape
    assert reaction_energies(DS(), "ie", ["s"], "elec", "disp", logging.getLogger("t")) == {}
