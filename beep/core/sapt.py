"""Pure molecule-fragment utilities for SAPT workflows."""
from collections import Counter
from collections.abc import Mapping
import re
from typing import Any

import qcelemental as qcel
from qcelemental.models import Molecule


WATER_CLUSTER_PATTERN = re.compile(r"^W([1-9][0-9]*)_[0-9]+$", re.IGNORECASE)
HARTREE_TO_KCAL_MOL = qcel.constants.conversion_factor("hartree", "kcal/mol")
SAPT_COMPONENT_LABELS = {
    "electrostatics": "elst",
    "exchange": "exch",
    "induction": "ind",
    "dispersion": "disp",
    "total_sapt": "total",
}


def water_count_from_cluster(cluster_name: str) -> int:
    """Return the number of water molecules encoded in a cluster name."""
    match = WATER_CLUSTER_PATTERN.fullmatch(cluster_name)
    if match is None:
        raise ValueError(
            f"Invalid water-cluster name '{cluster_name}'. Expected "
            "W<number>_<site>, for example W5_01 or W22_14."
        )
    return int(match.group(1))


def build_sapt_molecule(
    structure: Molecule,
    cluster_name: str,
    *,
    surface_charge: int = 0,
    surface_multiplicity: int = 1,
    molecule_charge: int = 0,
    molecule_multiplicity: int = 1,
) -> Molecule:
    """Create a two-fragment SAPT molecule from an optimized binding site.

    BEEP binding-site datasets store the water-cluster atoms first and the
    adsorbate atoms second. This function validates that convention before
    assigning explicit surface and adsorbate fragments.
    """
    water_count = water_count_from_cluster(cluster_name)
    surface_atom_count = 3 * water_count
    atom_count = len(structure.symbols)
    if atom_count <= surface_atom_count:
        raise ValueError(
            f"Structure has {atom_count} atoms; {cluster_name.upper()} requires "
            f"{surface_atom_count} surface atoms plus at least one adsorbate atom."
        )

    surface_symbols = structure.symbols[:surface_atom_count]
    found = Counter(str(symbol) for symbol in surface_symbols)
    expected = Counter({"O": water_count, "H": 2 * water_count})
    if found != expected:
        raise ValueError(
            f"Water validation failed in the first {surface_atom_count} atoms: "
            f"expected {dict(expected)}, found {dict(found)}."
        )

    surface_fragment = list(range(surface_atom_count))
    molecule_fragment = list(range(surface_atom_count, atom_count))
    return Molecule(
        symbols=list(structure.symbols),
        geometry=structure.geometry,
        molecular_charge=surface_charge + molecule_charge,
        molecular_multiplicity=molecule_multiplicity,
        fragments=[surface_fragment, molecule_fragment],
        fragment_charges=[surface_charge, molecule_charge],
        fragment_multiplicities=[
            surface_multiplicity,
            molecule_multiplicity,
        ],
        fix_com=True,
        fix_orientation=True,
    )


def _qcvars_from_source(source: Any) -> dict[str, Any]:
    """Return a case-normalized QC-variable mapping."""
    if isinstance(source, Mapping):
        data = source
    else:
        extras = getattr(source, "extras", None)
        if extras is not None:
            data = extras
        elif hasattr(source, "dict"):
            data = source.dict()
        else:
            raise TypeError("SAPT result source must be a mapping or record object")

    if isinstance(data.get("extras"), Mapping):
        data = data["extras"]
    if isinstance(data.get("qcvars"), Mapping):
        data = data["qcvars"]
    return {str(key).lower(): value for key, value in data.items()}


def extract_sapt_components(source: Any, method: str = "sapt0") -> dict[str, float]:
    """Extract SAPT components from QC variables and convert to kcal/mol.

    ``source`` may be a QCFractal singlepoint record, its serialized mapping,
    an ``extras`` mapping, or the QC-variable mapping itself.
    """
    qcvars = _qcvars_from_source(source)
    method_name = method.strip().lower()
    result: dict[str, float] = {}
    missing: list[str] = []

    for component, label in SAPT_COMPONENT_LABELS.items():
        candidates = [
            f"{method_name} {label} energy",
            f"sapt {label} energy",
        ]
        value = next((qcvars[key] for key in candidates if key in qcvars), None)
        if value is None:
            missing.append(candidates[0])
            continue
        result[f"{component}_kcal_mol"] = float(value) * HARTREE_TO_KCAL_MOL

    if missing:
        raise KeyError("Missing SAPT QC variables: " + ", ".join(missing))
    return result
