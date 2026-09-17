"""Fragmentation logic for the Many-Body Expansion (MBE) based on env_unit_len.

The fragmentation rules assume OptimizationDataset cluster geometries place
small-molecule (adsorbate) atoms last, with the final fragment corresponding to
the adsorbate. Environment fragments are defined by ``env_unit_len`` and must
evenly divide the number of surface atoms.

Ported from the standalone beep-mbe package (``beep_mbe.fragmentation``); the
only change is the exception import, which now comes from
:mod:`beep.core.exceptions`.
"""

import logging
from typing import List, Optional, Sequence, Tuple

from qcelemental.models import Molecule

from .exceptions import MbeFragmentationError

logger = logging.getLogger("beep")


def _parent_fragment_state(
    molecule: Molecule, fragments: List[List[int]]
) -> Optional[Tuple[List[int], List[int]]]:
    """Return the parent's per-fragment (charges, multiplicities) if its own
    fragment partition is identical to ``fragments``; otherwise ``None``."""
    parent = getattr(molecule, "fragments", None)
    if parent is None or len(parent) != len(fragments):
        return None
    if any(list(p) != list(f) for p, f in zip(parent, fragments)):
        return None
    charges = list(molecule.fragment_charges)
    mults = list(molecule.fragment_multiplicities)
    if len(charges) != len(fragments) or len(mults) != len(fragments):
        return None
    return [int(round(c)) for c in charges], [int(m) for m in mults]


def _with_fragments(
    molecule: Molecule,
    fragments: List[List[int]],
    fragment_charges: Sequence[int],
    fragment_multiplicities: Sequence[int],
) -> Molecule:
    """Return a Molecule with fragment metadata applied."""
    data = molecule.dict()
    data.update(
        {
            "fragments": fragments,
            "fragment_charges": list(fragment_charges),
            "fragment_multiplicities": list(fragment_multiplicities),
        }
    )
    return Molecule(**data)


def fragment_small_molecule(molecule: Molecule) -> Molecule:
    """Define a single-fragment representation for a standalone small molecule.

    Parameters
    ----------
    molecule
        Small-molecule geometry to treat as one fragment.

    Returns
    -------
    Molecule
        Molecule annotated with one fragment covering all atoms.
    """
    atom_count = len(molecule.symbols)
    fragments = [list(range(atom_count))]
    # Single fragment: it carries the whole molecule's charge and multiplicity
    # (a radical adsorbate such as HCO must stay a doublet).
    charges = [int(round(molecule.molecular_charge))]
    multiplicities = [int(molecule.molecular_multiplicity)]
    logger.debug("Small molecule fragments: %s", fragments)
    return _with_fragments(molecule, fragments, charges, multiplicities)


def fragment_surface_model(molecule: Molecule, env_unit_len: int) -> Molecule:
    """Split a surface model into equal-sized environment fragments.

    Parameters
    ----------
    molecule
        Surface model geometry to fragment.
    env_unit_len
        Number of atoms per environment fragment; must evenly divide the surface.

    Returns
    -------
    Molecule
        Surface model with fragment annotations for MBE.

    Raises
    ------
    MbeFragmentationError
        If the surface atom count is not divisible by ``env_unit_len``.
    """
    atom_count = len(molecule.symbols)
    if atom_count % env_unit_len != 0:
        raise MbeFragmentationError(
            "Surface model atom count must be divisible by env_unit_len. "
            f"Got {atom_count} atoms and env_unit_len={env_unit_len}."
        )

    fragments = [
        list(range(start, start + env_unit_len))
        for start in range(0, atom_count, env_unit_len)
    ]
    parent_state = _parent_fragment_state(molecule, fragments)
    if parent_state is not None:
        # The input already carries per-fragment charge/multiplicity on the
        # same partition: keep it.
        charges, multiplicities = parent_state
    else:
        charges = [0 for _ in fragments]
        multiplicities = [1 for _ in fragments]
        if (int(round(molecule.molecular_charge)) != 0
                or int(molecule.molecular_multiplicity) != 1):
            logger.warning(
                "Surface model has charge %s / multiplicity %s but no per-unit "
                "fragment information on the env_unit_len partition; "
                "environment fragments are assigned charge 0 / multiplicity 1.",
                molecule.molecular_charge, molecule.molecular_multiplicity,
            )
    logger.debug("Surface model fragments: %s", fragments)
    return _with_fragments(molecule, fragments, charges, multiplicities)


def fragment_cluster(
    molecule: Molecule, env_unit_len: int, small_molecule_atoms: int
) -> Molecule:
    """Fragment a cluster into environment units plus a terminal adsorbate fragment.

    The cluster geometry is assumed to list all small-molecule atoms last, and
    the final fragment is treated as the adsorbate. Environment fragments are
    defined in contiguous blocks of length ``env_unit_len``.

    Parameters
    ----------
    molecule
        Cluster geometry containing surface atoms followed by small-molecule atoms.
    env_unit_len
        Number of atoms per environment fragment.
    small_molecule_atoms
        Count of atoms belonging to the small molecule adsorbate.

    Returns
    -------
    Molecule
        Cluster annotated with environment fragments plus adsorbate fragment.

    Raises
    ------
    MbeFragmentationError
        If the cluster is smaller than the adsorbate or ``env_unit_len`` does
        not evenly divide the number of surface atoms.
    """
    atom_count = len(molecule.symbols)
    if atom_count < small_molecule_atoms:
        raise MbeFragmentationError(
            "Cluster atom count is smaller than small molecule atom count. "
            f"cluster={atom_count}, small={small_molecule_atoms}"
        )

    env_atoms = atom_count - small_molecule_atoms
    if env_atoms % env_unit_len != 0:
        raise MbeFragmentationError(
            "Cluster environment atom count must be divisible by env_unit_len. "
            f"env_atoms={env_atoms}, env_unit_len={env_unit_len}"
        )

    fragments = [
        list(range(start, start + env_unit_len))
        for start in range(0, env_atoms, env_unit_len)
    ]
    fragments.append(list(range(env_atoms, atom_count)))
    parent_state = _parent_fragment_state(molecule, fragments)
    if parent_state is not None:
        # The input already carries per-fragment charge/multiplicity on the
        # same partition: keep it.
        charges, multiplicities = parent_state
    else:
        # Environment units are closed-shell neutral; the terminal adsorbate
        # fragment carries the whole cluster's charge and multiplicity so a
        # radical adsorbate (HCO, CH3O, CH2OH) is not silently made a singlet.
        n_env = len(fragments) - 1
        charges = [0 for _ in range(n_env)] + [int(round(molecule.molecular_charge))]
        multiplicities = [1 for _ in range(n_env)] + [int(molecule.molecular_multiplicity)]
    logger.debug("Cluster fragments: %s", fragments)
    return _with_fragments(molecule, fragments, charges, multiplicities)
