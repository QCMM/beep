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
from typing import List, Sequence

from qcelemental.models import Molecule

from .exceptions import MbeFragmentationError

logger = logging.getLogger("beep")


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
    charges = [0]
    multiplicities = [1]
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
    charges = [0 for _ in fragments]
    multiplicities = [1 for _ in fragments]
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
    charges = [0 for _ in fragments]
    multiplicities = [1 for _ in fragments]
    logger.debug("Cluster fragments: %s", fragments)
    return _with_fragments(molecule, fragments, charges, multiplicities)
