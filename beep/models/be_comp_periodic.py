"""Config for be_comp_periodic — submits periodic BE single-points on sampling_periodic outputs."""
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, model_validator
from .base import ServerConfig, LevelOfTheory


class BeCompPeriodicConfig(BaseModel):
    """Submission workflow for periodic binding energies.

    Registers a range-separated MACE + explicit dispersion pair of specs on
    three SinglepointDatasets (per slab: complex, bare-surface; plus one
    shared gas-phase adsorbate) and submits the single-point energies.
    Assembly into per-site BEs happens in ``be_assemble_periodic``.

    Consumes the datasets ``sampling_periodic`` produces:
    - ``<molecule>_<slab>``            optimized adsorbate + slab complexes
    - ``<molecule>_<slab>_surface``    per-site optimized bare slabs
    - ``<small_molecule_collection>``  gas-phase adsorbate reference

    Depends on the QCEngine MACE and dftd3/dftd4 harness patches that read
    ``cell`` / ``pbc`` from spec keywords, so the periodic dispersion
    contribution is actually computed.
    """
    workflow: Literal["be_comp_periodic"] = Field(..., description="Must be 'be_comp_periodic'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")

    # Adsorbate + slab lookup
    molecule: str = Field(..., description="Adsorbate name in the small_molecule_collection")
    small_molecule_collection: str = Field("Small_molecules", description="Adsorbate lookup dataset")
    surface_clusters: List[str] = Field(..., description="Slab names to process (must be non-empty)")

    # BE level of theory — range-separated
    be_electronic_lot: LevelOfTheory = Field(
        ...,
        description=(
            "MACE model trained on the *electronic* (dispersion-free) energy; "
            "the range-separated 'left half' of the BE. Must be an MLP."
        ),
    )
    opt_level_of_theory: str = Field(
        ...,
        description=(
            "Name of the optimization specification holding the geometries to "
            "evaluate, i.e. the LOT sampling_periodic ran with (e.g. "
            "'lmft-co-d-v0'). In a range-separated setup this differs from "
            "be_electronic_lot: geometries come from the dispersion-inclusive "
            "model, the BE from the electronic model plus explicit dispersion."
        ),
    )
    be_dispersion: str = Field(
        ...,
        description=(
            "Explicit dispersion 'right half', e.g. 'mpwb1k-d4' or 'b3lyp-d3bj'. "
            "The suffix (-d3, -d3bj, -d3m, -d3mbj, -d4) selects the harness "
            "(s-dftd3 / dftd4). Method prefix supplies the functional-specific "
            "damping parameters."
        ),
    )

    # Periodic cell (applied to complex + bare_surface SPs; gas-phase SPs are non-periodic)
    cell: Optional[List[List[float]]] = Field(
        None,
        description=(
            "3x3 cell vectors in Angstrom applied to periodic BE evaluations. "
            "If None, the cell is read from surface.extras['cell'] on each slab "
            "record (same fallback as sampling_periodic)."
        ),
    )
    pbc: List[bool] = Field(
        [True, True, False],
        description="Periodic axes for the slab SPs (default = 2D slab).",
    )

    # Compute
    be_tag: str = Field("be_periodic_sp", description="Queue tag for the single-point energies")

    @model_validator(mode="after")
    def _validate(self):
        if not self.be_electronic_lot.is_mace:
            raise ValueError(
                "be_comp_periodic requires an MLP electronic LOT "
                "(set 'mace_model' in be_electronic_lot)."
            )
        if not self.surface_clusters:
            raise ValueError("surface_clusters must be non-empty (list of slab names).")
        if len(self.pbc) != 3:
            raise ValueError(f"pbc must be length 3 (got {len(self.pbc)}).")
        if self.cell is not None:
            if len(self.cell) != 3 or any(len(row) != 3 for row in self.cell):
                raise ValueError("cell must be a 3x3 list of lattice vectors in Angstrom.")
        return self
