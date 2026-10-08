"""Config for be_comp_periodic — submits periodic BE single-points on sampling_periodic outputs."""
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, model_validator
from .base import ServerConfig, LevelOfTheory


class BeCompPeriodicConfig(BaseModel):
    """Submission workflow for periodic binding energies.

    Per slab, builds one ReactionDataset per quantity (``_be``, ``_ie``, ``_de``),
    each entry carrying its site's stoichiometry, registers a range-separated
    MACE + explicit dispersion pair of periodic reaction specs and submits them.
    Assembly into per-site BE / IE / DE happens in ``be_assemble_periodic``.

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
    quantity: Literal["be", "ie", "all"] = Field(
        "be",
        description=(
            "'be': binding energies, E(complex) - E(relaxed bare surface) - E(relaxed gas "
            "adsorbate); needs the bare-surface references of sampling_periodic. "
            "'ie': interaction energies, E(complex) - E(slab) - E(adsorbate) with both "
            "fragments frozen at the complex geometry; needs no bare-surface optimizations "
            "(two single points per site), e.g. for active-learning rounds. "
            "'all': both, which also gives the deformation energy DE = BE - IE."
        ),
    )
    ie_site_filter: Literal["unique", "all"] = Field(
        "unique",
        description=(
            "Sites for quantity='ie' (no bare-surface dataset to define them): 'unique' "
            "applies sampling_periodic's periodic duplicate filter to the complete "
            "complexes, so IE and a later BE cover the same sites; 'all' keeps every "
            "complete complex. With 'all' quantity the BE sites are used."
        ),
    )
    com_tol_ang: float = Field(0.40, description="Periodic duplicate filter: adsorbate COM tolerance (A), "
                                                "as sampling_periodic's rmsd_value.")
    orientation_tol_ang: Optional[float] = Field(0.3, description="Periodic duplicate filter: height-profile "
                                                                   "tolerance (A), as sampling_periodic.")
    dataset_suffix: str = Field(
        "",
        description=(
            "Suffix of the sampling run to evaluate ('<mol>_<slab><suffix>' and its "
            "'_surface'); the reactions go to '<mol>_<slab><suffix>_be', '_ie' and '_de'. "
            "Default '' keeps the historical names."
        ),
    )
    sp_dataset_suffix: str = Field(
        "",
        description=(
            "Suffix appended to the per-slab ReactionDataset names "
            "('<mol>_<slab>_be<suffix>', '_ie<suffix>', '_de<suffix>'). "
            "Entries are keyed by site name, so geometries from a different "
            "opt_level_of_theory must go to their own datasets (e.g. '_v1'). "
            "Default '' keeps the historical names."
        ),
    )
    surface_family: Optional[str] = Field(
        None,
        description=(
            "When set (e.g. 'npASW'), the reactions of ALL slabs go to one ReactionDataset per "
            "quantity, '<mol>_<surface_family><suffix>_be' / '_ie' / '_de' (+ sp_dataset_suffix), "
            "instead of one per slab. The slabs must share the lateral cell; the non-periodic axis "
            "is padded to the largest value over the slabs so that one specification fits every "
            "entry. Entry names carry the slab ('<slab>_X..._Y...'). None: one dataset per slab."
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
