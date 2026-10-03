"""Config for be_assemble_periodic — extracts per-site periodic BEs from be_comp_periodic output."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, model_validator
from .base import ServerConfig, LevelOfTheory


class BeAssemblePeriodicConfig(BaseModel):
    """Extraction workflow for periodic binding energies.

    Fetches the paired MACE electronic + explicit dispersion reaction records
    submitted by ``be_comp_periodic`` and sums them per site, e.g.::

        BE = E(complex) - E(bare_site) - E(adsorbate_gas)

    optionally shifting by a per-adsorbate ZPVE-correction scalar
    (``zpve_correction_kcal_mol``). Writes per-slab + aggregated CSV
    outputs under ``<molecule>/data/``.

    Only sites whose reaction records (and their components) are COMPLETE
    yield a value; missing/errored records are logged and the site is
    skipped in the CSV.
    """
    workflow: Literal["be_assemble_periodic"] = Field(..., description="Must be 'be_assemble_periodic'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")

    # Adsorbate + slab lookup — must match the be_comp_periodic run
    molecule: str = Field(..., description="Adsorbate name (must match be_comp_periodic)")
    surface_clusters: List[str] = Field(..., description="Slab names to assemble (must be non-empty)")

    # BE LOT — must match be_comp_periodic so we know which specs to fetch
    be_electronic_lot: LevelOfTheory = Field(
        ..., description="MACE electronic LOT (same as be_comp_periodic)"
    )
    be_dispersion: str = Field(
        ..., description="Dispersion method with suffix (same as be_comp_periodic, e.g. 'mpwb1k-d4')"
    )

    # ZPVE correction (per-adsorbate, in kcal/mol)
    zpve_correction_kcal_mol: float = Field(
        0.0,
        description=(
            "Scalar shift added to every site's BE to account for zero-point "
            "vibrational energy. Compute once for the adsorbate (e.g. from a "
            "gas-phase Hessian at a comparable LOT) and pass it in here — "
            "kept out of the workflow so it doesn't need periodic Hessians."
        ),
    )

    # Output
    dataset_suffix: str = Field(
        "",
        description=(
            "Suffix of the evaluated sampling run, as passed to be_comp_periodic. "
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
    output_prefix: str = Field(
        "be_periodic",
        description="Prefix for output CSV filenames (`<prefix>_<slab>.csv`, `<prefix>_summary.csv`).",
    )

    @model_validator(mode="after")
    def _validate(self):
        if not self.be_electronic_lot.is_mace:
            raise ValueError(
                "be_assemble_periodic requires an MLP electronic LOT "
                "(set 'mace_model' in be_electronic_lot)."
            )
        if not self.surface_clusters:
            raise ValueError("surface_clusters must be non-empty (list of slab names).")
        return self
