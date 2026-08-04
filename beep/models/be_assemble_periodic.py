"""Config for be_assemble_periodic — extracts per-site periodic BEs from be_comp_periodic output."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, model_validator
from .base import ServerConfig, LevelOfTheory


class BeAssemblePeriodicConfig(BaseModel):
    """Extraction workflow for periodic binding energies.

    Fetches the paired MACE electronic + explicit dispersion single-point
    energies submitted by ``be_comp_periodic``, sums them to composite BE
    energies, and computes per site::

        BE = E(complex) - E(bare_site) - E(adsorbate_gas)

    optionally shifting by a per-adsorbate ZPVE-correction scalar
    (``zpve_correction_kcal_mol``). Writes per-slab + aggregated CSV
    outputs under ``<molecule>/data/``.

    Only sites COMPLETE in all three record sets (complex SP + bare-surface
    SP + gas-phase SP) yield a BE; missing/errored records are logged and
    the site is skipped in the CSV.
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
