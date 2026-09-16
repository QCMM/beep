"""Geometry benchmark workflow config — entry-based, single OptimizationDataset."""
from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field, field_validator
from .base import ServerConfig


class GeomBenchmarkConfig(BaseModel):
    """Configuration for the geometry benchmark workflow.

    Entry-based: every geometry to benchmark is an entry in a single
    ``opt_dataset`` (OptimizationDataset). Each listed entry is optimized
    at the reference LOT and at every DFT functional in the curated
    lists, then compared by RMSD (and optionally per-step force RMSD
    along the reference trajectory). There is no adsorbate/surface
    special-casing — to benchmark a monomer or bare surface alongside
    the complexes, add it as another entry in ``opt_dataset``.
    """
    workflow: Literal["geom_benchmark"] = Field(..., description="Must be 'geom_benchmark'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    opt_dataset: str = Field(
        ...,
        description=(
            "Name of the OptimizationDataset containing every entry in "
            "``benchmark_structures``. Entry names are free-form labels."
        ),
    )
    benchmark_structures: List[str] = Field(
        ...,
        description=(
            "Entry names within ``opt_dataset`` to benchmark. Each stored "
            "molecule carries its own charge/multiplicity — no global "
            "adsorbate multiplicity is assumed."
        ),
    )
    reference_geometry_level_of_theory: List[str] = Field(
        ["ccsd(t)", "aug-cc-pvtz", "psi4"],
        description="Reference geometry level of theory [method, basis, program]",
    )
    reference_geometry_keywords: Optional[Dict[str, str]] = Field(
        {"scf_type": "df", "cc_type": "df", "freeze_core": "true"},
        description="QC keywords for reference geometry (e.g. scf_type, cc_type)",
    )
    tag_reference_geometry: Optional[str] = Field(None, description="Queue tag for reference geometry tasks")
    dft_optimization_program: str = Field("psi4", description="Program for DFT geometry optimizations")
    dft_optimization_keyword: Optional[int] = Field(None, description="QCFractal keyword ID for DFT optimizations")
    tag_dft_geometry: Optional[str] = Field(None, description="Queue tag for DFT geometry tasks")
    use_initial_reference_geometry: bool = Field(False, description="Use initial (unoptimized) reference geometry")
    trajectory_analysis: bool = Field(
        True,
        description=(
            "If True (default), evaluate each DFT functional via SP+gradient "
            "at every geometry along the reference optimization trajectory "
            "and report RMSD of the per-component force (meV/Å) vs the "
            "reference. Combined with the equilibrium-geometry RMSD via a "
            "z-score-weighted score (see ``score_weights``). Absolute "
            "energies are not used here — use the energy_benchmark workflow "
            "for relative-energy comparison. Set to False to keep the "
            "legacy eq-geometry-only behaviour."
        ),
    )
    score_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "rmsd_eq": 1.0,
            "rmsd_force": 1.0,
        },
        description=(
            "Weights for the combined z-score ranking when "
            "trajectory_analysis is enabled. Keys: rmsd_eq, rmsd_force. "
            "Default: equal weighting."
        ),
    )
    generate_plots: bool = Field(
        False,
        description=(
            "If True, also produce eq-RMSD histograms and (when "
            "trajectory_analysis is enabled) trajectory force-error "
            "histograms + violin plot as SVG under data/plots/. Off by "
            "default — JSON results are always written."
        ),
    )

    @field_validator("reference_geometry_level_of_theory")
    @classmethod
    def _lower_ref_geom_lot(cls, v):
        """Lowercase method (index 0) and basis (index 1); leave program (index 2) as-is."""
        if not isinstance(v, list):
            return v
        out = list(v)
        for i in (0, 1):
            if i < len(out) and isinstance(out[i], str) and out[i]:
                out[i] = out[i].lower()
        return out
