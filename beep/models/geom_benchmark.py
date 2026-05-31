"""Geometry benchmark workflow config — maps to launch_geom_benchmark.py argparse flags."""
from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field, field_validator
from .base import ServerConfig, lowercase_list


class BSSETestConfig(BaseModel):
    """Configuration for BSSE/dispersion test via direct Slurm CP jobs."""
    functional: List[str] = Field(["pbe0"], description="Test functional(s) (default: ['pbe0'])")
    basis_sets: List[str] = Field(
        ["def2-svp", "def2-svpd", "def2-tzvpd"],
        description="Basis sets to test",
    )
    dispersion: List[str] = Field(
        ["", "d3bj", "d4"],
        description="Dispersion corrections for uncorrected optimizations (empty string = bare functional)",
    )
    cp_dispersion: Optional[List[str]] = Field(
        None,
        description="Dispersion corrections for CP-corrected optimizations. Defaults to same as dispersion. D4 is not supported with CP in Psi4.",
    )
    partition: str = Field(..., description="Slurm partition for CP jobs")
    cores: int = Field(6, description="Number of CPU cores per CP job")
    memory: str = Field("12GB", description="Memory per CP job")
    walltime: str = Field("24:00:00", description="Slurm walltime for CP jobs")

    _lower_functional = field_validator("functional")(lowercase_list)
    _lower_basis_sets = field_validator("basis_sets")(lowercase_list)
    _lower_dispersion = field_validator("dispersion")(lowercase_list)
    _lower_cp_dispersion = field_validator("cp_dispersion")(lowercase_list)


class GeomBenchmarkConfig(BaseModel):
    """Configuration for the geometry benchmark workflow."""
    workflow: Literal["geom_benchmark"] = Field(..., description="Must be 'geom_benchmark'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    molecule: str = Field(..., description="Name of the target molecule")
    benchmark_structures: List[str] = Field(..., description="List of benchmark structure identifiers")
    small_molecule_collection: str = Field("Small_molecules", description="Name of the small molecule collection")
    surface_model_collection: str = Field("small water", description="Name of the surface model collection")
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
    bsse_test: Optional[BSSETestConfig] = Field(None, description="Optional BSSE/dispersion test on a single functional")
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
