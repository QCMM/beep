"""Energy benchmark workflow config — maps to launch_energy_benchmark.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator
from .base import ServerConfig, lowercase_str, lowercase_list


class EnergyBenchmarkConfig(BaseModel):
    """Configuration for the energy benchmark workflow."""
    workflow: Literal["energy_benchmark"] = Field(..., description="Must be 'energy_benchmark'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    molecule: str = Field(..., description="Name of the target molecule")
    benchmark_structures: List[str] = Field(..., description="List of benchmark structure identifiers")
    small_molecule_collection: str = Field("Small_molecules", description="Name of the small molecule collection")
    surface_model_collection: str = Field("small water", description="Name of the surface model collection")
    opt_level_of_theory: List[str] = Field(..., description="DFT geometry optimization levels of theory (list of method_basis strings, e.g. ['MPWB1K-D3BJ_DEF2-TZVPD'])")
    reference_geometry_level_of_theory: str = Field(
        "ccsd(t)_aug-cc-pvtz",
        description="Specification name (method_basis) of the reference geometry optimization. The structures optimized at this level are used to compute the CCSD(T)/CBS reference binding energy.",
    )
    be_level_of_theory: List[str] = Field([], description="Levels of theory for BE single-point calculations")
    cbs_level_of_theory: List[str] = Field([], description="Levels of theory for CBS extrapolation")
    keyword_id: Optional[int] = Field(None, description="QCFractal keyword ID for custom options")
    program: str = Field("psi4", description="QC program to use")
    be_basis: str = Field("def2-tzvpd", description="Basis set for DFT binding energy single-point calculations")
    tag_reference_geometry: Optional[str] = Field(None, description="Queue tag for reference geometry tasks")
    tag_dft_geometry: Optional[str] = Field(None, description="Queue tag for DFT geometry tasks")
    tag_be: str = Field(..., description="Queue tag for binding energy tasks")
    tag_cbs: str = Field(..., description="Queue tag for CBS extrapolation tasks")
    use_initial_reference_geometry: bool = Field(False, description="Use initial (unoptimized) reference geometry")
    custom_dft_functionals: List[str] = Field([], description="Additional DFT functionals to include in the benchmark (e.g. ['RPBE-D4', 'BLYP-D4'])")
    gcp_correction: bool = Field(
        False,
        description=(
            "If True, submit a standalone gCP (Kruse & Grimme 2012) "
            "correction at dft/def2-tzvp for every unique molecule used in "
            "the binding-energy stoichiometries. The workflow combines it "
            "post-hoc with the bare DFT energies at extract time to "
            "produce a gCP-corrected BE column. Applied only when "
            "be_basis == def2-tzvp and only to functionals listed in "
            "core.dft_functionals.gcp_compatible_functionals() — -3c "
            "composites (gCP already baked in), double hybrids (gCP not "
            "parametrized), and HF-D3BJ (separate gCP parameter set) are "
            "skipped. For any other be_basis the workflow logs a warning "
            "and silently falls back to the no-gCP path."
        ),
    )
    generate_plots: bool = Field(
        False,
        description=(
            "If True, produce BE violin, density panel, mean-error and "
            "IE-vs-DE plots as SVG under data/plots/. Off by default — "
            "JSON results (BE/IE/DE + AE/RE variants) are always written."
        ),
    )
    functional_averages: List[List[str]] = Field(
        default_factory=list,
        description=(
            "Optional list of functional groups to average. Each group is "
            "a list of LOT strings ('method_basis'). For every group, "
            "the workflow computes the mean BE per binding site across "
            "the listed functionals and reports it as one extra row in "
            "the BE per-group MAE table, labelled 'DFT_Average_N' "
            "(1-indexed in config order). Members not present in the "
            "benchmark set are dropped with a warning; groups that end "
            "up empty are skipped. Empty default — configs without the "
            "field work unchanged."
        ),
    )

    _lower_opt_lot = field_validator("opt_level_of_theory")(lowercase_list)
    _lower_ref_lot = field_validator("reference_geometry_level_of_theory")(lowercase_str)
    _lower_be_lot = field_validator("be_level_of_theory")(lowercase_list)
    _lower_cbs_lot = field_validator("cbs_level_of_theory")(lowercase_list)
    _lower_be_basis = field_validator("be_basis")(lowercase_str)
    _lower_custom_func = field_validator("custom_dft_functionals")(lowercase_list)
