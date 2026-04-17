"""Energy benchmark workflow config — maps to launch_energy_benchmark.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator
from .base import ServerConfig, uppercase_str, uppercase_list


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
        "CCSD(T)_AUG-CC-PVTZ",
        description="Specification name (method_basis) of the reference geometry optimization. The structures optimized at this level are used to compute the CCSD(T)/CBS reference binding energy.",
    )
    be_level_of_theory: List[str] = Field([], description="Levels of theory for BE single-point calculations")
    cbs_level_of_theory: List[str] = Field([], description="Levels of theory for CBS extrapolation")
    keyword_id: Optional[int] = Field(None, description="QCFractal keyword ID for custom options")
    program: str = Field("psi4", description="QC program to use")
    be_basis: str = Field("DEF2-TZVPD", description="Basis set for DFT binding energy single-point calculations")
    tag_reference_geometry: Optional[str] = Field(None, description="Queue tag for reference geometry tasks")
    tag_dft_geometry: Optional[str] = Field(None, description="Queue tag for DFT geometry tasks")
    tag_be: str = Field(..., description="Queue tag for binding energy tasks")
    tag_cbs: str = Field(..., description="Queue tag for CBS extrapolation tasks")
    use_initial_reference_geometry: bool = Field(False, description="Use initial (unoptimized) reference geometry")
    custom_dft_functionals: List[str] = Field([], description="Additional DFT functionals to include in the benchmark (e.g. ['RPBE-D4', 'BLYP-D4'])")

    _upper_opt_lot = field_validator("opt_level_of_theory")(uppercase_list)
    _upper_ref_lot = field_validator("reference_geometry_level_of_theory")(uppercase_str)
    _upper_be_lot = field_validator("be_level_of_theory")(uppercase_list)
    _upper_cbs_lot = field_validator("cbs_level_of_theory")(uppercase_list)
    _upper_be_basis = field_validator("be_basis")(uppercase_str)
    _upper_custom_func = field_validator("custom_dft_functionals")(uppercase_list)
