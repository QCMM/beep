"""Energy benchmark workflow config — maps to launch_energy_benchmark.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from .base import ServerConfig


class EnergyBenchmarkConfig(BaseModel):
    """Configuration for the energy benchmark workflow."""
    workflow: Literal["energy_benchmark"] = Field(..., description="Must be 'energy_benchmark'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    molecule: str = Field(..., description="Name of the target molecule")
    benchmark_structures: List[str] = Field(..., description="List of benchmark structure identifiers")
    small_molecule_collection: str = Field("Small_molecules", description="Name of the small molecule collection")
    surface_model_collection: str = Field("small water", description="Name of the surface model collection")
    opt_level_of_theory: List[str] = Field(..., description="Optimization level of theory [method, basis, program]")
    reference_be_level_of_theory: List[str] = Field(
        ["df-ccsd(t)-f12", "cc-pvdz", "molpro"],
        description="Reference binding energy level of theory [method, basis, program]",
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
