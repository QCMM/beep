"""Geometry benchmark workflow config — maps to launch_geom_benchmark.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from .base import ServerConfig


class GeomBenchmarkConfig(BaseModel):
    """Configuration for the geometry benchmark workflow."""
    workflow: Literal["geom_benchmark"] = Field(..., description="Must be 'geom_benchmark'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    molecule: str = Field(..., description="Name of the target molecule")
    benchmark_structures: List[str] = Field(..., description="List of benchmark structure identifiers")
    small_molecule_collection: str = Field("Small_molecules", description="Name of the small molecule collection")
    surface_model_collection: str = Field("small water", description="Name of the surface model collection")
    reference_geometry_level_of_theory: List[str] = Field(
        ["df-ccsd(t)-f12", "cc-pvdz", "molpro"],
        description="Reference geometry level of theory [method, basis, program]",
    )
    tag_reference_geometry: Optional[str] = Field(None, description="Queue tag for reference geometry tasks")
    dft_optimization_program: str = Field("psi4", description="Program for DFT geometry optimizations")
    dft_optimization_keyword: Optional[int] = Field(None, description="QCFractal keyword ID for DFT optimizations")
    tag_dft_geometry: Optional[str] = Field(None, description="Queue tag for DFT geometry tasks")
    use_initial_reference_geometry: bool = Field(False, description="Use initial (unoptimized) reference geometry")
