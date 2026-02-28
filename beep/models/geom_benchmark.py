"""Geometry benchmark workflow config — maps to launch_geom_benchmark.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel
from .base import ServerConfig


class GeomBenchmarkConfig(BaseModel):
    """Configuration for the geometry benchmark workflow."""
    workflow: Literal["geom_benchmark"]
    server: ServerConfig = ServerConfig()
    molecule: str
    benchmark_structures: List[str]
    small_molecule_collection: str = "Small_molecules"
    surface_model_collection: str = "small water"
    reference_geometry_level_of_theory: List[str] = [
        "df-ccsd(t)-f12", "cc-pvdz", "molpro"
    ]
    tag_reference_geometry: Optional[str] = None
    dft_optimization_program: str = "psi4"
    dft_optimization_keyword: Optional[int] = None
    tag_dft_geometry: Optional[str] = None
    use_initial_reference_geometry: bool = False
