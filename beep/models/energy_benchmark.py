"""Energy benchmark workflow config — maps to launch_energy_benchmark.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel
from .base import ServerConfig


class EnergyBenchmarkConfig(BaseModel):
    """Configuration for the energy benchmark workflow."""
    workflow: Literal["energy_benchmark"]
    server: ServerConfig = ServerConfig()
    molecule: str
    benchmark_structures: List[str]
    small_molecule_collection: str = "Small_molecules"
    surface_model_collection: str = "small water"
    opt_level_of_theory: List[str]
    reference_be_level_of_theory: List[str] = [
        "df-ccsd(t)-f12", "cc-pvdz", "molpro"
    ]
    be_level_of_theory: List[str] = []
    cbs_level_of_theory: List[str] = []
    keyword_id: Optional[int] = None
    program: str = "psi4"
    tag_reference_geometry: Optional[str] = None
    tag_dft_geometry: Optional[str] = None
    tag_be: Optional[str] = None
    tag_cbs: Optional[str] = None
