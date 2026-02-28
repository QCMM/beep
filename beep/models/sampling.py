"""Sampling workflow config — maps to launch_sampling.py argparse flags."""
from typing import Optional, Literal
from pydantic import BaseModel
from .base import ServerConfig, LevelOfTheory


class SamplingConfig(BaseModel):
    """Configuration for the set-of-clusters sampling workflow."""
    workflow: Literal["sampling"]
    server: ServerConfig = ServerConfig()
    molecule: str
    surface_model_collection: str = "Water_22"
    small_molecule_collection: str = "Small_molecules"
    sampling_shell: float = 2.0
    sampling_condition: str = "normal"  # sparse|normal|fine|hyperfine
    sampling_level_of_theory: LevelOfTheory
    refinement_level_of_theory: LevelOfTheory
    rmsd_value: float = 0.40
    rmsd_symmetry: bool = False
    store_initial_structures: bool = False
    sampling_tag: str = "sampling"
    refinement_tag: str = "refinement"
    total_binding_sites: int = 220
    keyword_id: Optional[int] = None
