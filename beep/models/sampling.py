"""Sampling workflow config — maps to launch_sampling.py argparse flags."""
from typing import Optional, Literal
from pydantic import BaseModel, Field
from .base import ServerConfig, LevelOfTheory


class SamplingConfig(BaseModel):
    """Configuration for the set-of-clusters sampling workflow."""
    workflow: Literal["sampling"] = Field(..., description="Must be 'sampling'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    molecule: str = Field(..., description="Name of the target molecule in the small molecule collection")
    surface_model_collection: str = Field("Water_22", description="Name of the surface model collection")
    small_molecule_collection: str = Field("Small_molecules", description="Name of the small molecule collection")
    sampling_shell: float = Field(2.0, description="Radius of the sampling shell in Angstrom")
    sampling_condition: str = Field("normal", description="Sampling density: sparse|normal|fine|hyperfine")
    sampling_level_of_theory: LevelOfTheory = Field(..., description="Level of theory for initial sampling optimizations")
    refinement_level_of_theory: LevelOfTheory = Field(..., description="Level of theory for refinement optimizations")
    rmsd_value: float = Field(0.40, description="RMSD threshold in Angstrom for filtering duplicate structures")
    rmsd_symmetry: bool = Field(False, description="Account for molecular symmetry in RMSD comparison")
    store_initial_structures: bool = Field(False, description="Store initial (pre-optimization) structures")
    sampling_tag: str = Field("sampling", description="Queue tag for sampling computation tasks")
    refinement_tag: str = Field("refinement", description="Queue tag for refinement computation tasks")
    total_binding_sites: int = Field(220, description="Total number of binding sites to generate")
    keyword_id: Optional[int] = Field(None, description="QCFractal keyword ID for custom options")
