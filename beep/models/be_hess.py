"""BE + Hessian workflow config — maps to launch_be_hess.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from .base import ServerConfig


class BeHessConfig(BaseModel):
    """Configuration for the binding energy and Hessian computation workflow."""
    workflow: Literal["be_hess"] = Field(..., description="Must be 'be_hess'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    molecule: str = Field(..., description="Name of the target molecule")
    surface_model_collection: str = Field("Water_22", description="Name of the surface model collection")
    small_molecule_collection: str = Field("Small_molecules", description="Name of the small molecule collection")
    level_of_theory: List[str] = Field([], description="Levels of theory for single-point energy calculations")
    exclude_clusters: List[str] = Field([], description="Cluster names to exclude from computation")
    opt_level_of_theory: str = Field(..., description="Level of theory used for geometry optimization (method_basis format)")
    keyword_id: Optional[str] = Field(None, description="QCFractal keyword ID for custom options")
    hessian_clusters: List[str] = Field([], description="Cluster names for Hessian calculations")
    program: str = Field("psi4", description="QC program to use")
    energy_tag: Optional[str] = Field(None, description="Queue tag for energy computation tasks")
    hessian_tag: Optional[str] = Field(None, description="Queue tag for Hessian computation tasks")
