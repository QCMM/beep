"""BE + Hessian workflow config — maps to launch_be_hess.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel
from .base import ServerConfig


class BeHessConfig(BaseModel):
    """Configuration for the binding energy and hessian computation workflow."""
    workflow: Literal["be_hess"]
    server: ServerConfig = ServerConfig()
    molecule: str
    surface_model_collection: str = "Water_22"
    small_molecule_collection: str = "Small_molecules"
    level_of_theory: List[str] = []
    exclude_clusters: List[str] = []
    opt_level_of_theory: str
    keyword_id: Optional[str] = None
    hessian_clusters: List[str] = []
    program: str = "psi4"
    energy_tag: Optional[str] = None
    hessian_tag: Optional[str] = None
