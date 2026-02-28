"""Pre-exponential factor workflow config — maps to launch_pre_exp.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel
from .base import ServerConfig


class PreExpConfig(BaseModel):
    """Configuration for the pre-exponential factor workflow."""
    workflow: Literal["pre_exp"]
    server: ServerConfig = ServerConfig()
    molecule: Optional[List[str]] = None
    molecule_collection: str = "small_molecules"
    level_of_theory: str = "blyp_def2-svp"
    range_of_temperature: List[int] = [10, 273]
    temperature_step: int = 1
    molecule_surface_area: float = 10e-19
