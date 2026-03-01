"""Pre-exponential factor workflow config — maps to launch_pre_exp.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from .base import ServerConfig


class PreExpConfig(BaseModel):
    """Configuration for the pre-exponential factor workflow."""
    workflow: Literal["pre_exp"] = Field(..., description="Must be 'pre_exp'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    molecule: Optional[List[str]] = Field(None, description="List of molecule names (None = all in collection)")
    molecule_collection: str = Field("small_molecules", description="Name of the molecule collection")
    level_of_theory: str = Field("blyp_def2-svp", description="Level of theory (method_basis format)")
    range_of_temperature: List[int] = Field([10, 273], description="Temperature range [min, max] in Kelvin")
    temperature_step: int = Field(1, description="Temperature step size in Kelvin")
    molecule_surface_area: float = Field(10e-19, description="Molecule surface area in m^2")
