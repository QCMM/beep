"""Pre-exponential factor workflow config — maps to launch_pre_exp.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator
from .base import ServerConfig, lowercase_str


class PreExpConfig(BaseModel):
    """Configuration for the pre-exponential factor workflow."""
    workflow: Literal["pre_exp"] = Field(..., description="Must be 'pre_exp'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    molecule: Optional[List[str]] = Field(None, description="List of molecule names (None or [] = all in collection)")
    molecule_collection: str = Field("small_molecules", description="Name of the molecule collection")
    level_of_theory: str = Field("blyp_def2-svp", description="Level of theory (method_basis format)")
    range_of_temperature: List[int] = Field([10, 273], description="Temperature range [min, max] in Kelvin (inclusive of max); a single-element list [T] evaluates one temperature")
    temperature_step: int = Field(1, description="Temperature step size in Kelvin")
    molecule_surface_area: float = Field(
        1e-19,
        description=(
            "Surface area per adsorbed molecule in m^2 (inverse of the site density); "
            "1e-19 m^2 (10 A^2) for most small molecules (Minissale et al. 2022)."
        ),
    )

    _lower_lot = field_validator("level_of_theory")(lowercase_str)

    @field_validator("range_of_temperature")
    @classmethod
    def _check_temperature_range(cls, v):
        if len(v) not in (1, 2):
            raise ValueError(
                "range_of_temperature must be [T_min, T_max] (or [T] for a "
                f"single temperature); got {len(v)} entries: {v}"
            )
        if len(v) == 2 and v[0] > v[1]:
            raise ValueError(
                f"range_of_temperature must satisfy T_min <= T_max; got {v}"
            )
        if any(t <= 0 for t in v):
            raise ValueError(f"range_of_temperature entries must be positive Kelvin; got {v}")
        return v

    @field_validator("temperature_step")
    @classmethod
    def _check_temperature_step(cls, v):
        if v <= 0:
            raise ValueError(f"temperature_step must be a positive integer; got {v}")
        return v
