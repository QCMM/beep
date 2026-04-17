"""Base config models shared across workflows."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


def uppercase_str(v):
    """Normalize a method/basis string to uppercase. Skips None and empty."""
    if v is None or not isinstance(v, str) or not v:
        return v
    return v.upper()


def uppercase_list(v):
    """Normalize a list of method/basis strings to uppercase (per-item)."""
    if v is None:
        return v
    return [s.upper() if isinstance(s, str) and s else s for s in v]


class ServerConfig(BaseModel):
    """QCFractal server connection settings."""
    address: str = Field("localhost:7777", description="Server address (host:port)")
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    verify: bool = Field(False, description="Verify SSL certificate")


class LevelOfTheory(BaseModel):
    """A quantum chemistry level of theory."""
    method: str = Field(..., description="QC method name (e.g. 'HF', 'B3LYP-D3BJ')")
    basis: Optional[str] = Field(None, description="Basis set name (e.g. 'DEF2-SVP')")
    program: str = Field("psi4", description="QC program to use")

    _upper_method = field_validator("method")(uppercase_str)
    _upper_basis = field_validator("basis")(uppercase_str)
