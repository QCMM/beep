"""Base config models shared across workflows."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


def lowercase_str(v):
    """Normalize a method/basis string to lowercase. Skips None and empty.

    BEEP standardised on lowercase QCSpec names after the qcportal 0.63
    migration; the server stores specs lowercase and case-sensitive lookups
    fail otherwise.
    """
    if v is None or not isinstance(v, str) or not v:
        return v
    return v.lower()


def lowercase_list(v):
    """Normalize a list of method/basis strings to lowercase (per-item)."""
    if v is None:
        return v
    return [s.lower() if isinstance(s, str) and s else s for s in v]


class ServerConfig(BaseModel):
    """QCFractal server connection settings."""
    address: str = Field("localhost:7777", description="Server address (host:port)")
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    verify: bool = Field(False, description="Verify SSL certificate")


class LevelOfTheory(BaseModel):
    """A quantum chemistry level of theory."""
    method: str = Field(..., description="QC method name (e.g. 'hf', 'b3lyp-d3bj')")
    basis: Optional[str] = Field(None, description="Basis set name (e.g. 'def2-svp')")
    program: str = Field("psi4", description="QC program to use")

    _lower_method = field_validator("method")(lowercase_str)
    _lower_basis = field_validator("basis")(lowercase_str)
