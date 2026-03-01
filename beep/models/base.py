"""Base config models shared across workflows."""
from typing import Optional
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """QCFractal server connection settings."""
    address: str = Field("localhost:7777", description="Server address (host:port)")
    username: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    verify: bool = Field(False, description="Verify SSL certificate")


class LevelOfTheory(BaseModel):
    """A quantum chemistry level of theory."""
    method: str = Field(..., description="QC method name (e.g. 'HF', 'B3LYP-D3BJ')")
    basis: Optional[str] = Field(None, description="Basis set name (e.g. 'def2-svp')")
    program: str = Field("psi4", description="QC program to use")
