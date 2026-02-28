"""Base config models shared across workflows."""
from typing import Optional
from pydantic import BaseModel


class ServerConfig(BaseModel):
    """QCFractal server connection settings."""
    address: str = "localhost:7777"
    username: Optional[str] = None
    password: Optional[str] = None
    verify: bool = False


class LevelOfTheory(BaseModel):
    """A quantum chemistry level of theory."""
    method: str
    basis: Optional[str] = None
    program: str = "psi4"
