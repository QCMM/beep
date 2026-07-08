"""SAPT workflow configuration."""
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .base import ServerConfig, lowercase_str


EntrySelection = Union[Literal["all"], List[str], Dict[str, List[str]]]


class SaptConfig(BaseModel):
    """Configuration for symmetry-adapted perturbation theory calculations."""

    workflow: Literal["sapt"] = Field(..., description="Must be 'sapt'")
    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="QCFractal server connection settings",
    )
    molecule: str = Field(..., min_length=1, description="Name of the target adsorbate")
    surface_model: str = Field(
        ...,
        min_length=1,
        description="OptimizationDataset containing the surface cluster names",
    )
    optimization_spec: str = Field(
        ...,
        min_length=1,
        description="Specification used for the optimized binding-site structures",
    )
    entries: EntrySelection = Field(
        "all",
        description=(
            "Binding-site entries to process: 'all', a global entry list, "
            "or a mapping from cluster name to entry list"
        ),
    )
    exclude_clusters: List[str] = Field(
        default_factory=list,
        description="Surface cluster names to exclude",
    )

    method: str = Field("sapt0", min_length=1, description="SAPT energy method")
    basis: str = Field("jun-cc-pvdz", min_length=1, description="Orbital basis set")
    program: str = Field("psi4", min_length=1, description="QC program")
    keywords: Dict[str, object] = Field(
        default_factory=lambda: {
            "scf_type": "df",
            "freeze_core": True,
            "guess": "sad",
        },
        description="Program-specific keywords for the SAPT calculation",
    )
    sapt_tag: Optional[str] = Field(
        "sapt",
        description="QCFractal compute tag for SAPT tasks",
    )
    dry_run: bool = Field(
        True,
        description=(
            "If true, build fragmented SAPT entries and write the plan report "
            "without creating/submitting QCFractal records"
        ),
    )
    wait_for_completion: bool = Field(
        False,
        description="If true, poll submitted SAPT records until they finish",
    )
    wait_interval: int = Field(
        600,
        ge=1,
        description="Polling interval in seconds when wait_for_completion is true",
    )

    surface_charge: int = Field(0, description="Charge of the surface fragment")
    surface_multiplicity: int = Field(
        1,
        ge=1,
        description="Spin multiplicity of the surface fragment",
    )
    molecule_charge: int = Field(0, description="Charge of the adsorbate fragment")
    molecule_multiplicity: int = Field(
        1,
        ge=1,
        description="Spin multiplicity of the adsorbate fragment",
    )

    _lower_optimization_spec = field_validator("optimization_spec")(lowercase_str)
    _lower_method = field_validator("method")(lowercase_str)
    _lower_basis = field_validator("basis")(lowercase_str)
    _lower_program = field_validator("program")(lowercase_str)
