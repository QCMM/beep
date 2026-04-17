"""Extract BE data workflow config — maps to launch_extract_be_data.py argparse flags."""
from typing import Optional, Literal, List, Tuple
from pydantic import BaseModel, Field, field_validator
from .base import ServerConfig, uppercase_str, uppercase_list


class ExtractConfig(BaseModel):
    """Configuration for the binding energy data extraction workflow."""
    workflow: Literal["extract"] = Field(..., description="Must be 'extract'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    opt_method: str = Field("MPWB1K-D3BJ_DEF2-TZVP", description="Optimization method_basis identifier")
    be_methods: List[str] = Field(["WB97X-V", "M06-HF", "WPBE-D3MBJ"], description="Methods for binding energy evaluation")
    surface_model: str = Field(..., description="Name of the surface model")
    hessian_clusters: List[str] = Field([], description="Cluster names with Hessian data for ZPVE correction")
    molecules: List[str] = Field(..., description="Molecules to extract binding energies for")
    be_range: List[float] = Field([-0.1, -25.0], description="Binding energy range filter [max, min] in kcal/mol")
    scale_factor: float = Field(0.958, description="ZPVE scale factor for the level of theory")
    basis: str = Field("DEF2-TZVP", description="Basis set used for binding energy calculations")
    exclude_clusters: List[str] = Field([], description="Cluster names to exclude from extraction")
    stoichiometry: str = Field("bsse", description="Stoichiometry type for BE extraction: bsse (BSSE-corrected), be_nocp, ie, or de")
    no_zpve: bool = Field(False, description="Skip ZPVE correction")
    imag_threshold: float = Field(50.0, description="Imaginary frequencies below this value (cm⁻¹) are ignored in ZPVE. Default 50 cm⁻¹")
    generate_plots: bool = Field(False, description="Generate binding energy distribution plots")

    _upper_opt_method = field_validator("opt_method")(uppercase_str)
    _upper_basis = field_validator("basis")(uppercase_str)
    _upper_be_methods = field_validator("be_methods")(uppercase_list)
