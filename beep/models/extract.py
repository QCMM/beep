"""Extract BE data workflow config — maps to launch_extract_be_data.py argparse flags."""
from typing import Optional, Literal, List, Tuple
from pydantic import BaseModel, Field
from .base import ServerConfig


class ExtractConfig(BaseModel):
    """Configuration for the binding energy data extraction workflow."""
    workflow: Literal["extract"] = Field(..., description="Must be 'extract'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    opt_method: str = Field("mpwb1k-d3bj_def2-tzvp", description="Optimization method_basis identifier")
    be_methods: List[str] = Field(["WB97X-V", "M06-HF", "WPBE-D3MBJ"], description="Methods for binding energy evaluation")
    mol_coll_name: str = Field(..., description="Name of the molecule collection")
    surface_model: str = Field(..., description="Name of the surface model")
    hessian_clusters: List[str] = Field([], description="Cluster names with Hessian data for ZPVE correction")
    molecules: List[str] = Field([], description="Subset of molecules to extract (empty = all)")
    be_range: List[float] = Field([-0.1, -25.0], description="Binding energy range filter [max, min] in kcal/mol")
    scale_factor: float = Field(0.958, description="ZPVE scale factor for the level of theory")
    basis: str = Field("def2-tzvp", description="Basis set used for binding energy calculations")
    exclude_clusters: List[str] = Field([], description="Cluster names to exclude from extraction")
    no_zpve: bool = Field(False, description="Skip ZPVE correction")
    generate_plots: bool = Field(False, description="Generate binding energy distribution plots")
