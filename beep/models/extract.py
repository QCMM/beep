"""Extract BE data workflow config — maps to launch_extract_be_data.py argparse flags."""
from typing import Optional, Literal, List, Tuple
from pydantic import BaseModel
from .base import ServerConfig


class ExtractConfig(BaseModel):
    """Configuration for the binding energy data extraction workflow."""
    workflow: Literal["extract"]
    server: ServerConfig = ServerConfig()
    opt_method: str = "mpwb1k-d3bj_def2-tzvp"
    be_methods: List[str] = ["WB97X-V", "M06-HF", "WPBE-D3MBJ"]
    mol_coll_name: str
    surface_model: str
    hessian_clusters: List[str] = []
    molecules: List[str] = []
    be_range: List[float] = [-0.1, -25.0]
    scale_factor: float = 0.958
    basis: str = "def2-tzvp"
    exclude_clusters: List[str] = []
    no_zpve: bool = False
    generate_plots: bool = False
