"""BE + Hessian workflow config — maps to launch_be_hess.py argparse flags."""
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator
from .base import ServerConfig, lowercase_str, lowercase_list, validate_mace_model_path


class BeHessConfig(BaseModel):
    """Configuration for the binding energy and Hessian computation workflow."""
    workflow: Literal["be_hess"] = Field(..., description="Must be 'be_hess'")
    server: ServerConfig = Field(ServerConfig(), description="QCFractal server connection settings")
    molecule: str = Field(..., description="Name of the target molecule")
    surface_model_collection: str = Field("Water_22", description="Name of the surface model collection")
    small_molecule_collection: str = Field("Small_molecules", description="Name of the small molecule collection")
    atoms_collection: str = Field("atoms", description="Name of the SinglepointDataset containing atomic species")
    level_of_theory: List[str] = Field([], description="Levels of theory for single-point energy calculations")
    mace_models: List[str] = Field(
        [],
        description=(
            "Paths to serialized MACE model files for BE single-point "
            "evaluation (stock QCEngine 'mace' harness). Each spec/column "
            "is named by the file stem. The bsse (counterpoise) "
            "stoichiometry is skipped for these — MLPs have no basis "
            "functions, and ghost atoms would be treated as real atoms."
        ),
    )
    exclude_clusters: List[str] = Field([], description="Cluster names to exclude from computation")
    opt_level_of_theory: str = Field(..., description="Level of theory used for geometry optimization (method_basis format)")
    keyword_id: Optional[str] = Field(None, description="QCFractal keyword ID for custom options")
    hessian_clusters: List[str] = Field([], description="Cluster names for Hessian calculations")
    program: str = Field("psi4", description="QC program to use")
    energy_tag: Optional[str] = Field(None, description="Queue tag for energy computation tasks")
    hessian_tag: Optional[str] = Field(None, description="Queue tag for Hessian computation tasks")

    _lower_lot = field_validator("level_of_theory")(lowercase_list)
    _lower_opt_lot = field_validator("opt_level_of_theory")(lowercase_str)

    @field_validator("mace_models")
    @classmethod
    def _check_mace_models(cls, v):
        return [validate_mace_model_path(p) for p in v]
