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
    mace_dispersion: Optional[str] = Field(
        None,
        description=(
            "Optional analytic dispersion paired with the MACE electronic "
            "models for range separation, e.g. 'mpwb1k-d4' or 'mpwb1k-d3bj'. "
            "When set, each MLP is treated as electronic-only and be_hess also "
            "computes the dispersion (dftd4 / s-dftd3, functional params from "
            "the prefix) as a separate spec; extract sums them into the "
            "composite MLP+dispersion BE, exactly as for DFT-D."
        ),
    )
    exclude_clusters: List[str] = Field([], description="Cluster names to exclude from computation")
    opt_level_of_theory: str = Field(..., description="Level of theory used for geometry optimization (method_basis format)")
    keyword_id: Optional[str] = Field(None, description="QCFractal keyword ID for custom options")
    hessian_clusters: List[str] = Field([], description="Cluster names for Hessian calculations")
    program: str = Field("psi4", description="QC program to use")
    energy_tag: Optional[str] = Field(None, description="Queue tag for energy computation tasks")
    dispersion_tag: Optional[str] = Field(
        None,
        description=(
            "Queue tag for the analytic dispersion single-points when "
            "'mace_dispersion' is set. Defaults to 'energy_tag'. Use a "
            "separate tag to route dispersion (dftd4/s-dftd3, CPU-only) to a "
            "CPU manager while the electronic MLP energies run on a GPU "
            "manager under 'energy_tag'."
        ),
    )
    hessian_tag: Optional[str] = Field(None, description="Queue tag for Hessian computation tasks")

    _lower_lot = field_validator("level_of_theory")(lowercase_list)
    _lower_opt_lot = field_validator("opt_level_of_theory")(lowercase_str)

    @field_validator("mace_models")
    @classmethod
    def _check_mace_models(cls, v):
        return [validate_mace_model_path(p) for p in v]

    @field_validator("mace_dispersion")
    @classmethod
    def _check_mace_dispersion(cls, v):
        if v is None:
            return v
        v = v.lower()
        # Lazy import: single source of truth for recognized dispersion suffixes.
        from ..adapters.qcfractal_adapter import _has_dispersion_suffix
        if not _has_dispersion_suffix(v):
            raise ValueError(
                f"mace_dispersion '{v}' must end with a known dispersion suffix "
                "(e.g. -d4, -d3bj) and carry a functional prefix, e.g. 'mpwb1k-d4'."
            )
        return v
