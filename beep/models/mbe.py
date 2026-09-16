"""MBE workflow config models — the ``mbe`` and ``mbe_extract`` workflows.

These replace the dataclass configs of the standalone beep-mbe package with
Pydantic v2 models that match the rest of BEEP (``workflow`` discriminator,
nested ``server``). The historical flat JSON shape is intentionally *not*
supported: the ``address``/``username``/``password`` fields move under
``server.*`` and ``levels`` becomes a list of objects instead of 4-element
lists.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .base import ServerConfig, lowercase_str


_BSSE_SCHEMES = {"cp", "nocp", "vmfc"}


def _coerce_spec(v):
    """Accept a bare string spec and wrap it in a list (beep-mbe allowed both)."""
    if isinstance(v, str):
        return [v]
    return v


def _validate_single_bsse(v):
    """MBE reporting requires exactly one BSSE scheme (cp, nocp, or vmfc)."""
    if isinstance(v, str):
        v = [v]
    if not v or len(v) != 1:
        raise ValueError(
            "bsse must specify exactly one scheme for reporting, e.g. ['vmfc']."
        )
    scheme = v[0].lower()
    if scheme not in _BSSE_SCHEMES:
        raise ValueError(
            f"Unknown BSSE scheme {v[0]!r}; expected one of {sorted(_BSSE_SCHEMES)}."
        )
    return [scheme]


class MbeLevel(BaseModel):
    """A single MBE level: an n-body order paired with a level of theory.

    The ``index`` is the MBE order (1 = 1-body / monomers, 2 = 2-body, ...).
    ``program`` is not carried per level — it is set once on the workflow
    config and applied to every level, matching beep-mbe semantics.
    """
    index: int = Field(..., gt=0, description="MBE order (1=1-body, 2=2-body, ...)")
    method: str = Field(..., description="QC method name (e.g. 'scf', 'ccsd(t)')")
    basis: str = Field(..., description="Basis set name (e.g. 'aug-cc-pvdz')")
    keywords: Optional[Dict[str, Any]] = Field(
        None, description="Optional QCSpecification keywords for this level"
    )

    _lower_method = field_validator("method")(lowercase_str)
    _lower_basis = field_validator("basis")(lowercase_str)


class MbeMonitorConfig(BaseModel):
    """Post-submission monitoring settings for the ``mbe`` workflow."""
    enabled: bool = Field(False, description="Poll for completion after submission")
    poll_interval: int = Field(300, gt=0, description="Seconds between status polls")
    max_wait: Optional[int] = Field(
        None, gt=0, description="Max seconds to wait before timing out (null = no limit)"
    )


class MbeZpveConfig(BaseModel):
    """Optional ZPVE correction for ``mbe_extract``, borrowed read-only from a
    prior ``be_hess`` run on the same binding sites.

    When present and ``enabled``, ``mbe_extract`` reads the Hessians produced by
    ``be_hess`` (never submits any), computes the per-site ZPVE correction, and
    emits ZPVE-corrected binding energies alongside the electronic values. Sites
    without a usable Hessian are reported as NaN.
    """
    enabled: bool = Field(True, description="Apply the ZPVE correction")
    molecule: Optional[str] = Field(
        None,
        description="Molecule name used to locate be_hess datasets; defaults to small_molecule",
    )
    hessian_clusters: Optional[List[str]] = Field(
        None,
        description="Cluster names whose be_hess Hessians to borrow; null = auto-discover be_<MOL>_* on the server",
    )
    scale_factor: float = Field(0.958, description="Harmonic ZPVE scaling factor")
    imag_threshold: float = Field(
        50.0, description="Imaginary-frequency threshold (cm^-1) below which modes are ignored"
    )


class MbeConfig(BaseModel):
    """Submit and monitor Many-Body Expansion binding-energy computations.

    Re-evaluates binding energies on existing OptimizationDataset binding sites
    at a (typically higher) level of theory via n-body fragmentation on a
    qcmanybody ManybodyDataset, plus a monomer SinglepointDataset for the
    isolated adsorbate reference.
    """
    workflow: Literal["mbe"] = Field(..., description="Must be 'mbe'")
    server: ServerConfig = Field(
        ServerConfig(), description="QCFractal server connection settings"
    )
    # --- source geometries ---
    opt_level_of_theory: str = Field(
        ..., description="Level of theory of the source geometries (method_basis format)"
    )
    opt_dataset: str = Field(
        ..., description="OptimizationDataset holding the binding-site geometries"
    )
    entries: Optional[List[str]] = Field(
        None, description="Cluster entry names to submit; null = all entries in opt_dataset"
    )
    small_molecule_collection: str = Field(
        ..., description="Collection holding the isolated small molecule"
    )
    small_molecule: str = Field(..., description="Name of the adsorbate small molecule")
    surface_model_collection: str = Field(
        ..., description="Collection holding the isolated surface model"
    )
    surface_model: str = Field(..., description="Name of the surface-model reference entry")
    # --- fragmentation ---
    env_unit_len: int = Field(
        ..., gt=0, description="Atoms per environment fragment (must divide surface atoms)"
    )
    # --- manybody dataset / specs ---
    dataset: str = Field(
        ..., description="Base ManybodyDataset name (server name gets the opt-LOT suffix)"
    )
    spec: List[str] = Field(
        ..., min_length=1, description="Specification name(s) to submit/monitor"
    )
    program: str = Field("psi4", description="QC program for the single-point evaluations")
    bsse: List[str] = Field(
        ..., description="Exactly one BSSE scheme: 'cp', 'nocp', or 'vmfc'"
    )
    tag: str = Field("mbe", description="Compute tag for the submitted tasks")
    levels: List[MbeLevel] = Field(
        ..., min_length=1, description="MBE levels (must include index 1 for the monomer)"
    )
    # --- behavior flags (were CLI flags in beep-mbe) ---
    update_existing_entries: bool = Field(
        False, description="Overwrite entries that already exist in the dataset"
    )
    fetch_only: bool = Field(
        False, description="Skip submission; only fetch/report existing records"
    )
    show_children: bool = Field(
        False, description="Log child-cluster metadata while fetching records"
    )
    monitor: MbeMonitorConfig = Field(
        MbeMonitorConfig(), description="Post-submission monitoring settings"
    )

    _lower_opt_lot = field_validator("opt_level_of_theory")(lowercase_str)
    _coerce_spec = field_validator("spec", mode="before")(_coerce_spec)
    _single_bsse = field_validator("bsse", mode="before")(_validate_single_bsse)

    @model_validator(mode="after")
    def _check_levels(self):
        indices = sorted(lvl.index for lvl in self.levels)
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate MBE level indices in levels: {indices}.")
        # Levels must be contiguous 1..N — qcmanybody expects a level for every
        # body order up to the truncation order, with no gaps.
        expected = list(range(1, len(indices) + 1))
        if indices != expected:
            raise ValueError(
                f"MBE level indices must be contiguous starting at 1 (got {indices}, "
                f"expected {expected}); every body order up to the truncation order "
                "needs its own level."
            )
        return self


class MbeExtractConfig(BaseModel):
    """Assemble MBE binding energies from completed ManybodyDataset records.

    Reads the ManybodyDataset (cluster and surface supermolecule energies) and
    the monomer SinglepointDataset, then writes per-site binding energies and
    n-body decomposition tables. Optionally applies a read-only ZPVE correction
    borrowed from a prior ``be_hess`` run (see :class:`MbeZpveConfig`).
    """
    workflow: Literal["mbe_extract"] = Field(..., description="Must be 'mbe_extract'")
    server: ServerConfig = Field(
        ServerConfig(), description="QCFractal server connection settings"
    )
    opt_level_of_theory: str = Field(
        ..., description="Level of theory of the source geometries (method_basis format)"
    )
    small_molecule_collection: str = Field(
        ..., description="Collection holding the isolated small molecule"
    )
    small_molecule: str = Field(..., description="Name of the adsorbate small molecule")
    surface_model_collection: str = Field(
        ..., description="Collection holding the isolated surface model"
    )
    surface_model: str = Field(..., description="Name of the surface-model reference entry")
    dataset: str = Field(
        ..., description="Base ManybodyDataset name (opt-LOT suffix is applied)"
    )
    entries: Optional[List[str]] = Field(
        None, description="Site entry names to extract; null = all except the surface model"
    )
    spec: List[str] = Field(
        ..., min_length=1, description="Specification name(s) to extract"
    )
    bsse: List[str] = Field(
        ..., description="Exactly one BSSE scheme: 'cp', 'nocp', or 'vmfc'"
    )
    zpve: Optional[MbeZpveConfig] = Field(
        None, description="Optional ZPVE correction borrowed from be_hess; null = electronic-only"
    )
    convergence_tol: float = Field(
        0.05,
        gt=0,
        description="Relative-error threshold for the MBE 'converged' flag (|error_bar/BE|)",
    )

    _lower_opt_lot = field_validator("opt_level_of_theory")(lowercase_str)
    _coerce_spec = field_validator("spec", mode="before")(_coerce_spec)
    _single_bsse = field_validator("bsse", mode="before")(_validate_single_bsse)
