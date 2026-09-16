"""Base config models shared across workflows."""
import json
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


def safe_config_dump(config) -> str:
    """JSON-serialize a workflow config without exposing credentials.

    Strips ``server.password`` and ``server.username`` before dumping.
    Every workflow writes a copy of its input config to disk for
    reproducibility; this helper ensures that copy never contains
    plaintext credentials.
    """
    return json.dumps(
        config.model_dump(exclude={"server": {"password", "username"}}),
        indent=4,
        default=str,
    )


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
    # Default is `False` because BEEP deployments typically target QCFractal
    # servers on internal HTTP-only addresses (e.g. `http://qcf-astrochem`) or
    # reverse-proxied with TLS terminated upstream — there is no client-side
    # certificate to verify. Set to `True` when the server is reachable over
    # HTTPS with a publicly-trusted certificate.
    verify: bool = Field(
        False,
        description=(
            "Verify the server's TLS certificate. Defaults to False for "
            "internal HTTP-only QCFractal deployments; set True for "
            "HTTPS servers with a trusted certificate."
        ),
    )


def split_lot_string(lot: str) -> Tuple[str, Optional[str]]:
    """Split a ``method_basis`` LOT string into ``(method, basis)``.

    Basis-less LOTs (``gfn2-xtb``, MACE model aliases) return ``basis=None``.
    Only the first underscore separates method from basis, matching the
    spec-name convention used across all workflows.
    """
    parts = lot.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else None)


def validate_mace_model_path(v):
    """Validate a MACE model file path used as a QCSpec method.

    qcportal force-lowercases QCSpec method strings server-side, so the
    path must already be all-lowercase or record lookups would silently
    miss. The file stem becomes the spec/column/dataset alias, and BEEP
    splits LOT strings on ``_``, so the stem must not contain underscores.
    """
    if v is None:
        return v
    if v != v.lower():
        raise ValueError(
            f"mace_model path must be all-lowercase (qcportal lowercases "
            f"QCSpec methods server-side): '{v}'"
        )
    stem = Path(v).stem
    if "_" in stem:
        raise ValueError(
            f"mace_model file stem must not contain underscores (BEEP "
            f"splits LOT names on '_'); use hyphens instead: '{stem}'"
        )
    return v


class LevelOfTheory(BaseModel):
    """A quantum chemistry level of theory.

    Either a conventional QC method (``method`` + optional ``basis`` on
    ``program``) or a MACE machine-learning potential via ``mace_model``.
    When ``mace_model`` is set, ``method``/``basis``/``program`` are muted:
    the spec runs as ``program='mace'`` with the model file path as method
    (stock QCEngine MACE harness), and the model file stem (e.g.
    ``mace-polar-ft0`` for ``.../mace-polar-ft0.model``) is used as the
    LOT name for specs, datasets, columns, and files.
    """
    method: Optional[str] = Field(None, description="QC method name (e.g. 'hf', 'b3lyp-d3bj'); required unless mace_model is set")
    basis: Optional[str] = Field(None, description="Basis set name (e.g. 'def2-svp')")
    program: str = Field("psi4", description="QC program to use")
    mace_model: Optional[str] = Field(
        None,
        description=(
            "Path to a serialized MACE model file (all-lowercase, stem "
            "without underscores). Mutes method/basis/program. Supported "
            "by the sampling, be_hess, and extract workflows."
        ),
    )

    _lower_method = field_validator("method")(lowercase_str)
    _lower_basis = field_validator("basis")(lowercase_str)
    _check_mace_model = field_validator("mace_model")(validate_mace_model_path)

    @model_validator(mode="after")
    def _require_method_or_mace(self):
        if self.mace_model is None and self.method is None:
            raise ValueError("LevelOfTheory requires either 'method' or 'mace_model'")
        return self

    @property
    def is_mace(self) -> bool:
        return self.mace_model is not None

    @property
    def alias(self) -> Optional[str]:
        """LOT name for a MACE model: the model file stem."""
        if self.mace_model is None:
            return None
        return Path(self.mace_model).stem

    @property
    def lot_name(self) -> str:
        """Name used for specs/datasets/columns: alias or method[_basis]."""
        if self.is_mace:
            return self.alias
        if self.basis:
            return f"{self.method}_{self.basis}".lower()
        return self.method.lower()

    @property
    def qc_method(self) -> str:
        """Method string for the QCSpecification (model path for MACE)."""
        return self.mace_model if self.is_mace else self.method

    @property
    def qc_basis(self) -> Optional[str]:
        return None if self.is_mace else self.basis

    @property
    def qc_program(self) -> str:
        return "mace" if self.is_mace else self.program

    @property
    def display(self) -> str:
        """Human-readable LOT for logs and config summaries."""
        if self.is_mace:
            return f"{self.alias} (mace)"
        return f"{self.method}/{self.basis or 'N/A'} ({self.program})"
