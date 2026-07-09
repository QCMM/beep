"""Tests for beep/models/mbe.py (MbeConfig / MbeExtractConfig validation).

Ports beep-mbe's test_config.py + test_levels.py to the Pydantic-v2 schema.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from beep.models import MbeConfig, MbeExtractConfig
from beep.models.base import safe_config_dump

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _mbe_kwargs(**overrides):
    base = dict(
        workflow="mbe",
        opt_level_of_theory="hf3c_minix",
        opt_dataset="opt_ds",
        small_molecule_collection="small_ds",
        small_molecule="H2CO",
        surface_model_collection="surface_ds",
        surface_model="surface",
        env_unit_len=3,
        dataset="mbe_ds",
        spec=["spec"],
        bsse=["vmfc"],
        levels=[{"index": 1, "method": "scf", "basis": "sto-3g"}],
    )
    base.update(overrides)
    return base


# --- example JSONs round-trip -------------------------------------------------

def test_example_mbe_json_loads():
    raw = json.loads((EXAMPLES_DIR / "mbe.json").read_text())
    cfg = MbeConfig(**raw)
    assert cfg.workflow == "mbe"
    assert cfg.levels
    assert cfg.spec == ["scf_adz_vmfc"]
    assert cfg.bsse == ["vmfc"]


def test_example_mbe_extract_json_loads():
    raw = json.loads((EXAMPLES_DIR / "mbe_extract.json").read_text())
    cfg = MbeExtractConfig(**raw)
    assert cfg.workflow == "mbe_extract"
    assert cfg.zpve is not None
    assert cfg.zpve.enabled is True


# --- spec coercion ------------------------------------------------------------

def test_spec_string_coerced_to_list():
    cfg = MbeConfig(**_mbe_kwargs(spec="only_one"))
    assert cfg.spec == ["only_one"]


def test_spec_list_preserved():
    cfg = MbeConfig(**_mbe_kwargs(spec=["a", "b"]))
    assert cfg.spec == ["a", "b"]


# --- BSSE single-scheme enforcement ------------------------------------------

def test_multiple_bsse_rejected():
    with pytest.raises(ValidationError):
        MbeConfig(**_mbe_kwargs(bsse=["cp", "vmfc"]))


def test_unknown_bsse_rejected():
    with pytest.raises(ValidationError):
        MbeConfig(**_mbe_kwargs(bsse=["bogus"]))


def test_bsse_string_accepted_and_lowercased():
    cfg = MbeConfig(**_mbe_kwargs(bsse="VMFC"))
    assert cfg.bsse == ["vmfc"]


# --- level validation ---------------------------------------------------------

def test_duplicate_level_indices_rejected():
    levels = [
        {"index": 1, "method": "scf", "basis": "sto-3g"},
        {"index": 1, "method": "scf", "basis": "sto-3g"},
    ]
    with pytest.raises(ValidationError):
        MbeConfig(**_mbe_kwargs(levels=levels))


def test_missing_one_body_level_rejected():
    levels = [{"index": 2, "method": "scf", "basis": "sto-3g"}]
    with pytest.raises(ValidationError):
        MbeConfig(**_mbe_kwargs(levels=levels))


def test_nonpositive_level_index_rejected():
    levels = [{"index": 0, "method": "scf", "basis": "sto-3g"}]
    with pytest.raises(ValidationError):
        MbeConfig(**_mbe_kwargs(levels=levels))


def test_invalid_keywords_type_rejected():
    levels = [{"index": 1, "method": "scf", "basis": "sto-3g", "keywords": ["bad"]}]
    with pytest.raises(ValidationError):
        MbeConfig(**_mbe_kwargs(levels=levels))


def test_env_unit_len_must_be_positive():
    with pytest.raises(ValidationError):
        MbeConfig(**_mbe_kwargs(env_unit_len=0))


# --- extract: ZPVE optional ---------------------------------------------------

def test_extract_zpve_optional():
    cfg = MbeExtractConfig(
        workflow="mbe_extract",
        opt_level_of_theory="hf3c_minix",
        small_molecule_collection="small_ds",
        small_molecule="H2CO",
        surface_model_collection="surface_ds",
        surface_model="surface",
        dataset="mbe_ds",
        spec=["spec"],
        bsse=["vmfc"],
    )
    assert cfg.zpve is None


# --- level -> QCSpecification construction (real qcportal objects) -----------

def test_levels_build_real_qcspecifications():
    """MbeLevel objects (keywords omitted => None) must build valid QCSpecs.

    Regression: QCSpecification rejects keywords=None, so the adapter must
    coerce a missing keywords to an empty dict.
    """
    from beep.adapters import qcfractal_adapter as qcf
    cfg = MbeConfig(**_mbe_kwargs(levels=[
        {"index": 1, "method": "scf", "basis": "sto-3g"},                 # keywords omitted -> None
        {"index": 2, "method": "scf", "basis": "sto-3g", "keywords": {"e_convergence": 1e-8}},
    ]))
    specs = qcf.mbe_levels_to_qc_specifications(cfg.levels, "psi4")
    assert set(specs) == {1, 2}
    assert specs[1].keywords == {}
    assert specs[2].keywords == {"e_convergence": 1e-8}
    mb_spec = qcf.build_manybody_specification(specs, cfg.bsse)
    assert mb_spec.program == "qcmanybody"
    assert mb_spec.bsse_correction == ["vmfc"]
    assert mb_spec.keywords.return_total_data is True


# --- credential stripping on dump --------------------------------------------

def test_safe_config_dump_strips_credentials():
    cfg = MbeConfig(**_mbe_kwargs(server={"address": "h:1", "username": "u", "password": "p"}))
    dumped = json.loads(safe_config_dump(cfg))
    assert "username" not in dumped["server"]
    assert "password" not in dumped["server"]
    assert dumped["server"]["address"] == "h:1"
