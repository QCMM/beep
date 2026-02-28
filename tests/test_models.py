"""Tests for beep/models/ Pydantic config schemas."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from beep.models import (
    SamplingConfig,
    BeHessConfig,
    ExtractConfig,
    PreExpConfig,
    GeomBenchmarkConfig,
    EnergyBenchmarkConfig,
)
from beep.models.base import ServerConfig, LevelOfTheory

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    return json.loads((EXAMPLES_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Valid example configs
# ---------------------------------------------------------------------------

def test_sampling_config_valid():
    data = _load("sampling.json")
    cfg = SamplingConfig(**data)
    assert cfg.workflow == "sampling"


def test_be_hess_config_valid():
    data = _load("be_hess.json")
    cfg = BeHessConfig(**data)
    assert cfg.workflow == "be_hess"


def test_extract_config_valid():
    data = _load("extract.json")
    cfg = ExtractConfig(**data)
    assert cfg.workflow == "extract"


def test_pre_exp_config_valid():
    data = _load("pre_exp.json")
    cfg = PreExpConfig(**data)
    assert cfg.workflow == "pre_exp"


def test_geom_benchmark_config_valid():
    data = _load("geom_benchmark.json")
    cfg = GeomBenchmarkConfig(**data)
    assert cfg.workflow == "geom_benchmark"


def test_energy_benchmark_config_valid():
    data = _load("energy_benchmark.json")
    cfg = EnergyBenchmarkConfig(**data)
    assert cfg.workflow == "energy_benchmark"


# ---------------------------------------------------------------------------
# Defaults and edge cases
# ---------------------------------------------------------------------------

def test_server_config_defaults():
    sc = ServerConfig()
    assert sc.address == "localhost:7777"
    assert sc.verify is False


def test_level_of_theory_defaults():
    lot = LevelOfTheory(method="b3lyp")
    assert lot.program == "psi4"


def test_sampling_config_missing_molecule():
    data = _load("sampling.json")
    del data["molecule"]
    with pytest.raises(ValidationError):
        SamplingConfig(**data)


def test_sampling_config_wrong_workflow():
    data = _load("sampling.json")
    data["workflow"] = "extract"
    with pytest.raises(ValidationError):
        SamplingConfig(**data)
