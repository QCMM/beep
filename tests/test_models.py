"""Tests for beep/models/ Pydantic config schemas."""
import pytest
from pydantic import ValidationError

from beep.models import SamplingConfig
from beep.models.base import ServerConfig, LevelOfTheory

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
    with pytest.raises(ValidationError):
        SamplingConfig(
            workflow="sampling",
            sampling_level_of_theory={"method": "hf3c"},
            refinement_level_of_theory={"method": "b3lyp-d3bj", "basis": "def2-tzvp"},
        )


def test_sampling_config_wrong_workflow():
    with pytest.raises(ValidationError):
        SamplingConfig(
            workflow="extract",
            molecule="CO",
            sampling_level_of_theory={"method": "hf3c"},
            refinement_level_of_theory={"method": "b3lyp-d3bj", "basis": "def2-tzvp"},
        )
