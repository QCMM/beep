"""Tests for beep/models/ Pydantic config schemas."""
import pytest
from pydantic import ValidationError

from beep.models import (
    SamplingConfig,
    BeHessConfig,
    EnergyBenchmarkConfig,
    ExtractConfig,
    GeomBenchmarkConfig,
    PreExpConfig,
)
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


# ---------------------------------------------------------------------------
# Case normalization — every spec-name field must be lowercased on load,
# because qcportal 0.63+ stores spec names lowercase and case-sensitive
# lookups fail otherwise.
# ---------------------------------------------------------------------------

def test_level_of_theory_lowercases_method_and_basis():
    lot = LevelOfTheory(method="B3LYP-D3BJ", basis="DEF2-TZVP")
    assert lot.method == "b3lyp-d3bj"
    assert lot.basis == "def2-tzvp"


def test_pre_exp_lowercases_level_of_theory():
    cfg = PreExpConfig(workflow="pre_exp", level_of_theory="MPWB1K-D3BJ_DEF2-TZVPD")
    assert cfg.level_of_theory == "mpwb1k-d3bj_def2-tzvpd"


def test_extract_lowercases_opt_method_basis_and_be_methods():
    cfg = ExtractConfig(
        workflow="extract",
        surface_model="w22",
        molecules=["H2O"],
        opt_method="HF3C_MINIX",
        basis="DEF2-TZVP",
        be_methods=["WB97X-V", "M06-HF"],
    )
    assert cfg.opt_method == "hf3c_minix"
    assert cfg.basis == "def2-tzvp"
    assert cfg.be_methods == ["wb97x-v", "m06-hf"]


def test_be_hess_lowercases_lots():
    cfg = BeHessConfig(
        workflow="be_hess",
        molecule="H2O",
        opt_level_of_theory="HF3C_MINIX",
        level_of_theory=["PBE-D3BJ_DEF2-TZVP", "BLYP_DEF2-SVP"],
    )
    assert cfg.opt_level_of_theory == "hf3c_minix"
    assert cfg.level_of_theory == ["pbe-d3bj_def2-tzvp", "blyp_def2-svp"]


def test_energy_benchmark_lowercases_lots():
    cfg = EnergyBenchmarkConfig(
        workflow="energy_benchmark",
        molecule="H2O",
        benchmark_structures=["W22_01"],
        opt_level_of_theory=["MPWB1K-D3BJ_DEF2-TZVPD"],
        reference_geometry_level_of_theory="CCSD(T)_AUG-CC-PVTZ",
        be_basis="DEF2-TZVPD",
        tag_be="be",
        tag_cbs="cbs",
    )
    assert cfg.opt_level_of_theory == ["mpwb1k-d3bj_def2-tzvpd"]
    assert cfg.reference_geometry_level_of_theory == "ccsd(t)_aug-cc-pvtz"
    assert cfg.be_basis == "def2-tzvpd"


def test_geom_benchmark_lowercases_reference_method_and_basis():
    cfg = GeomBenchmarkConfig(
        workflow="geom_benchmark",
        molecule="H2O",
        benchmark_structures=["W22_01"],
        reference_geometry_level_of_theory=["CCSD(T)", "AUG-CC-PVTZ", "psi4"],
    )
    # Method (idx 0) and basis (idx 1) lowercased; program (idx 2) left alone.
    assert cfg.reference_geometry_level_of_theory == ["ccsd(t)", "aug-cc-pvtz", "psi4"]
