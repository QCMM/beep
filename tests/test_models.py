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
from beep.models.base import ServerConfig, LevelOfTheory, safe_config_dump

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


def test_safe_config_dump_strips_credentials():
    """Workflow configs written to disk for reproducibility must not
    contain plaintext credentials."""
    cfg = SamplingConfig(
        workflow="sampling",
        molecule="CO",
        sampling_level_of_theory={"method": "hf3c"},
        refinement_level_of_theory={"method": "b3lyp", "basis": "def2-tzvp"},
        server={
            "address": "http://example:7777",
            "username": "joe",
            "password": "top-secret-bytes",
            "verify": False,
        },
    )
    text = safe_config_dump(cfg)
    # Credentials stripped
    assert "joe" not in text
    assert "top-secret-bytes" not in text
    # Other server fields preserved
    assert "http://example:7777" in text
    # Other config fields preserved
    assert "CO" in text


def test_sampling_opt_keywords_defaults_none():
    """Both opt-keyword fields default to None so the workflow keeps
    its historical defaults (sampling: {'maxiter': 125}, refinement: None)."""
    cfg = SamplingConfig(
        workflow="sampling",
        molecule="CO",
        sampling_level_of_theory={"method": "hf3c"},
        refinement_level_of_theory={"method": "b3lyp", "basis": "def2-tzvp"},
    )
    assert cfg.sampling_opt_keywords is None
    assert cfg.refinement_opt_keywords is None


def test_sampling_opt_keywords_accepted_as_dict():
    cfg = SamplingConfig(
        workflow="sampling",
        molecule="CO",
        sampling_level_of_theory={"method": "hf3c"},
        refinement_level_of_theory={"method": "b3lyp", "basis": "def2-tzvp"},
        sampling_opt_keywords={"coordsys": "cart"},
        refinement_opt_keywords={"coordsys": "cart", "maxiter": 200},
    )
    assert cfg.sampling_opt_keywords == {"coordsys": "cart"}
    assert cfg.refinement_opt_keywords == {"coordsys": "cart", "maxiter": 200}


def test_sampling_opt_keywords_merge_semantics():
    """Sampling-stage merge preserves the workflow's {'maxiter': 125} default
    unless the user explicitly overrides it."""
    user_kw = {"coordsys": "cart"}
    merged = {"maxiter": 125, **(user_kw or {})}
    assert merged == {"maxiter": 125, "coordsys": "cart"}

    user_kw = {"maxiter": 200}
    merged = {"maxiter": 125, **(user_kw or {})}
    assert merged == {"maxiter": 200}

    user_kw = None
    merged = {"maxiter": 125, **(user_kw or {})}
    assert merged == {"maxiter": 125}


def test_geom_benchmark_lowercases_reference_method_and_basis():
    cfg = GeomBenchmarkConfig(
        workflow="geom_benchmark",
        opt_dataset="mix-h2o-h2s",
        benchmark_structures=["W22_01"],
        reference_geometry_level_of_theory=["CCSD(T)", "AUG-CC-PVTZ", "psi4"],
    )
    # Method (idx 0) and basis (idx 1) lowercased; program (idx 2) left alone.
    assert cfg.reference_geometry_level_of_theory == ["ccsd(t)", "aug-cc-pvtz", "psi4"]


# ---------------------------------------------------------------------------
# PreExpConfig.range_of_temperature validation
# ---------------------------------------------------------------------------

def test_pre_exp_range_of_temperature_accepts_min_max_and_single():
    cfg = PreExpConfig(workflow="pre_exp", range_of_temperature=[10, 273])
    assert cfg.range_of_temperature == [10, 273]
    cfg = PreExpConfig(workflow="pre_exp", range_of_temperature=[50])
    assert cfg.range_of_temperature == [50]


@pytest.mark.parametrize("bad", [[], [10, 20, 30], [273, 10], [0, 100], [-5]])
def test_pre_exp_range_of_temperature_rejects_bad_ranges(bad):
    with pytest.raises(ValidationError):
        PreExpConfig(workflow="pre_exp", range_of_temperature=bad)


def test_pre_exp_temperature_step_must_be_positive():
    with pytest.raises(ValidationError):
        PreExpConfig(workflow="pre_exp", temperature_step=0)


# ---------------------------------------------------------------------------
# Keyword fields are inline dicts, not QCFractal 0.15 keyword IDs
# ---------------------------------------------------------------------------

def _nm_kwargs(**extra):
    base = dict(
        workflow="nm_sampling",
        opt_dataset="ds",
        benchmark_structures=["h2o_2"],
        fragments={"h2o_2": [[0, 1, 2], [3, 4, 5]]},
        geometry_opt_lot="hf3c_minix",
    )
    base.update(extra)
    return base


def test_geom_benchmark_qc_keywords_accepts_dict():
    cfg = GeomBenchmarkConfig(
        workflow="geom_benchmark", opt_dataset="ds", benchmark_structures=["W22_01"],
        qc_keywords={"scf_type": "df", "maxiter": 200},
    )
    assert cfg.qc_keywords == {"scf_type": "df", "maxiter": 200}


def test_geom_benchmark_qc_keywords_rejects_int():
    """A legacy integer keyword ID used to validate and then be silently
    dropped by every consumer; it must now be rejected with a clear error."""
    with pytest.raises(ValidationError) as exc_info:
        GeomBenchmarkConfig(
            workflow="geom_benchmark", opt_dataset="ds",
            benchmark_structures=["W22_01"], qc_keywords=7,
        )
    errs = exc_info.value.errors()
    assert errs[0]["loc"] == ("qc_keywords",)
    assert "dict" in errs[0]["msg"]


def test_nm_sampling_qc_keywords_accepts_dict():
    from beep.models import NmSamplingConfig
    cfg = NmSamplingConfig(**_nm_kwargs(qc_keywords={"scf_type": "df"}))
    assert cfg.qc_keywords == {"scf_type": "df"}
    assert NmSamplingConfig(**_nm_kwargs()).qc_keywords is None


def test_nm_sampling_qc_keywords_rejects_int():
    from beep.models import NmSamplingConfig
    with pytest.raises(ValidationError) as exc_info:
        NmSamplingConfig(**_nm_kwargs(qc_keywords=7))
    errs = exc_info.value.errors()
    assert errs[0]["loc"] == ("qc_keywords",)
    assert "dict" in errs[0]["msg"]


# ---------------------------------------------------------------------------
# Deprecated per-workflow names for the QC-program keywords: null still loads
# (old configs), anything else errors pointing at qc_keywords.
# ---------------------------------------------------------------------------

def _nm_model():
    from beep.models import NmSamplingConfig
    return NmSamplingConfig


_DEPRECATED_QC_KEYWORD_FIELDS = [
    pytest.param(
        SamplingConfig, "keyword_id",
        dict(workflow="sampling", molecule="CO",
             sampling_level_of_theory={"method": "gfn2-xtb", "program": "xtb"},
             refinement_level_of_theory={"method": "hf", "basis": "sto-3g"}),
        id="sampling.keyword_id",
    ),
    pytest.param(
        BeHessConfig, "keyword_id",
        dict(workflow="be_hess", molecule="CO", opt_level_of_theory="hf3c_minix"),
        id="be_hess.keyword_id",
    ),
    pytest.param(
        EnergyBenchmarkConfig, "keyword_id",
        dict(workflow="energy_benchmark", molecule="CO", benchmark_structures=["W22_01"],
             opt_level_of_theory=["pbe_def2-svp"], tag_be="be", tag_cbs="cbs"),
        id="energy_benchmark.keyword_id",
    ),
    pytest.param(
        _nm_model, "dft_keyword", _nm_kwargs(),
        id="nm_sampling.dft_keyword",
    ),
    pytest.param(
        GeomBenchmarkConfig, "dft_optimization_keyword",
        dict(workflow="geom_benchmark", opt_dataset="ds", benchmark_structures=["W22_01"]),
        id="geom_benchmark.dft_optimization_keyword",
    ),
]


def _resolve(model):
    return model() if model is _nm_model else model


@pytest.mark.parametrize("model, old, kwargs", _DEPRECATED_QC_KEYWORD_FIELDS)
def test_deprecated_qc_keyword_name_accepts_null_only(model, old, kwargs):
    model = _resolve(model)
    cfg = model(**kwargs, **{old: None})
    assert getattr(cfg, old) is None
    # never written back out: the config copy on disk and --schema use qc_keywords
    assert old not in cfg.model_dump()
    assert old not in model.model_json_schema()["properties"]
    for bad in (7, "legacy", {"guess": "gwh"}):
        with pytest.raises(ValidationError) as exc_info:
            model(**kwargs, **{old: bad})
        errs = exc_info.value.errors()
        assert errs[0]["loc"] == (old,)
        assert "qc_keywords" in errs[0]["msg"]


@pytest.mark.parametrize("model, kwargs", [
    pytest.param(m.values[0], m.values[2], id=m.id)
    for m in _DEPRECATED_QC_KEYWORD_FIELDS if m.id != "energy_benchmark.keyword_id"
])
def test_qc_keywords_is_the_unified_program_keyword_field(model, kwargs):
    model = _resolve(model)
    cfg = model(**kwargs, qc_keywords={"guess": "gwh", "damping_percentage": 20})
    assert cfg.qc_keywords == {"guess": "gwh", "damping_percentage": 20}
    assert cfg.model_dump()["qc_keywords"] == {"guess": "gwh", "damping_percentage": 20}
    assert "qc_keywords" in model.model_json_schema()["properties"]


def test_energy_benchmark_has_no_qc_keywords_field():
    """energy_benchmark.keyword_id was never read; it has no replacement."""
    assert "qc_keywords" not in EnergyBenchmarkConfig.model_fields
