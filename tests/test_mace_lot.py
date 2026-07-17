"""Tests for MACE model-file support in LevelOfTheory and workflows.

A LevelOfTheory (or BeHessConfig.mace_models entry) can point at a
serialized MACE model file. The model file path becomes the QCSpec
method (stock QCEngine ``mace`` harness); the file stem becomes the
LOT name used for specs, datasets, dataframe columns, and files.
"""
import pytest
from pydantic import ValidationError

from beep.models import BeHessConfig, SamplingConfig
from beep.models.base import (
    LevelOfTheory,
    split_lot_string,
    validate_mace_model_path,
)
from beep.workflows.be_hess import _rdset_base_name
from beep.workflows.extract import resolve_be_column

MODEL_PATH = "/home/user/models/mace-polar-ft0.model"


# ---------------------------------------------------------------------------
# LevelOfTheory with mace_model
# ---------------------------------------------------------------------------

def test_mace_lot_mutes_method_basis_program():
    lot = LevelOfTheory(mace_model=MODEL_PATH)
    assert lot.is_mace
    assert lot.qc_program == "mace"
    assert lot.qc_method == MODEL_PATH
    assert lot.qc_basis is None


def test_mace_lot_alias_and_lot_name_from_stem():
    lot = LevelOfTheory(mace_model=MODEL_PATH)
    assert lot.alias == "mace-polar-ft0"
    assert lot.lot_name == "mace-polar-ft0"


def test_mace_lot_mutes_explicit_method_and_basis():
    lot = LevelOfTheory(method="b3lyp", basis="def2-svp", mace_model=MODEL_PATH)
    assert lot.qc_method == MODEL_PATH
    assert lot.qc_basis is None
    assert lot.qc_program == "mace"
    assert lot.lot_name == "mace-polar-ft0"


def test_conventional_lot_unaffected():
    lot = LevelOfTheory(method="mpwb1k-d3bj", basis="def2-tzvp")
    assert not lot.is_mace
    assert lot.qc_program == "psi4"
    assert lot.qc_method == "mpwb1k-d3bj"
    assert lot.qc_basis == "def2-tzvp"
    assert lot.lot_name == "mpwb1k-d3bj_def2-tzvp"


def test_basisless_conventional_lot_name():
    lot = LevelOfTheory(method="gfn2-xtb", program="xtb")
    assert lot.lot_name == "gfn2-xtb"


def test_lot_requires_method_or_mace_model():
    with pytest.raises(ValidationError):
        LevelOfTheory()


def test_mace_model_path_must_be_lowercase():
    with pytest.raises(ValidationError):
        LevelOfTheory(mace_model="/home/user/Models/mace-polar-ft0.model")


def test_mace_model_stem_must_not_contain_underscore():
    with pytest.raises(ValidationError):
        LevelOfTheory(mace_model="/home/user/models/mace_polar_ft0.model")


def test_validate_mace_model_path_accepts_none():
    assert validate_mace_model_path(None) is None


def test_mace_lot_display():
    lot = LevelOfTheory(mace_model=MODEL_PATH)
    assert "mace-polar-ft0" in lot.display
    assert "mace" in lot.display


# ---------------------------------------------------------------------------
# split_lot_string
# ---------------------------------------------------------------------------

def test_split_lot_string_with_basis():
    assert split_lot_string("mpwb1k-d3bj_def2-tzvp") == ("mpwb1k-d3bj", "def2-tzvp")


def test_split_lot_string_without_basis():
    assert split_lot_string("gfn2-xtb") == ("gfn2-xtb", None)
    assert split_lot_string("mace-polar-ft0") == ("mace-polar-ft0", None)


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

def test_sampling_config_accepts_mace_lots():
    cfg = SamplingConfig(
        workflow="sampling",
        molecule="H2S",
        sampling_level_of_theory={"method": "gfn2-xtb", "program": "xtb"},
        refinement_level_of_theory={"mace_model": MODEL_PATH},
    )
    assert cfg.refinement_level_of_theory.lot_name == "mace-polar-ft0"


def test_be_hess_config_mace_models():
    cfg = BeHessConfig(
        workflow="be_hess",
        molecule="H2S",
        opt_level_of_theory="mace-polar-ft0",
        mace_models=[MODEL_PATH],
    )
    assert cfg.mace_models == [MODEL_PATH]


def test_be_hess_config_rejects_bad_mace_model():
    with pytest.raises(ValidationError):
        BeHessConfig(
            workflow="be_hess",
            molecule="H2S",
            opt_level_of_theory="mace-polar-ft0",
            mace_models=["/home/user/models/BAD_Model.model"],
        )


# ---------------------------------------------------------------------------
# Reaction dataset naming
# ---------------------------------------------------------------------------

def test_rdset_name_with_basis():
    assert (
        _rdset_base_name("H2S", "W22_02", "mpwb1k-d3bj_def2-tzvp")
        == "be_H2S_W22_02_MPWB1K-D3BJ_DEF2-TZVP"
    )


def test_rdset_name_basisless():
    assert (
        _rdset_base_name("H2S", "W22_02", "mace-polar-ft0")
        == "be_H2S_W22_02_MACE-POLAR-FT0"
    )


# ---------------------------------------------------------------------------
# Extract column resolution
# ---------------------------------------------------------------------------

def test_resolve_be_column_prefers_slash_form():
    cols = ["wpbe-d3bj/def2-tzvp", "mace-polar-ft0"]
    assert resolve_be_column("wpbe-d3bj", "def2-tzvp", cols) == "wpbe-d3bj/def2-tzvp"


def test_resolve_be_column_basisless_method():
    cols = ["wpbe-d3bj/def2-tzvp", "mace-polar-ft0"]
    assert resolve_be_column("mace-polar-ft0", "def2-tzvp", cols) == "mace-polar-ft0"


def test_resolve_be_column_missing_falls_back_to_slash():
    assert resolve_be_column("m06-hf", "def2-tzvp", []) == "m06-hf/def2-tzvp"


# ---------------------------------------------------------------------------
# compute_be_mace_energies — bsse must be skipped, specs named by alias
# ---------------------------------------------------------------------------

import logging
from unittest.mock import MagicMock

from beep.adapters.qcfractal_adapter import (
    STOICH_TYPES,
    compute_be_mace_energies,
    submit_energies,
)


def _mock_client_with_datasets():
    """Client whose get_dataset returns a fresh MagicMock per dataset name."""
    client = MagicMock()
    datasets = {}

    def get_dataset(_dstype, name):
        if name not in datasets:
            ds = MagicMock()
            ds.submit.return_value = MagicMock(n_inserted=1, n_existing=0)
            ds.iterate_records.return_value = iter([])
            datasets[name] = ds
        return datasets[name]

    client.get_dataset.side_effect = get_dataset
    return client, datasets


def test_mace_energies_skip_bsse_and_use_alias_spec():
    client, datasets = _mock_client_with_datasets()
    logger = logging.getLogger("test")

    compute_be_mace_energies(
        client, "be_H2S_W22_02_MACE-POLAR-FT0", [MODEL_PATH],
        tag="energies", logger=logger,
    )

    submitted = {
        name: ds for name, ds in datasets.items()
        if ds.add_specification.called
    }
    # bsse dataset must never receive a MACE spec
    assert "be_H2S_W22_02_MACE-POLAR-FT0_bsse" not in submitted
    expected = {
        f"be_H2S_W22_02_MACE-POLAR-FT0_{s}" for s in STOICH_TYPES if s != "bsse"
    }
    assert set(submitted) == expected

    for ds in submitted.values():
        spec_name, rxn_spec = ds.add_specification.call_args.args
        assert spec_name == "mace-polar-ft0"
        qc_spec = rxn_spec.singlepoint_specification
        assert qc_spec.program == "mace"
        assert qc_spec.method == MODEL_PATH
        assert qc_spec.basis is None


def test_mace_energies_committee_one_spec_per_model():
    client, datasets = _mock_client_with_datasets()
    logger = logging.getLogger("test")
    models = [
        "/home/user/models/mace-polar-ft0.model",
        "/home/user/models/mace-polar-ft1.model",
    ]

    compute_be_mace_energies(
        client, "be_H2S_W22_02_MACE-POLAR-FT0", models,
        tag="energies", logger=logger,
    )

    ds = datasets["be_H2S_W22_02_MACE-POLAR-FT0_be_nocp"]
    spec_names = [c.args[0] for c in ds.add_specification.call_args_list]
    assert spec_names == ["mace-polar-ft0", "mace-polar-ft1"]


def test_submit_energies_spec_name_override_and_default():
    client, datasets = _mock_client_with_datasets()

    submit_energies(client, "base", method=MODEL_PATH, basis=None,
                    program="mace", stoich="be_nocp", tag="t",
                    spec_name="mace-polar-ft0")
    ds = datasets["base_be_nocp"]
    assert ds.add_specification.call_args.args[0] == "mace-polar-ft0"

    submit_energies(client, "base", method="wpbe-d3bj", basis="def2-tzvp",
                    program="psi4", stoich="be_nocp", tag="t")
    assert ds.add_specification.call_args.args[0] == "wpbe-d3bj_def2-tzvp"


# ---------------------------------------------------------------------------
# Extract — MACE/basis-free columns survive the column cleanup
# ---------------------------------------------------------------------------

import pandas as pd

from beep.workflows import extract as extract_wf


def test_concatenate_frames_keeps_basisless_method_columns(monkeypatch):
    df = pd.DataFrame(
        {
            "mpwb1k/def2-tzvpd": [-2.5, -3.0],
            "mpwb1k-d3bj/def2-tzvpd": [-3.1, -3.6],
            "mpwb1k-d3bj": [-0.6, -0.6],
            "mace-polar-ft0": [-3.2, -3.5],
            "gfn2-xtb": [-2.9, -3.3],
        },
        index=["H2S_W22_02_0001", "H2S_W22_02_0002"],
    )
    monkeypatch.setattr(
        extract_wf.qcf, "check_collection_exists", lambda *a, **k: True
    )
    monkeypatch.setattr(
        extract_wf.qcf, "fetch_reaction_values", lambda *a, **k: df.copy()
    )
    ds_w = MagicMock()
    ds_w.entry_names = ["W22_02"]

    df_be, success = extract_wf.concatenate_frames(
        MagicMock(), "H2S", ds_w, "mace-polar-ft0",
        be_range=(-0.1, -25.0), stoichiometry="be_nocp",
    )

    assert success
    # Basis-free method columns are real BE data — kept
    assert "mace-polar-ft0" in df_be.columns
    assert "gfn2-xtb" in df_be.columns
    # Dispersion bookkeeping columns are still cleaned up
    assert "mpwb1k-d3bj" not in df_be.columns
    assert "mpwb1k/def2-tzvpd" not in df_be.columns
    assert "mpwb1k-d3bj/def2-tzvpd" in df_be.columns


# ---------------------------------------------------------------------------
# Sampling — refinement spec built from a MACE LOT
# ---------------------------------------------------------------------------

from beep.workflows import sampling as sampling_wf


def test_process_refinement_builds_mace_spec(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sampling_wf.qcf, "add_opt_specification",
        lambda ds, spec, overwrite=True: captured.update(spec),
    )
    monkeypatch.setattr(
        sampling_wf.qcf, "submit_optimizations", lambda *a, **k: 5
    )

    lot = LevelOfTheory(mace_model=MODEL_PATH)
    sampling_wf.process_refinement(
        MagicMock(), lot.lot_name, lot.qc_method, lot.qc_basis,
        lot.qc_program, None, MagicMock(), logging.getLogger("test"),
        lot_display=lot.display,
    )

    assert captured["name"] == "mace-polar-ft0"
    qc_spec = captured["qc_spec"]
    assert qc_spec["program"] == "mace"
    assert qc_spec["method"] == MODEL_PATH
    assert qc_spec["basis"] is None
    assert "mace-polar-ft0" in captured["description"]
