"""Tests for the SAPT workflow layer."""
from unittest.mock import MagicMock

import numpy as np
import pytest
from qcelemental.models import Molecule
from qcportal.record_models import RecordStatusEnum

from beep.models.sapt import SaptConfig
from beep.workflows.sapt import collect_fragmented_entries, run, sapt_dataset_name, sapt_keywords


def _w5_so2_structure():
    surface_symbols = ["O", "H", "H"] * 5
    adsorbate_symbols = ["S", "O", "O"]
    symbols = surface_symbols + adsorbate_symbols
    geometry = np.arange(len(symbols) * 3, dtype=float).reshape(-1, 3)
    return Molecule(symbols=symbols, geometry=geometry, fix_com=True, fix_orientation=True)


def _mock_client_with_one_complete_so2():
    client = MagicMock()
    surface_ds = MagicMock()
    surface_ds.entry_names = ["W5_01"]

    opt_ds = MagicMock()
    opt_ds.entry_names = ["SO2_W5_01_0008"]
    opt_ds.specification_names = ["mpwb1k-d3bj_def2-tzvpd"]
    record = MagicMock()
    record.status = RecordStatusEnum.complete
    record.final_molecule = _w5_so2_structure()
    opt_ds.get_record.return_value = record

    def get_dataset(dataset_type, name):
        assert dataset_type == "optimization"
        if name == "w5-7":
            return surface_ds
        if name == "SO2_W5_01":
            return opt_ds
        raise KeyError(name)

    client.get_dataset.side_effect = get_dataset
    return client


def test_collect_fragmented_entries_prepares_one_so2_entry():
    config = SaptConfig(
        workflow="sapt",
        molecule="SO2",
        surface_model="w5-7",
        optimization_spec="mpwb1k-d3bj_def2-tzvpd",
        entries=["SO2_W5_01_0008"],
    )
    entries = collect_fragmented_entries(config, _mock_client_with_one_complete_so2(), MagicMock())

    assert len(entries) == 1
    entry_name, molecule = entries[0]
    assert entry_name == "SO2_W5_01_0008"
    assert [len(fragment) for fragment in molecule.fragments] == [15, 3]


def test_collect_fragmented_entries_skips_unrequested_cluster_datasets():
    config = SaptConfig(
        workflow="sapt",
        molecule="SO2",
        surface_model="w5-7",
        optimization_spec="mpwb1k-d3bj_def2-tzvpd",
        entries=["SO2_W5_01_0008"],
    )
    client = MagicMock()
    surface_ds = MagicMock()
    surface_ds.entry_names = ["W5_01", "W6_01"]

    opt_ds = MagicMock()
    opt_ds.entry_names = ["SO2_W5_01_0008"]
    opt_ds.specification_names = ["mpwb1k-d3bj_def2-tzvpd"]
    record = MagicMock()
    record.status = RecordStatusEnum.complete
    record.final_molecule = _w5_so2_structure()
    opt_ds.get_record.return_value = record

    requested_datasets = []

    def get_dataset(dataset_type, name):
        assert dataset_type == "optimization"
        requested_datasets.append(name)
        if name == "w5-7":
            return surface_ds
        if name == "SO2_W5_01":
            return opt_ds
        raise AssertionError(f"Unexpected dataset lookup: {name}")

    client.get_dataset.side_effect = get_dataset

    entries = collect_fragmented_entries(config, client, MagicMock())

    assert [entry_name for entry_name, _ in entries] == ["SO2_W5_01_0008"]
    assert requested_datasets == ["w5-7", "SO2_W5_01"]


def test_collect_fragmented_entries_rejects_requested_entries_from_missing_clusters():
    config = SaptConfig(
        workflow="sapt",
        molecule="SO2",
        surface_model="w5-7",
        optimization_spec="mpwb1k-d3bj_def2-tzvpd",
        entries=["SO2_W8_01_0001"],
    )
    client = MagicMock()
    surface_ds = MagicMock()
    surface_ds.entry_names = ["W5_01"]

    def get_dataset(dataset_type, name):
        assert dataset_type == "optimization"
        if name == "w5-7":
            return surface_ds
        raise AssertionError(f"Unexpected dataset lookup: {name}")

    client.get_dataset.side_effect = get_dataset

    with pytest.raises(KeyError, match="SO2_W8_01_0001"):
        collect_fragmented_entries(config, client, MagicMock())


def test_sapt_keywords_keep_singlet_default_reference_implicit():
    config = SaptConfig(
        workflow="sapt",
        molecule="SO2",
        surface_model="w5-7",
        optimization_spec="mpwb1k-d3bj_def2-tzvpd",
    )

    assert "reference" not in sapt_keywords(config)


def test_sapt_keywords_infer_uhf_for_open_shell_fragment():
    config = SaptConfig(
        workflow="sapt",
        molecule="CH2SH",
        surface_model="w5-7",
        optimization_spec="mpwb1k-d3bj_def2-tzvpd",
        molecule_multiplicity=2,
    )

    assert sapt_keywords(config)["reference"] == "uhf"


def test_sapt_keywords_preserve_explicit_reference():
    config = SaptConfig(
        workflow="sapt",
        molecule="CH2SH",
        surface_model="w5-7",
        optimization_spec="mpwb1k-d3bj_def2-tzvpd",
        molecule_multiplicity=2,
        keywords={"scf_type": "df", "reference": "rohf"},
    )

    assert sapt_keywords(config)["reference"] == "rohf"


def test_sapt_workflow_dry_run_writes_plan_without_submission(tmp_path, monkeypatch):
    config = SaptConfig(
        workflow="sapt",
        molecule="SO2",
        surface_model="w5-7",
        optimization_spec="mpwb1k-d3bj_def2-tzvpd",
        entries=["SO2_W5_01_0008"],
        dry_run=True,
    )
    client = _mock_client_with_one_complete_so2()
    monkeypatch.chdir(tmp_path)

    run(config, client)

    plan = tmp_path / "SO2" / "sapt" / "sapt_plan.csv"
    copied_config = tmp_path / "SO2" / "sapt" / "sapt_SO2.json"
    assert plan.exists()
    assert copied_config.exists()
    assert "SO2_W5_01_0008" in plan.read_text()
    client.add_dataset.assert_not_called()


def test_sapt_dataset_name_is_generic():
    so2 = SaptConfig(
        workflow="sapt",
        molecule="SO2",
        surface_model="w5-7",
        optimization_spec="mpwb1k-d3bj_def2-tzvpd",
    )
    diol = SaptConfig(
        workflow="sapt",
        molecule="CH2OHCH2OH",
        surface_model="W22",
        optimization_spec="hf3c_minix",
    )

    assert sapt_dataset_name(so2) != sapt_dataset_name(diol)
    assert "SO2" in sapt_dataset_name(so2)
    assert "CH2OHCH2OH" in sapt_dataset_name(diol)
