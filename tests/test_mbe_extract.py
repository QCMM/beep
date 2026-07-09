"""Tests for beep/workflows/mbe_extract.py (BE assembly core).

Ported from beep-mbe's test_assemble_be.py. Calls ``assemble_mbe_be`` directly
with a tmp output folder and monkeypatched ``qcf.get_collection``.
"""
import pandas as pd
import pytest
import qcelemental

constants = qcelemental.constants

from beep.adapters import qcfractal_adapter as qcf
from beep.workflows import mbe_extract as ex
from beep.core import mbe_be_tools as bt
from beep.models.mbe import MbeExtractConfig


class DummyRecord:
    def __init__(self, properties=None, return_result=None):
        self.properties = properties or {}
        self.return_result = return_result
        self.status = "COMPLETE"


class DummyManybodyDataset:
    def __init__(self, records, entry_names):
        self._records = records
        self.entry_names = entry_names

    def fetch_records(self, **kwargs):
        return None

    def get_record(self, entry_name, specification_name):
        return self._records.get((entry_name, specification_name))


class DummySinglepointDataset:
    def __init__(self, records):
        self._records = records

    def fetch_records(self, **kwargs):
        return None

    def get_record(self, entry_name, specification_name):
        return self._records.get((entry_name, specification_name))


def _properties_for(pref, e1=None, e2=None, e3=None, etot=None):
    results = {}
    if e1 is not None:
        results[f"{pref}_total_energy_through_1_body"] = e1
    if e2 is not None:
        results[f"{pref}_2_body_contribution_to_energy"] = e2
    if e3 is not None:
        results[f"{pref}_3_body_contribution_to_energy"] = e3
    if etot is not None:
        results[f"{pref}_total_energy"] = etot
    return {"results": results}


def _patch_collections(monkeypatch, mb_ds, sp_ds):
    def fake_get_collection(client, collection_type, name):
        if collection_type == "ManybodyDataset":
            return mb_ds
        if collection_type == "SinglepointDataset":
            return sp_ds
        raise KeyError(collection_type)
    monkeypatch.setattr(qcf, "get_collection", fake_get_collection)


def _make_config(specs, surface="surface"):
    return MbeExtractConfig(
        workflow="mbe_extract",
        opt_level_of_theory="hf3c_minix",
        small_molecule_collection="sp_ds",
        small_molecule="monomer",
        surface_model_collection="surface_ds",
        surface_model=surface,
        dataset="mbe_ds",
        entries=None,
        spec=specs,
        bsse=["vmfc"],
    )


def test_run_writes_report_and_csvs(tmp_path, monkeypatch):
    pref = "vmfc_corrected"
    spec = "spec"
    surface = "surface"
    entries = ["cluster-a", "cluster-b"]

    mb_records = {
        (surface, spec): DummyRecord(_properties_for(pref, e1=4.0, e2=0.5, e3=0.1, etot=4.6)),
        ("cluster-a", spec): DummyRecord(_properties_for(pref, e1=10.0, e2=1.5, e3=0.2, etot=11.7)),
        ("cluster-b", spec): DummyRecord(_properties_for(pref, e1=9.0, e2=None, e3=0.1, etot=10.2)),
    }
    sp_records = {("monomer", f"monomer_{spec}"): DummyRecord(return_result=1.2)}

    mb_ds = DummyManybodyDataset(mb_records, [surface, *entries])
    sp_ds = DummySinglepointDataset(sp_records)
    _patch_collections(monkeypatch, mb_ds, sp_ds)

    cfg = _make_config([spec], surface)
    out_path = ex.assemble_mbe_be(cfg, client=object(), res_folder=tmp_path)

    assert out_path.exists()
    data_dir = tmp_path / "be_data"
    total_csv = data_dir / "total_be.csv"
    assert total_csv.exists()
    assert (data_dir / "decomp__spec.csv").exists()
    assert (data_dir / "contrib__spec.csv").exists()

    df_total = pd.read_csv(total_csv, index_col=0)
    assert list(df_total.columns) == [spec]
    assert list(df_total.index) == entries

    total_expected = bt.compute_be_values(
        {"e1": 10.0, "e2": 1.5, "e3": 0.2, "etot": 11.7},
        {"e1": 4.0, "e2": 0.5, "e3": 0.1, "etot": 4.6},
        1.2,
    )["be_total"]
    assert df_total.loc["cluster-a", spec] == pytest.approx(total_expected * constants.hartree2kcalmol)

    df_decomp = pd.read_csv(data_dir / "decomp__spec.csv", index_col=0)
    assert list(df_decomp.columns) == ["BE_1b", "BE_2b", "BE_3b"]
    be_1b = df_decomp.loc["cluster-a", "BE_1b"]
    be_2b = df_decomp.loc["cluster-a", "BE_2b"]
    be_3b = df_decomp.loc["cluster-a", "BE_3b"]
    assert be_2b == pytest.approx(be_1b + (-(1.5 - 0.5) * constants.hartree2kcalmol))
    assert be_3b == pytest.approx(be_2b + (-(0.2 - 0.1) * constants.hartree2kcalmol))
    assert pd.isna(df_decomp.loc["cluster-b", "BE_2b"])

    report_text = out_path.read_text()
    assert "BSSE scheme: vmfc" in report_text
    assert "Per-body contributions" in report_text
    assert "\t" not in report_text
    # No ZPVE table when zpve config is absent.
    assert not (data_dir / "total_be_zpve.csv").exists()
    assert "ZPVE" not in report_text


def test_zpve_enabled_writes_corrected_table(tmp_path, monkeypatch):
    pref = "vmfc_corrected"
    spec = "spec"
    surface = "surface"
    entries = ["cluster-a"]

    mb_records = {
        (surface, spec): DummyRecord(_properties_for(pref, e1=4.0, e2=0.5, e3=0.1, etot=4.6)),
        ("cluster-a", spec): DummyRecord(_properties_for(pref, e1=10.0, e2=1.5, e3=0.2, etot=11.7)),
    }
    sp_records = {("monomer", f"monomer_{spec}"): DummyRecord(return_result=1.2)}
    mb_ds = DummyManybodyDataset(mb_records, [surface, *entries])
    sp_ds = DummySinglepointDataset(sp_records)
    _patch_collections(monkeypatch, mb_ds, sp_ds)

    # Stub the read-only ZPVE borrow to a known per-site value.
    monkeypatch.setattr(
        bt, "borrow_zpve_corrections",
        lambda *a, **k: pd.Series({"cluster-a": 1.5}, name="Delta_ZPVE"),
    )

    cfg = MbeExtractConfig(
        workflow="mbe_extract",
        opt_level_of_theory="hf3c_minix",
        small_molecule_collection="sp_ds",
        small_molecule="monomer",
        surface_model_collection="surface_ds",
        surface_model=surface,
        dataset="mbe_ds",
        spec=[spec],
        bsse=["vmfc"],
        zpve={"enabled": True, "hessian_clusters": ["cd5"]},
    )
    out_path = ex.assemble_mbe_be(cfg, client=object(), res_folder=tmp_path)

    data_dir = tmp_path / "be_data"
    zpve_csv = data_dir / "total_be_zpve.csv"
    assert zpve_csv.exists()
    df_zpve = pd.read_csv(zpve_csv, index_col=0)
    assert f"{spec}+ZPVE" in df_zpve.columns
    assert "Delta_ZPVE" in df_zpve.columns
    df_total = pd.read_csv(data_dir / "total_be.csv", index_col=0)
    assert df_zpve.loc["cluster-a", f"{spec}+ZPVE"] == pytest.approx(
        df_total.loc["cluster-a", spec] + 1.5
    )
    assert "ZPVE correction" in out_path.read_text()


def test_run_supports_multiple_specs(tmp_path, monkeypatch):
    pref = "vmfc_corrected"
    specs = ["spec-a", "spec-b"]
    surface = "surface"
    entries = ["cluster-a"]

    mb_records = {}
    sp_records = {}
    for spec in specs:
        mb_records[(surface, spec)] = DummyRecord(_properties_for(pref, e1=4.0, e2=0.5, e3=0.1, etot=4.6))
        mb_records[(entries[0], spec)] = DummyRecord(_properties_for(pref, e1=10.0, e2=1.5, e3=0.2, etot=11.7))
        sp_records[("monomer", f"monomer_{spec}")] = DummyRecord(return_result=1.2)

    mb_ds = DummyManybodyDataset(mb_records, [surface, *entries])
    sp_ds = DummySinglepointDataset(sp_records)
    _patch_collections(monkeypatch, mb_ds, sp_ds)

    cfg = _make_config(specs, surface)
    out_path = ex.assemble_mbe_be(cfg, client=object(), res_folder=tmp_path)

    data_dir = tmp_path / "be_data"
    df_total = pd.read_csv(data_dir / "total_be.csv", index_col=0)
    assert list(df_total.columns) == specs
    for spec in specs:
        assert (data_dir / f"decomp__{spec}.csv").exists()
        assert (data_dir / f"contrib__{spec}.csv").exists()
