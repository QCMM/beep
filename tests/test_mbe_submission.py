"""Tests for beep/workflows/mbe.py (submission core).

Ported from beep-mbe's test_submission_entries.py. The workflow reaches the
server only through ``beep.adapters.qcfractal_adapter`` (aliased ``qcf``), so we
monkeypatch those adapter functions with in-memory dummies.
"""
from typing import List

import pytest

from beep.core.mbe_be_tools import submitted_entry_names
from beep.adapters import qcfractal_adapter as qcf
from beep.workflows import mbe as mbe_wf
from beep.models.mbe import MbeConfig


class DummyMol:
    def __init__(self, symbols, fragments):
        self.symbols = symbols
        self.fragments = fragments


class DummyManybodyDataset:
    def __init__(self):
        self.entry_names = []
        self.added_entries: List[str] = []
        self.submitted_entries: List[str] = []

    def add_specification(self, spec_name, spec):
        return None

    def add_entry(self, name, initial_molecule, overwrite=False):
        self.added_entries.append(name)

    def submit(self, entry_names, specification_names, compute_tag, find_existing=True):
        self.submitted_entries = entry_names
        return {"submitted": entry_names}

    def fetch_records(self, **kwargs):
        return None

    def get_record(self, entry_name, specification_name):
        return None


class DummySinglepointDataset:
    def __init__(self):
        self.specifications = {}
        self.entry_names = []
        self.added_entries: List[str] = []
        self.submit_calls = 0

    def add_specification(self, name, spec):
        self.specifications[name] = spec

    def add_entry(self, name, molecule):
        self.added_entries.append(name)

    def get_record(self, entry_name, specification_name):
        from qcportal.record_models import RecordStatusEnum

        class Record:
            # Real enum, not the string "COMPLETE": the workflow must skip
            # re-submitting COMPLETE monomers via qcf.is_complete.
            status = RecordStatusEnum.complete
        return Record()

    def submit(self, **kwargs):
        self.submit_calls += 1
        return {}


def _make_config(**overrides):
    base = dict(
        workflow="mbe",
        opt_level_of_theory="hf3c_minix",
        opt_dataset="opt_ds",
        entries=["cluster-a", "cluster-b"],
        small_molecule_collection="small_ds",
        small_molecule="small-ref",
        surface_model_collection="surface_ds",
        surface_model="surface-ref",
        env_unit_len=3,
        dataset="mbe_ds",
        spec=["spec"],
        bsse=["vmfc"],
        tag="tag",
        levels=[{"index": 1, "method": "scf", "basis": "sto-3g", "keywords": {}}],
        monitor={"enabled": False, "poll_interval": 1, "max_wait": None},
    )
    base.update(overrides)
    return MbeConfig(**base)


def _patch_common(monkeypatch, mb_ds, sp_ds):
    monkeypatch.setattr(qcf, "get_collection", lambda *a, **k: object())
    monkeypatch.setattr(qcf, "fetch_final_molecule", lambda *a, **k: object())
    monkeypatch.setattr(qcf, "get_or_create_manybody_dataset", lambda *a, **k: mb_ds)
    monkeypatch.setattr(qcf, "get_or_create_singlepoint_dataset", lambda *a, **k: sp_ds)
    monkeypatch.setattr(qcf, "mbe_levels_to_qc_specifications", lambda *a, **k: {1: object()})
    monkeypatch.setattr(qcf, "build_manybody_specification", lambda *a, **k: object())
    monkeypatch.setattr(
        mbe_wf.mbe_fragmentation, "fragment_small_molecule",
        lambda *a, **k: DummyMol(["H"], [[0]]),
    )
    monkeypatch.setattr(
        mbe_wf.mbe_fragmentation, "fragment_surface_model",
        lambda *a, **k: DummyMol(["H", "O"], [[0, 1]]),
    )
    monkeypatch.setattr(
        mbe_wf.mbe_fragmentation, "fragment_cluster",
        lambda *a, **k: DummyMol(["H", "O", "H"], [[0, 1, 2]]),
    )
    monkeypatch.setattr(mbe_wf, "_fetch_record_summary", lambda *a, **k: "complete")


def test_submitted_entry_names_dedupes_preserving_order():
    names = submitted_entry_names("surface", "small", ["surface", "cluster-a", "small"])
    assert names == ["surface", "cluster-a"]


def test_monitor_includes_reference_entries(monkeypatch):
    captured = {}
    mb_ds = DummyManybodyDataset()
    sp_ds = DummySinglepointDataset()
    _patch_common(monkeypatch, mb_ds, sp_ds)

    class Result:
        timed_out = False
        errored_entries: List[str] = []

    def fake_wait(**kwargs):
        captured["entry_names"] = kwargs["entry_names"]
        captured["dataset_name"] = kwargs["dataset_name"]
        return Result()

    monkeypatch.setattr(qcf, "wait_for_manybody_completion", fake_wait)

    cfg = _make_config(monitor={"enabled": True, "poll_interval": 1, "max_wait": None})
    mbe_wf.submit_mbe(cfg, client=object())

    assert captured["entry_names"] == ["surface-ref", "cluster-a", "cluster-b"]
    assert captured["dataset_name"] == "mbe_ds_hf3c_minix"
    assert mb_ds.submitted_entries == ["surface-ref", "cluster-a", "cluster-b"]
    assert mb_ds.added_entries == ["surface-ref", "cluster-a", "cluster-b"]
    # COMPLETE monomer must not be re-submitted (regression: enum vs "COMPLETE" string)
    assert sp_ds.submit_calls == 0


def test_singlepoint_dataset_name_includes_opt_level(monkeypatch):
    captured = {}
    mb_ds = DummyManybodyDataset()
    sp_ds = DummySinglepointDataset()

    def fake_mb(_client, name):
        captured["mb_ds_name"] = name
        return mb_ds

    def fake_sp(_client, name):
        captured["sp_ds_name"] = name
        return sp_ds

    _patch_common(monkeypatch, mb_ds, sp_ds)
    monkeypatch.setattr(qcf, "get_or_create_manybody_dataset", fake_mb)
    monkeypatch.setattr(qcf, "get_or_create_singlepoint_dataset", fake_sp)

    cfg = _make_config(
        entries=["cluster-a"],
        small_molecule_collection="astro_molecules",
        fetch_only=True,
    )
    mbe_wf.submit_mbe(cfg, client=object())

    assert captured["sp_ds_name"] == "astro_molecules_hf3c_minix"
    assert captured["mb_ds_name"] == "mbe_ds_hf3c_minix"
