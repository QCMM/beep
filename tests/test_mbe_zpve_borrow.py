"""Tests for beep.core.mbe_be_tools.borrow_zpve_corrections.

The borrow must be strictly READ-ONLY on the be_hess datasets: it may only call
check_collection_exists / fetch_reaction_entries / get_zpve_mol, always with
on_imaginary="return", and must never submit or mutate anything.
"""
import logging

import pandas as pd
import pytest
import qcelemental
from unittest.mock import MagicMock

from beep.adapters import qcfractal_adapter as qcf
from beep.core import mbe_be_tools as bt

HARTREE2KCAL = qcelemental.constants.hartree2kcalmol
LOGGER = logging.getLogger("beep")


def _rxn_frame():
    """be_nocp rows: dimer (+1) and two fragments (-1) per site."""
    return pd.DataFrame([
        {"name": "cluster-a", "molecule": 10, "coefficient": 1.0},
        {"name": "cluster-a", "molecule": 11, "coefficient": -1.0},
        {"name": "cluster-a", "molecule": 12, "coefficient": -1.0},
        {"name": "cluster-b", "molecule": 20, "coefficient": 1.0},
        {"name": "cluster-b", "molecule": 21, "coefficient": -1.0},
        {"name": "cluster-b", "molecule": 22, "coefficient": -1.0},
    ])


def _patch(monkeypatch, zpve_map, exists=True):
    monkeypatch.setattr(qcf, "check_collection_exists", lambda *a, **k: exists)
    monkeypatch.setattr(qcf, "fetch_reaction_entries", lambda *a, **k: _rxn_frame())

    calls = {"on_imaginary": []}

    def fake_zpve(client, mol, lot_opt, on_imaginary="return", imag_threshold=50.0):
        calls["on_imaginary"].append(on_imaginary)
        return zpve_map[mol]

    monkeypatch.setattr(qcf, "get_zpve_mol", fake_zpve)
    return calls


def test_zpve_math_with_scale_factor(monkeypatch):
    zpve_map = {
        10: (1.0, True), 11: (0.3, True), 12: (0.2, True),   # cluster-a: 0.5
        20: (2.0, True), 21: (0.5, True), 22: (0.5, True),   # cluster-b: 1.0
    }
    calls = _patch(monkeypatch, zpve_map)
    client = MagicMock()

    series = bt.borrow_zpve_corrections(
        client, molecule="H2CO", hessian_clusters=["cd5"], opt_lot="hf3c_minix",
        entry_names=["cluster-a", "cluster-b"], scale_factor=0.958, logger=LOGGER,
    )

    assert series["cluster-a"] == pytest.approx(0.5 * HARTREE2KCAL * 0.958)
    assert series["cluster-b"] == pytest.approx(1.0 * HARTREE2KCAL * 0.958)
    # Always read-only imaginary handling.
    assert set(calls["on_imaginary"]) == {"return"}
    # Never touched the client directly (no submit / add).
    client.add_dataset.assert_not_called()
    client.submit.assert_not_called()


def test_nan_for_missing_hessian(monkeypatch):
    # cluster-a dimer has no Hessian yet (zpve None) -> NaN; cluster-b fine.
    zpve_map = {
        10: (None, True), 11: (0.3, True), 12: (0.2, True),
        20: (2.0, True), 21: (0.5, True), 22: (0.5, True),
    }
    _patch(monkeypatch, zpve_map)
    series = bt.borrow_zpve_corrections(
        MagicMock(), molecule="H2CO", hessian_clusters=["cd5"], opt_lot="hf3c_minix",
        entry_names=["cluster-a", "cluster-b"], logger=LOGGER,
    )
    assert pd.isna(series["cluster-a"])
    assert not pd.isna(series["cluster-b"])


def test_nan_for_imaginary_dimer(monkeypatch):
    # cluster-a dimer flagged non-real (significant imaginary) -> NaN.
    zpve_map = {
        10: (1.0, False), 11: (0.3, True), 12: (0.2, True),
        20: (2.0, True), 21: (0.5, True), 22: (0.5, True),
    }
    _patch(monkeypatch, zpve_map)
    series = bt.borrow_zpve_corrections(
        MagicMock(), molecule="H2CO", hessian_clusters=["cd5"], opt_lot="hf3c_minix",
        entry_names=["cluster-a", "cluster-b"], logger=LOGGER,
    )
    assert pd.isna(series["cluster-a"])
    assert not pd.isna(series["cluster-b"])


def test_all_nan_when_dataset_absent(monkeypatch):
    zpve_map = {}
    _patch(monkeypatch, zpve_map, exists=False)
    series = bt.borrow_zpve_corrections(
        MagicMock(), molecule="H2CO", hessian_clusters=["cd5"], opt_lot="hf3c_minix",
        entry_names=["cluster-a", "cluster-b"], logger=LOGGER,
    )
    assert series.isna().all()


def test_nan_for_unexpected_stoichiometry(monkeypatch):
    zpve_map = {10: (1.0, True)}

    def frame_one_row(*a, **k):
        return pd.DataFrame([{"name": "cluster-a", "molecule": 10, "coefficient": 1.0}])

    monkeypatch.setattr(qcf, "check_collection_exists", lambda *a, **k: True)
    monkeypatch.setattr(qcf, "fetch_reaction_entries", frame_one_row)
    monkeypatch.setattr(qcf, "get_zpve_mol", lambda *a, **k: (1.0, True))

    series = bt.borrow_zpve_corrections(
        MagicMock(), molecule="H2CO", hessian_clusters=["cd5"], opt_lot="hf3c_minix",
        entry_names=["cluster-a"], logger=LOGGER,
    )
    assert pd.isna(series["cluster-a"])
