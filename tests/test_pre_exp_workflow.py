"""Tests for beep/workflows/pre_exp.py (offline, adapter fully mocked)."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from beep.models.pre_exp import PreExpConfig
from beep.workflows import pre_exp as wf

H2O_XYZ = "O  0.000  0.000  0.117\nH  0.000  0.756 -0.469\nH  0.000 -0.756 -0.469"


def _run(config, tmp_path, monkeypatch, entry_names=("H2O",)):
    monkeypatch.chdir(tmp_path)
    ds = MagicMock()
    ds.entry_names = list(entry_names)
    with patch.object(wf.qcf, "get_collection", return_value=ds), \
         patch.object(wf.qcf, "check_collection_existence"), \
         patch.object(wf.qcf, "check_optimized_molecule"), \
         patch.object(wf.qcf, "get_xyz", return_value=H2O_XYZ), \
         patch.object(wf, "get_sym_num", return_value=("C2v", 2)):
        wf.run(config, MagicMock())


def _read_table(tmp_path, molecule, folder=None):
    # All per-molecule tables go under <cwd>/<first molecule>/data/.
    path = tmp_path / (folder or molecule) / "data" / f"v_{molecule}.dat"
    assert path.exists()
    return pd.read_csv(path, sep=r"\s+")


def test_temperature_range_is_inclusive_of_t_max(tmp_path, monkeypatch):
    """Regression: range(T_min, T_max, T_step) excluded T_max while the log
    message said the range ran up to T_max."""
    cfg = PreExpConfig(workflow="pre_exp", molecule=["H2O"],
                       range_of_temperature=[10, 20], temperature_step=5)
    _run(cfg, tmp_path, monkeypatch)
    table = _read_table(tmp_path, "H2O")
    assert table["T"].tolist() == [10, 15, 20]


def test_empty_molecule_list_means_all_in_collection(tmp_path, monkeypatch):
    """Regression: ``molecule: []`` slipped past the ``is None`` check and
    produced no output at all; it must behave like None (all molecules)."""
    cfg = PreExpConfig(workflow="pre_exp", molecule=[],
                       range_of_temperature=[10, 12], temperature_step=1)
    _run(cfg, tmp_path, monkeypatch, entry_names=("H2O", "CO"))
    for name in ("H2O", "CO"):
        table = _read_table(tmp_path, name, folder="H2O")
        assert table["T"].tolist() == [10, 11, 12]
        assert (table["v"] > 0).all()
