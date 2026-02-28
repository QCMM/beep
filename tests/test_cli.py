"""Tests for beep/cli.py."""
import json
import sys
import types
from unittest.mock import patch, MagicMock

import pytest

from beep.cli import main, WORKFLOW_MODELS


def test_main_missing_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["beep"])
    with pytest.raises(SystemExit):
        main()


def test_main_nonexistent_file(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["beep", "--config", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_main_invalid_workflow(monkeypatch, tmp_path):
    cfg_file = tmp_path / "bad.json"
    cfg_file.write_text(json.dumps({"workflow": "bogus"}))
    monkeypatch.setattr(sys, "argv", ["beep", "--config", str(cfg_file)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("beep.cli.connect")
def test_main_valid_dispatch(mock_connect, monkeypatch, tmp_path):
    mock_client = MagicMock()
    mock_connect.return_value = mock_client

    cfg_file = tmp_path / "sampling.json"
    cfg_file.write_text(json.dumps({
        "workflow": "sampling",
        "molecule": "CO",
        "sampling_level_of_theory": {"method": "gfn2-xtb", "program": "xtb"},
        "refinement_level_of_theory": {"method": "bhandhlyp", "basis": "def2-svp", "program": "psi4"},
    }))
    monkeypatch.setattr(sys, "argv", ["beep", "--config", str(cfg_file)])

    # Create a fake module with a run function to intercept the lazy import
    mock_run = MagicMock()
    fake_module = types.ModuleType("beep.workflows.sampling")
    fake_module.run = mock_run

    with patch.dict(sys.modules, {"beep.workflows.sampling": fake_module}):
        main()

    mock_connect.assert_called_once()
    mock_run.assert_called_once()


def test_workflow_models_keys():
    expected = {"sampling", "be_hess", "extract", "pre_exp", "geom_benchmark", "energy_benchmark"}
    assert set(WORKFLOW_MODELS.keys()) == expected
