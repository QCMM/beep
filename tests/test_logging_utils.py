"""Tests for beep/core/logging_utils.py."""
import logging

import pytest

from beep.core.logging_utils import (
    setup_logging,
    log_formatted_list,
    padded_log,
    log_progress,
    write_energy_log,
)


def test_setup_logging_returns_logger(tmp_path):
    logger = setup_logging(str(tmp_path / "test"), "mol")
    assert isinstance(logger, logging.Logger)
    # Clean up handlers to avoid leaking into other tests
    logger.handlers.clear()


def test_log_formatted_list_basic(caplog):
    logger = logging.getLogger("beep_log_test_basic")
    logger.setLevel(logging.INFO)
    items = ["A", "B", "C", "D", "E"]
    with caplog.at_level(logging.INFO, logger="beep_log_test_basic"):
        log_formatted_list(logger, items, "Test list")
    assert "Test list" in caplog.text


def test_log_formatted_list_invalid_maxrows():
    logger = logging.getLogger("beep_log_test_maxrows")
    with pytest.raises(ValueError):
        log_formatted_list(logger, ["a"], "desc", max_rows=0)


def test_padded_log_output(caplog):
    logger = logging.getLogger("beep_log_test_padded")
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="beep_log_test_padded"):
        padded_log(logger, "Hello World")
    assert "Hello World" in caplog.text


def test_log_progress_complete(caplog):
    logger = logging.getLogger("beep_log_test_progress")
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="beep_log_test_progress"):
        log_progress(logger, 10, 10)
    assert "=" in caplog.text
    assert "10/10" in caplog.text


def test_log_progress_out_of_bounds():
    logger = logging.getLogger("beep_log_test_oob")
    with pytest.raises(ValueError):
        log_progress(logger, 11, 10)


def test_write_energy_log_header():
    import pandas as pd
    # Minimal DataFrame with required columns
    df = pd.DataFrame({
        "Mean_Eb_all_dft": [-1.0, -2.0],
        "StdDev_all_dft": [0.1, 0.2],
    })
    content = write_energy_log(df, "H2O", existing_content="")
    assert content.startswith(" ")  # Header starts with padding
    assert "BE Average" in content
