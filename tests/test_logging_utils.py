"""Tests for beep/core/logging_utils.py."""
import logging

import pandas as pd
import pytest

from beep.core.logging_utils import (
    setup_logging,
    log_formatted_list,
    dict_to_log,
    padded_log,
    log_progress,
    log_dataframe_averages,
    log_energy_mae,
    log_dataframe,
    log_dictionary,
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
    # Minimal DataFrame with required columns
    df = pd.DataFrame({
        "Mean_Eb_all_dft": [-1.0, -2.0],
        "StdDev_all_dft": [0.1, 0.2],
    })
    content = write_energy_log(df, "H2O", existing_content="")
    assert content.startswith(" ")  # Header starts with padding
    assert "BE Average" in content


# ---------------------------------------------------------------------------
# dict_to_log
# ---------------------------------------------------------------------------

def test_dict_to_log_basic(caplog):
    logger = logging.getLogger("beep_log_test_dict")
    logger.setLevel(logging.INFO)
    data = {"GGA": ["PBE", "BLYP"], "Hybrid": ["B3LYP"]}
    with caplog.at_level(logging.INFO, logger="beep_log_test_dict"):
        dict_to_log(logger, data)
    assert "XC Functional Type: GGA" in caplog.text
    assert "PBE" in caplog.text
    assert "XC Functional Type: Hybrid" in caplog.text
    assert "B3LYP" in caplog.text


def test_dict_to_log_empty(caplog):
    logger = logging.getLogger("beep_log_test_dict_empty")
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="beep_log_test_dict_empty"):
        dict_to_log(logger, {})
    # No crash, no output for empty dict
    assert "XC Functional Type" not in caplog.text


# ---------------------------------------------------------------------------
# log_dataframe_averages
# ---------------------------------------------------------------------------

def test_log_dataframe_averages_basic(caplog):
    logger = logging.getLogger("beep_log_test_df_avg")
    logger.setLevel(logging.INFO)
    # Columns must follow the pattern: prefix_group1_group2_theory
    df = pd.DataFrame({
        "rmsd_GGA_DZ_PBE": [0.1, 0.2, 0.15],
        "rmsd_GGA_DZ_BLYP": [0.3, 0.4, 0.35],
        "dummy_first_col": [1, 2, 3],  # will be dropped (first col)
    })
    # Reorder so dummy is first
    df = df[["dummy_first_col", "rmsd_GGA_DZ_PBE", "rmsd_GGA_DZ_BLYP"]]
    with caplog.at_level(logging.INFO, logger="beep_log_test_df_avg"):
        log_dataframe_averages(logger, df)
    assert "Functional Group: GGA_DZ" in caplog.text
    assert "Summary" in caplog.text


# ---------------------------------------------------------------------------
# log_energy_mae
# ---------------------------------------------------------------------------

def test_log_energy_mae_returns_dataframe(caplog):
    logger = logging.getLogger("beep_log_test_mae")
    logger.setLevel(logging.INFO)
    # Index format: site_method_basis
    df = pd.DataFrame({
        "method_A": [0.5, -0.3, 0.1, -0.2],
        "method_B": [-1.0, 0.8, -0.5, 0.3],
    }, index=["site1_pbe_dz", "site2_pbe_dz", "site1_b3lyp_tz", "site2_b3lyp_tz"])
    with caplog.at_level(logging.INFO, logger="beep_log_test_mae"):
        result = log_energy_mae(logger, df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # two method_basis groups: pbe/dz and b3lyp/tz


def test_log_energy_mae_values(caplog):
    logger = logging.getLogger("beep_log_test_mae_vals")
    logger.setLevel(logging.INFO)
    # Single method_basis group, known MAE
    df = pd.DataFrame({
        "col1": [1.0, -1.0],
    }, index=["s1_pbe_dz", "s2_pbe_dz"])
    with caplog.at_level(logging.INFO, logger="beep_log_test_mae_vals"):
        result = log_energy_mae(logger, df)
    # MAE of abs([1.0, -1.0]).mean() = 1.0
    assert abs(result.loc["pbe/dz", "col1"] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# log_dataframe
# ---------------------------------------------------------------------------

def test_log_dataframe_basic(caplog):
    logger = logging.getLogger("beep_log_test_logdf")
    logger.setLevel(logging.INFO)
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with caplog.at_level(logging.INFO, logger="beep_log_test_logdf"):
        log_dataframe(logger, df, title="Test DF")
    assert "Test DF" in caplog.text
    assert "1" in caplog.text
    assert "4" in caplog.text


def test_log_dataframe_default_title(caplog):
    logger = logging.getLogger("beep_log_test_logdf_default")
    logger.setLevel(logging.INFO)
    df = pd.DataFrame({"X": [10]})
    with caplog.at_level(logging.INFO, logger="beep_log_test_logdf_default"):
        log_dataframe(logger, df)
    assert "DataFrame" in caplog.text


# ---------------------------------------------------------------------------
# log_dictionary
# ---------------------------------------------------------------------------

def test_log_dictionary_basic(caplog):
    logger = logging.getLogger("beep_log_test_logdict")
    logger.setLevel(logging.INFO)
    d = {"PBE": (1.5, 2.3, 0.98), "B3LYP": (0.8, 1.1, 0.99)}
    with caplog.at_level(logging.INFO, logger="beep_log_test_logdict"):
        log_dictionary(logger, d, title="Fit Results")
    assert "Fit Results" in caplog.text
    assert "PBE: m = 1.5, n = 2.3, R^2 = 0.98" in caplog.text
    assert "B3LYP: m = 0.8, n = 1.1, R^2 = 0.99" in caplog.text


def test_log_dictionary_default_title(caplog):
    logger = logging.getLogger("beep_log_test_logdict_default")
    logger.setLevel(logging.INFO)
    d = {"x": (1, 2, 3)}
    with caplog.at_level(logging.INFO, logger="beep_log_test_logdict_default"):
        log_dictionary(logger, d)
    assert "Dictionary:" in caplog.text
