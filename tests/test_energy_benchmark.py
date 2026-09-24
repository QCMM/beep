"""Tests for beep/workflows/energy_benchmark.py."""
import logging

import pandas as pd
import pytest

from beep.workflows.energy_benchmark import get_cc_keywords, log_mae_per_geometry


def _capture_logger():
    logger = logging.getLogger("beep")
    logger.setLevel(logging.INFO)
    records = []

    class _Grab(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Grab()
    logger.addHandler(handler)
    return logger, handler, records


def test_log_mae_per_geometry_one_labeled_section_per_lot():
    """Regression: with N opt LOTs the results summary must print N sections,
    each labeled with its geometry LOT and computed only from that LOT's
    entries. Pre-fix, a single unlabeled table pooled every geometry's
    errors into one MAE per functional."""
    lot_a = "mpwb1k-d3bj_def2-tzvpd"
    lot_b = "gfn2-xtb"
    # Signed errors: lot_a rows → MAE 2.0; lot_b row → MAE 5.0. The pooled
    # (pre-fix) value would be 3.0 and must NOT be what either section shows.
    df_ae = pd.DataFrame(
        {"pbe/def2-tzvp": [1.0, -3.0, 5.0]},
        index=[
            f"CO_W12_1_0001_{lot_a}",
            f"CO_W12_1_0002_{lot_a}",
            f"CO_W12_1_0001_{lot_b}",
        ],
    )
    logger, handler, records = _capture_logger()
    try:
        log_mae_per_geometry(
            logger, df_ae, {"GGA": ["pbe"]}, [lot_a, lot_b], "BE MAE",
        )
    finally:
        logger.removeHandler(handler)

    text = "\n".join(records)
    assert f"BE MAE — {lot_a} geometries" in text
    assert f"BE MAE — {lot_b} geometries" in text
    # Per-section MAEs, not the pooled 3.0
    assert "2.000000" in text
    assert "5.000000" in text
    assert "3.000000" not in text
    # Section order matches the config order and each MAE sits in its section
    a_pos = text.index(f"{lot_a} geometries")
    b_pos = text.index(f"{lot_b} geometries")
    assert a_pos < text.index("2.000000") < b_pos < text.index("5.000000")


def test_log_mae_per_geometry_skips_lot_without_entries():
    """A LOT with no matching entries warns and is skipped, without touching
    the other sections."""
    df_ae = pd.DataFrame(
        {"pbe/def2-tzvp": [1.0]}, index=["CO_W12_1_0001_gfn2-xtb"],
    )
    logger, handler, records = _capture_logger()
    logger.setLevel(logging.WARNING)
    try:
        log_mae_per_geometry(
            logger, df_ae, {"GGA": ["pbe"]}, ["hf3c_minix", "gfn2-xtb"], "BE MAE",
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(logging.INFO)

    text = "\n".join(records)
    assert "No entries for geometry hf3c_minix" in text


def test_get_cc_keywords_open_shell_has_iteration_cap():
    """Open-shell DF-CCSD needs a raised cc_maxiter: DIIS oscillates around
    the residual criterion with the energy already converged (CN, CH3O)."""
    kw = get_cc_keywords(2)
    assert kw["qc_module"] == "OCC"
    assert kw["reference"] == "uhf"
    assert kw["cc_maxiter"] == 200
    # Closed shell unchanged: no UHF machinery, no cap needed
    kw1 = get_cc_keywords(1)
    assert "cc_maxiter" not in kw1
    assert "reference" not in kw1
