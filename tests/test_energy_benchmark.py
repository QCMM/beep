"""Tests for beep/workflows/energy_benchmark.py."""
import logging

import pandas as pd
import pytest

from beep.workflows.energy_benchmark import (
    get_cc_keywords, log_mae_per_geometry,
    get_scf_state_keywords, cbs_spec_name, scf_state_jumps,
)


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


# --- open-shell state control across the CBS basis series ----------------

_CBS = [
    "scf_aug-cc-pvdz", "scf_aug-cc-pvtz", "scf_aug-cc-pvqz",
    "mp2_aug-cc-pvqz", "ccsd(t)_aug-cc-pvdz", "ccsd(t)_aug-cc-pvtz",
]


def test_scf_state_keywords_open_shell_projects_smallest_basis():
    """Every basis above the smallest starts from the projected smallest-basis
    solution; all bases follow UHF instabilities."""
    kw_d = get_scf_state_keywords(2, "aug-cc-pvdz", _CBS)
    kw_t = get_scf_state_keywords(2, "aug-cc-pvtz", _CBS)
    kw_q = get_scf_state_keywords(2, "aug-cc-pvqz", _CBS)
    for kw in (kw_d, kw_t, kw_q):
        assert kw["reference"] == "uhf"
        assert kw["stability_analysis"] == "follow"
    assert "basis_guess" not in kw_d
    assert kw_t["basis_guess"] == "aug-cc-pvdz"
    assert kw_q["basis_guess"] == "aug-cc-pvdz"


def test_scf_state_keywords_tight_d_family():
    cbs = [lot.replace("pv", "pv(").replace("z", "+d)z") for lot in _CBS]
    kw = get_scf_state_keywords(2, "aug-cc-pv(t+d)z", cbs)
    assert kw["basis_guess"] == "aug-cc-pv(d+d)z"


def test_scf_state_keywords_closed_shell_unchanged():
    assert get_scf_state_keywords(1, "aug-cc-pvtz", _CBS) == {}


def test_cbs_spec_name_closed_shell_unchanged_open_shell_suffixed():
    assert cbs_spec_name("scf", "aug-cc-pVTZ") == "scf_aug-cc-pvtz"
    assert cbs_spec_name("ccsd(t)", "aug-cc-pVTZ") == "ccsd(t)_aug-cc-pvtz_df"
    assert cbs_spec_name("scf", "aug-cc-pVTZ", 2) == "scf_aug-cc-pvtz_stab"
    assert cbs_spec_name("mp2", "aug-cc-pVQZ", 2) == "mp2_aug-cc-pvqz_df_stab"


def _scf_tables(ie_scf, be_scf):
    idx = ["aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz", "CBS"]
    return {
        "IE": pd.DataFrame({"SCF": ie_scf + [0.0]}, index=idx),
        "BE": pd.DataFrame({"SCF": be_scf + [0.0]}, index=idx),
    }


def test_scf_state_jumps_flags_oh_w3_0005():
    """OH_W3_01_0005: aVDZ UHF in a different state than aVTZ/aVQZ."""
    jumps = scf_state_jumps(_scf_tables([-2.50, -0.01, 0.01], [-2.0, 0.4, 0.5]))
    assert len(jumps) == 2
    assert jumps[0].startswith("IE aug-cc-pvdz->aug-cc-pvtz")
    assert jumps[1].startswith("BE aug-cc-pvdz->aug-cc-pvtz")


def test_scf_state_jumps_flags_last_basis():
    """CH3O_W2_01_0007: aVQZ in a different state."""
    jumps = scf_state_jumps(_scf_tables([-4.76, -4.28, 1.72], [-4.0, -3.6, -3.4]))
    assert jumps == ["IE aug-cc-pvtz->aug-cc-pvqz (-4.28 -> +1.72)"]


def test_scf_state_jumps_smooth_passes():
    assert scf_state_jumps(_scf_tables([-3.10, -2.85, -2.78], [-2.6, -2.3, -2.2])) == []
    # Threshold is respected
    assert scf_state_jumps(_scf_tables([-3.10, -2.85, -2.78], [-2.6, -2.3, -2.2]),
                           threshold=0.2) != []
