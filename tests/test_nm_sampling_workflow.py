"""Tests for beep/core/nm_sampling_workflow.py (server calls mocked)."""
from unittest.mock import MagicMock, patch

import pytest

from qcportal.record_models import RecordStatusEnum

from beep.core.nm_sampling_workflow import (
    wait_for_nm_completion, build_nm_sp_datasets,
)


def _errored_dataset(record_id=41):
    ds = MagicMock()
    ds.status.return_value = {"spec": {RecordStatusEnum.error: 1}}
    rec = MagicMock(); rec.id = record_id
    ds.iterate_records.return_value = [("e", "spec", rec)]
    return ds


@patch("beep.core.nm_sampling_workflow.time.sleep", lambda *a, **kw: None)
def test_wait_for_nm_completion_exits_when_reset_always_fails():
    """If client.reset_records raises persistently the retry budget must
    still be consumed, otherwise the loop never terminates."""
    client = MagicMock()
    client.reset_records.side_effect = RuntimeError("server says no")
    logger = MagicMock()

    complete, error = wait_for_nm_completion(
        client, {"sys": _errored_dataset()}, ["spec"], wait_interval=0,
        logger=logger, max_resets=2,
    )

    assert (complete, error) == (0, 1)
    assert client.reset_records.call_count == 2
    warns = " ".join(c.args[0] for c in logger.warning.call_args_list)
    assert "reset failed" in warns


@patch("beep.core.nm_sampling_workflow.time.sleep", lambda *a, **kw: None)
def test_wait_for_nm_completion_resets_up_to_budget_on_success():
    client = MagicMock()
    logger = MagicMock()
    complete, error = wait_for_nm_completion(
        client, {"sys": _errored_dataset()}, ["spec"], wait_interval=0,
        logger=logger, max_resets=3,
    )
    assert (complete, error) == (0, 1)
    assert client.reset_records.call_count == 3
    client.reset_records.assert_called_with([41])


@patch("beep.core.nm_sampling_workflow.qcf")
def test_build_nm_sp_datasets_passes_dft_keywords_through(mock_qcf):
    """NmSamplingConfig.qc_keywords is an inline dict (the ``dft_keyword``
    parameter here); it must reach the gradient spec unchanged (it used to
    be dropped when not a dict, with the model typed as an int keyword ID)."""
    mol = MagicMock()
    displaced = {"h2o_2": [("h2o_2_m1_p", mol, {})]}
    build_nm_sp_datasets(
        MagicMock(), displaced, ["pbe_def2-svp"], "ccsd(t)_aug-cc-pvtz",
        "ccsd(t)_aug-cc-pvtz", "psi4", {"cc_type": "df"}, "psi4",
        {"scf_type": "df", "dft_spherical_points": 590}, MagicMock(),
    )
    calls = mock_qcf.add_gradient_spec.call_args_list
    dft = [c for c in calls if c.kwargs["spec_name"] == "pbe_def2-svp"]
    assert dft[0].kwargs["keywords"] == {"scf_type": "df", "dft_spherical_points": 590}
    ref = [c for c in calls if c.kwargs["spec_name"] == "ccsd(t)_aug-cc-pvtz"]
    assert ref[0].kwargs["keywords"] == {"cc_type": "df"}
