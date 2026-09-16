"""Tests for the MBE monitoring helpers in the QCFractal adapter.

Ported from beep-mbe's test_monitor.py; targets the renamed adapter functions
``wait_for_manybody_completion`` / ``wait_for_dataset_records`` and asserts the
existing ``wait_for_completion`` is left untouched. Uses injected sleep_fn /
time_fn so no real time passes.
"""
import logging
from typing import Dict, List, Optional

import pytest

from beep.adapters.qcfractal_adapter import (
    wait_for_manybody_completion,
    wait_for_dataset_records,
    ManybodyMonitorResult,
)


class DummyRecord:
    def __init__(self, status, children_status: Optional[Dict[object, int]]):
        self.status = status
        self.children_status = children_status


class DummyStatus:
    def __init__(self, name: str):
        self.name = name


class DummyDataset:
    def __init__(self, snapshots: List[Dict[str, DummyRecord]]):
        self._snapshots = snapshots
        self._index = 0

    def detailed_status(self):
        return [("entry", "spec", DummyStatus("waiting"))]

    def get_record(self, entry_name, specification_name, **kwargs):
        snapshot = self._snapshots[min(self._index, len(self._snapshots) - 1)]
        return snapshot.get(entry_name)

    def advance(self):
        if self._index < len(self._snapshots) - 1:
            self._index += 1


class DummyClient:
    def __init__(self, dataset: DummyDataset):
        self._dataset = dataset

    def get_dataset(self, dataset_type, dataset_name):
        return self._dataset


class TimeController:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_wait_for_manybody_completion_logs_children_status(caplog):
    snapshots = [
        {
            "entry-a": DummyRecord(
                DummyStatus("waiting"),
                {DummyStatus("waiting"): 2, DummyStatus("running"): 1},
            ),
            "entry-b": DummyRecord(DummyStatus("running"), {DummyStatus("running"): 3}),
        },
        {
            "entry-a": DummyRecord(DummyStatus("complete"), {DummyStatus("complete"): 3}),
            "entry-b": DummyRecord(DummyStatus("complete"), {DummyStatus("complete"): 3}),
        },
    ]
    dataset = DummyDataset(snapshots)
    client = DummyClient(dataset)
    time_ctrl = TimeController()

    def sleep_fn(seconds):
        dataset.advance()
        time_ctrl.sleep(seconds)

    # The "beep" logger is configured with propagate=False elsewhere, so
    # caplog's root handler won't see its records; attach directly.
    beep_logger = logging.getLogger("beep")
    beep_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="beep"):
            result = wait_for_manybody_completion(
                client=client,
                dataset_name="dataset",
                spec_name="spec",
                entry_names=["entry-a", "entry-b"],
                poll_interval_s=1,
                max_wait_s=None,
                sleep_fn=sleep_fn,
                time_fn=time_ctrl.time,
            )
    finally:
        beep_logger.removeHandler(caplog.handler)

    assert isinstance(result, ManybodyMonitorResult)
    assert result.polls == 2
    assert result.per_entry_final_status["entry-a"] == "COMPLETE"
    assert (
        "entry=entry-a status=WAITING children_total=3 waiting=2 running=1 complete=0 error=0"
        in caplog.text
    )
    assert (
        "entry=entry-b status=RUNNING children_total=3 waiting=0 running=3 complete=0 error=0"
        in caplog.text
    )


def test_wait_for_manybody_completion_records_errors():
    snapshots = [
        {
            "entry-a": DummyRecord(DummyStatus("complete"), {DummyStatus("complete"): 1}),
            "entry-b": DummyRecord(DummyStatus("error"), {DummyStatus("error"): 1}),
        }
    ]
    client = DummyClient(DummyDataset(snapshots))
    time_ctrl = TimeController()

    result = wait_for_manybody_completion(
        client=client,
        dataset_name="dataset",
        spec_name="spec",
        entry_names=["entry-a", "entry-b"],
        poll_interval_s=1,
        max_wait_s=None,
        sleep_fn=time_ctrl.sleep,
        time_fn=time_ctrl.time,
    )

    assert result.n_error == 1
    assert result.errored_entries == ["entry-b"]


def test_wait_for_manybody_completion_max_wait_timeout():
    snapshots = [
        {"entry-a": DummyRecord(DummyStatus("running"), {DummyStatus("running"): 1})}
    ]
    client = DummyClient(DummyDataset(snapshots))
    time_ctrl = TimeController()

    result = wait_for_manybody_completion(
        client=client,
        dataset_name="dataset",
        spec_name="spec",
        entry_names=["entry-a"],
        poll_interval_s=5,
        max_wait_s=3,
        sleep_fn=time_ctrl.sleep,
        time_fn=time_ctrl.time,
    )

    assert result.timed_out is True


def test_wait_for_dataset_records_terminates_and_reports():
    snapshots = [
        {"m": DummyRecord(DummyStatus("running"), None)},
        {"m": DummyRecord(DummyStatus("complete"), None)},
    ]
    dataset = DummyDataset(snapshots)
    time_ctrl = TimeController()

    def sleep_fn(seconds):
        dataset.advance()
        time_ctrl.sleep(seconds)

    statuses, timed_out = wait_for_dataset_records(
        dataset,
        entry_names=["m"],
        specification_names=["monomer_spec"],
        poll_interval=1,
        max_wait=None,
        sleep_fn=sleep_fn,
        time_fn=time_ctrl.time,
    )

    assert timed_out is False
    assert statuses[("m", "monomer_spec")] == "COMPLETE"


def test_existing_wait_for_completion_is_untouched():
    """The frozen reaction-route monitor must still expose its original signature."""
    import inspect
    from beep.adapters.qcfractal_adapter import wait_for_completion

    params = list(inspect.signature(wait_for_completion).parameters)
    # Original signature: (client, pid_list, frequency, logger, max_wait=...)
    assert params[:4] == ["client", "pid_list", "frequency", "logger"]
