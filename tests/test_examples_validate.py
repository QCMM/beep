"""Every examples/*.json must validate against the model of its workflow.

The CLI (beep/cli.py) picks the model from the ``workflow`` key via
``WORKFLOW_MODELS``; the example files are named after that key.
"""
import json
from pathlib import Path

import pytest

from beep.cli import WORKFLOW_MODELS

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.json"))


def test_examples_directory_is_not_empty():
    assert EXAMPLE_FILES, f"no example configs found in {EXAMPLES_DIR}"


def test_every_workflow_has_an_example():
    names = {p.stem for p in EXAMPLE_FILES}
    missing = set(WORKFLOW_MODELS) - names
    assert not missing, f"workflows without an examples/<workflow>.json: {sorted(missing)}"


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=[p.name for p in EXAMPLE_FILES])
def test_example_validates_against_its_workflow_model(path):
    raw = json.loads(path.read_text())
    workflow = raw.get("workflow")
    assert workflow in WORKFLOW_MODELS, f"{path.name}: unknown workflow {workflow!r}"
    assert path.stem == workflow, (
        f"{path.name}: file is named after {path.stem!r} but declares workflow {workflow!r}"
    )
    model = WORKFLOW_MODELS[workflow]
    cfg = model(**raw)   # raises pydantic.ValidationError if the example is stale
    assert cfg.workflow == workflow
