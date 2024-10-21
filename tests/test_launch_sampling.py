import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import logging
import qcelemental as qcel
from argparse import Namespace
from pathlib import Path
from qcelemental.models.molecule import Molecule
from beep.errors import DatasetNotFound, LevelOfTheoryNotFound
from workflows.launch_sampling import (
parse_arguments,
sampling_args,
check_collection_existence,
check_optimized_molecule,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG)

def test_parse_arguments(monkeypatch):
    test_args = [
        "--client_address", "test_address:1234",
        "--username", "test_user",
        "--password", "test_password",
        "--molecule", "H2O",
        "--surface-model-collection", "Test_Water_22",
        "--small-molecule-collection", "Test_Small_molecules",
        "--sampling-shell", "2.5",
        "--total-binding-sites", "300",
        "--sampling-condition", "fine",
        "--sampling-level-of-theory", "mp2", "def2-TZVP", "psi4",
        "--refinement-level-of-theory", "ccsd", "aug-cc-pVTZ", "gaussian",
        "--rmsd-value", "0.5",
        "--rmsd-symmetry",
        "--store-initial-structures",
        "--sampling-tag", "test_tag",
        "--keyword-id", "123"
    ]

    # Set sys.argv to test_args
    monkeypatch.setattr('sys.argv', ['script_name'] + test_args)

    # Call the function and check if it returns the expected results
    args = parse_arguments()

    assert args.client_address == "test_address:1234"
    assert args.username == "test_user"
    assert args.password == "test_password"
    assert args.molecule == "H2O"
    assert args.surface_model_collection == "Test_Water_22"
    assert args.small_molecule_collection == "Test_Small_molecules"
    assert args.sampling_shell == 2.5
    assert args.total_binding_sites == 300
    assert args.sampling_condition == "fine"
    assert args.sampling_level_of_theory == ["mp2", "def2-TZVP", "psi4"]
    assert args.refinement_level_of_theory == ["ccsd", "aug-cc-pVTZ", "gaussian"]
    assert args.rmsd_value == 0.5
    assert args.rmsd_symmetry is True
    assert args.store_initial_structures is True
    assert args.sampling_tag == "test_tag"
    assert args.keyword_id == 123

def test_sampling_args():
    # Create a mock Namespace object simulating parsed command line arguments
    args = Namespace(
        sampling_level_of_theory=["test_method", "test_basis", "test_program"],
        sampling_tag="test_tag",
        keyword_id=123,
        rmsd_symmetry=True,
        store_initial_structures=False,
        rmsd_value=0.5,
        sampling_shell=2.5,
        sampling_condition="normal"
    )

    # Call the sampling_args function with this mock object
    result = sampling_args(args)

    # Assert that the returned dictionary has the correct values
    expected_result = {
        "method": "test_method",
        "basis": "test_basis",
        "program": "test_program",
        "tag": "test_tag",
        "kw_id": 123,
        "rmsd_symm": True,
        "store_initial": False,
        "rmsd_val": 0.5,
        "sampling_shell": 2.5,
        "sampling_condition": "normal"
    }

    assert result == expected_result
   
# Mock the client and its get_collection method behavior
@patch('workflows.launch_sampling.FractalClient')
def test_check_collection_existence(mock_client):
    # Create a MagicMock for the get_collection method
    mock_client.get_collection = MagicMock()

    # Define the collections that should exist
    existing_collections = ['existing_collection_1', 'existing_collection_2']

    # Set up the get_collection method to only raise a KeyError for non-existing collections
    def get_collection_side_effect(collection_type, name):
        if name not in existing_collections:
            raise KeyError
        return MagicMock()  # Return a mock object for existing collections

    # Apply the side effect to the mock
    mock_client.get_collection.side_effect = get_collection_side_effect

    # Call the function with an existing collection, should pass without error
    check_collection_existence(mock_client, 'existing_collection_1')

    # Call the function with a non-existing collection, should raise DatasetNotFound
    with pytest.raises(DatasetNotFound):
        check_collection_existence(mock_client, 'non_existing_collection') 

## Mock the OptimizationDataset and its get_record method
#@patch('scripts.launch_sampling.OptimizationDataset')
#def test_check_optimized_molecule(mock_ds):
#    # Create a MagicMock for the get_record method
#    mock_ds.get_record = MagicMock()
#
#    # Define the mock optimization record with a 'COMPLETE' status
#    complete_record = MagicMock()
#    complete_record.status = "COMPLETE"
#
#    # Define the side effect function for get_record
#    def get_record_side_effect(mol, opt_lot):
#        if mol == "existing_molecule":
#            return complete_record
#        else:
#            raise KeyError
#
#    # Apply the side effect to the mock
#    mock_ds.get_record.side_effect = get_record_side_effect
#
#    # Existing molecule with 'COMPLETE' status should not raise an exception
#    check_optimized_molecule(mock_ds, "test_level_of_theory", ["existing_molecule"])
#
#    # Non-existing molecule should raise LevelOfTheoryNotFound
#    with pytest.raises(LevelOfTheoryNotFound):
#        check_optimized_molecule(mock_ds, "test_level_of_theory", ["non_existing_molecule"])
#
#    # Incomplete record should raise ValueError
#    incomplete_record = MagicMock()
#    incomplete_record.status = "INCOMPLETE"
#    mock_ds.get_record.return_value = incomplete_record
#    with pytest.raises(ValueError):
#        check_optimized_molecule(mock_ds, "test_level_of_theory", ["existing_molecule"])
#
#    # Error record should also raise ValueError
#    error_record = MagicMock()
#    error_record.status = "ERROR"
#    mock_ds.get_record.return_value = error_record
#    with pytest.raises(ValueError):
#        check_optimized_molecule(mock_ds, "test_level_of_theory", ["existing_molecule"])

