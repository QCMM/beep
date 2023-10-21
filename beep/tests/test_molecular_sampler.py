import numpy as np
import pytest
import logging
from ..molecule_sampler import (
    generate_shift_vector,
    calculate_displacements,
    create_molecule,
    random_molecule_sampler,
    com,
)
import qcelemental as qcel
from qcelemental.models.molecule import Molecule
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG)

def load_molecules(pattern: str) -> list:
    """
    Load molecules from files matching the given pattern.

    Parameters:
    - pattern: A string pattern for the file names (e.g. "w22_*.xyz").

    Returns:
    A Dictionary of Molecule objects.
    """
    mol_dict = {}
    for f in DATA_DIR.rglob(pattern):
        mol_dict[f.stem] = Molecule.from_file(f)
    return mol_dict


def build_test_dictionary(molecules: dict, prop: float) -> dict:
    """
    Build a test dictionary for the given molecules.

    Parameters:
    - molecules: A dictionary of  Molecule objects with the key as the ids.
    - prop: The property to be asserted.

    Returns:
    A dictionary where keys are the file stems and values are tuples of the molecule and its max displacement.
    """
    return {f: (mol, prop) for f, mol in molecules.items()}


def generate_mock_shift(MAX_DISPLACEMENT: float) -> np.ndarray:
    # Generate a random vector with values between [-0.5, 0.5) for better isotropy
    vector = np.random.random_sample((3,)) - 0.5
    # Normalize to get a unit vector
    unit_vector = vector / np.linalg.norm(vector)
    mock_shift = unit_vector * MAX_DISPLACEMENT
    return mock_shift


# Loading the molecules for testing and saving them in a list
clusters_22 = load_molecules("w22_*.xyz")
clusters_60 = load_molecules("w60_*.xyz")
small_mols = load_molecules("sm*.xyz")

# Data for calculate displacements test
MAX_DISPLACEMENT_22 = 15.0
MAX_DISPLACEMENT_60 = 20.0
SAMPLING_SHELL = 2.0
NUMBER_OF_STRUCTURES = 10

water_clusters_22 = build_test_dictionary(clusters_22, MAX_DISPLACEMENT_22)
water_clusters_60 = build_test_dictionary(clusters_60, MAX_DISPLACEMENT_60)

water_clusters = water_clusters_22 | water_clusters_60


@pytest.mark.parametrize(
    "molecule, expected_max_disp",
    list(water_clusters.values()),
    ids=list(water_clusters.keys()),
)
def test_calculate_displacements(molecule: Molecule, expected_max_disp: float):
    dis_min, dis_max = calculate_displacements(molecule, SAMPLING_SHELL)

    # Assert that dis_min is less than dis_max
    assert dis_min < dis_max

    # Assert that dis_max is below the expected maximum displacement
    assert dis_max <= expected_max_disp


def test_vector_magnitude():
    dis_min = 5.0
    dis_max = 8.0
    vector = generate_shift_vector(dis_min, dis_max)
    magnitude = np.linalg.norm(vector)

    assert (
        dis_min <= magnitude <= dis_max
    ), f"Expected magnitude between {dis_min} and {dis_max}, but got {magnitude}"


def test_vector_randomness():
    dis_min = 5.0
    dis_max = 8.0
    vector1 = generate_shift_vector(dis_min, dis_max)
    vector2 = generate_shift_vector(dis_min, dis_max)
    # Check if the vectors are different
    assert not np.array_equal(
        vector1, vector2
    ), f"Expected two different vectors, but got {vector1} and {vector2}"


test_data_create_molecule = {}
test_data_create_shifted_molecule = {}
for key, vals in water_clusters.items():
    cluster, max_disp = vals
    mock_shift_vect = generate_mock_shift(max_disp)
    for n, sm in small_mols.items():
        mol_shift = sm.scramble(
            do_shift=mock_shift_vect, do_rotate=True, do_resort=False, deflection=1.0
        )[0]
        test_data_create_shifted_molecule[key + "_" + n] = (cluster, mol_shift)
        test_data_create_molecule[key + "_" + n] = (cluster, sm)


@pytest.mark.parametrize(
    "cluster, shifted_mol",
    test_data_create_shifted_molecule.values(),
    ids=test_data_create_shifted_molecule.keys(),
)
def test_create_molecules(cluster, shifted_mol, request):
    test_id = request.node.name.split("[")[1].split("]")[0]
    mol = create_molecule(cluster, shifted_mol)
    mol.to_file("test_data_output/" + test_id + ".xyz", dtype="xyz")
    assert len(mol.symbols) == len(cluster.symbols) + len(shifted_mol.symbols)


#@pytest.mark.parametrize(
#    "cluster, target_mol",
#    test_data_create_molecule.values(),
#    ids=test_data_create_molecule.keys(),
#)
#def test_random_molecule_sampler(cluster, target_mol, request):
#    test_id = request.node.name.split("[")[1].split("]")[0]
#    mol_list, debug_mol = random_molecule_sampler(
#        cluster, target_mol, NUMBER_OF_STRUCTURES, SAMPLING_SHELL,
#    debug=True)
#    assert len(mol_list) == NUMBER_OF_STRUCTURES
#    assert all(isinstance(item, Molecule) for item in mol_list)
#    debug_mol.to_file("test_data_output/sampling_mol_" + test_id + ".xyz", dtype="xyz")

@pytest.mark.parametrize(
    "cluster, target_mol",
    test_data_create_molecule.values(),
    ids=test_data_create_molecule.keys(),
)
def test_random_molecule_sampler(cluster, target_mol, request, caplog):
    test_id = request.node.name.split("[")[1].split("]")[0]
    
    caplog.set_level(logging.DEBUG)  # or logging.INFO, as per your requirements

    mol_list, debug_mol = random_molecule_sampler(
        cluster, target_mol, NUMBER_OF_STRUCTURES, SAMPLING_SHELL,
    debug=True)
    
    # Log assertions
    log_messages = [record.message for record in caplog.records]
    assert "Welcome to the molecule sampler!" in log_messages
    assert "Thank you for using Molecule sampling! Goodbye" in log_messages

    # Other assertions
    assert len(mol_list) == NUMBER_OF_STRUCTURES
    assert all(isinstance(item, Molecule) for item in mol_list)
    debug_mol.to_file("test_data_output/sampling_mol_" + test_id + ".xyz", dtype="xyz")


