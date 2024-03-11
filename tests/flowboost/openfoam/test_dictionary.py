import logging
from pathlib import Path
from typing import Callable, Generator

import pytest

from flowboost.openfoam.dictionary import Dictionary, DictionaryReader
from flowboost.openfoam.interface import FOAM


@pytest.fixture
def tutorial_dictionary_reader(foam_in_env, tmp_path) -> Callable[[str], DictionaryReader]:
    """Fixture to create a Dictionary reader for testing."""
    def _reader(path: str) -> DictionaryReader:
        return DictionaryReader(FOAM.tutorials() / path)

    return _reader


@pytest.fixture
def foam_tutorial_dict_paths(foam_in_env) -> Generator[Path, None, None]:
    """Generates FOAM dictionary paths from tutorials folder."""
    tutorials_path = FOAM.tutorials()
    assert tutorials_path.exists(
    ), f"Tutorials folder does not exist: {tutorials_path}"

    for constant_system_folder in tutorials_path.rglob('*'):
        if constant_system_folder.name in ('constant', 'system'):
            for foam_file in constant_system_folder.iterdir():
                if foam_file.is_file() and foam_file.suffix != '.dat':
                    yield foam_file


@pytest.mark.parametrize("limit", [None, 10])
def test_init_on_all_tutorials(foam_tutorial_dict_paths, limit):
    """Test initializing Dictionaries on all tutorial paths, optionally limited."""
    for i, foam_file in enumerate(foam_tutorial_dict_paths):
        if limit and i >= limit:
            break

        reader = Dictionary.reader(foam_file)
        reader.preload()

        # Consider asserting something here to make it a valid test


def test_entry_read_and_write(tutorial_dictionary_reader):
    """Test reading and writing entries in a dictionary."""
    foam_file = "XiFluid/moriyoshiHomogeneous/moriyoshiHomogeneous/constant/physicalProperties.hydrogen"
    reader: DictionaryReader = tutorial_dictionary_reader(foam_file)

    # Test read of existing entry
    mol_weight = reader.entry("reactants/specie/molWeight")
    assert mol_weight == 16.0243, f"Unexpected molWeight value: {mol_weight}"

    # Write a new value and read back to verify the write
    new_mol_weight = 18.015
    reader.write("reactants/specie/molWeight", new_mol_weight)
    updated_mol_weight = reader.entry("reactants/specie/molWeight")
    assert updated_mol_weight == new_mol_weight, "Write operation failed"

    # Reset the value back to original
    reader.write("reactants/specie/molWeight", mol_weight)
    reset_mol_weight = reader.entry("reactants/specie/molWeight")
    assert reset_mol_weight == mol_weight, "Reset operation failed"


def test_add_and_delete_entry(tutorial_dictionary_reader):
    """Test adding and deleting entries."""
    foam_file = "XiFluid/moriyoshiHomogeneous/moriyoshiHomogeneous/constant/physicalProperties.hydrogen"
    reader: DictionaryReader = tutorial_dictionary_reader(foam_file)

    # Add a new test entry
    test_entry_key = "testEntry"
    test_value = [1, 2, 3, 4]
    reader.add(test_entry_key, test_value)

    # Verify the entry exists and has the correct value
    added_entry = reader.entry(test_entry_key)
    assert added_entry is not None, "Entry was not added successfully"
    assert added_entry == test_value, "Entry value is not as expected"

    # Delete the entry and verify it's gone
    reader.delete(test_entry_key)
    deleted_entry = reader.entry(test_entry_key)
    assert deleted_entry is None, "Entry was not deleted successfully"


def test_entry_indexing(tutorial_dictionary_reader):
    """Test indexing functionality of entries."""
    p = "XiFluid/moriyoshiHomogeneous/moriyoshiHomogeneous/constant/combustionProperties"
    reader: DictionaryReader = tutorial_dictionary_reader(p)

    # Create a test array entry
    test_array_key = "testArray"
    test_array_value = [10, 20, 30, 40]
    reader.add(test_array_key, value=test_array_value)
    added_entry = reader.entry(test_array_key)

    assert added_entry, f"Added a None-type entry? Entry={added_entry}"

    # Verify that the array can be indexed correctly
    for i, expected_value in enumerate(test_array_value):
        # Assuming the reader or entry provides a method to index into the array, like `index`
        actual_value = added_entry.index(i)
        assert actual_value == expected_value, f"Array value at index {
            i} is incorrect: expected {expected_value}, got {actual_value}"

    # Delete the test array entry after verification
    reader.delete(test_array_key)
    assert reader.entry(
        test_array_key) is None, "Failed to delete the test array entry"


def test_linking(tutorial_dictionary_reader):
    """Test linking between dictionary entries."""
    header_link = Dictionary.link(
        "constant/physicalProperties").entry("FoamFile/format")


def test_dimensioned_entry_RW(foam_in_env, test_case):
    soi_path = "subModels/injectionModels/model1/SOI"
    soi_entry = test_case.dictionary(
        "constant/cloudProperties").entry(soi_path)

    logging.info(f"SOI entry: '{soi_entry}'")
    logging.info(f"SOI entry (raw): '{soi_entry._raw_value}'")

    # TODO handling for typed values, e.g. "[SOI] 355"
    # Syntax is same as unit-typing ([1 0 0 ...])
    # Idea: strip into unit, value
    # - when printing, include both
    # - value only returns value
    # - ".unit" for accessing the prefix (can it be a suffix?)
    # - write back on conversion

    # Dimensioned types
    # https://doc.cfd.direct/openfoam/user-guide-v11/basic-file-format
