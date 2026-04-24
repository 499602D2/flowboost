"""Tests for DictionaryLink chaining and immutability — no OpenFOAM CLI needed."""

import numpy as np

from flowboost.openfoam.dictionary import Dictionary, DictionaryLink, Entry


class TestDictionaryLinkChaining:
    def test_entry_returns_new_link(self):
        original = Dictionary.link("constant/physicalProperties")
        chained = original.entry("reactants")
        assert chained is not original

    def test_original_not_mutated(self):
        original = Dictionary.link("0/U")
        original.entry("boundaryField")
        assert original.entry_path == ""

    def test_single_entry(self):
        link = Dictionary.link("0/U").entry("boundaryField")
        assert link.path == "0/U"
        assert link.entry_path == "boundaryField"

    def test_chained_entries(self):
        link = Dictionary.link("0/U").entry("boundaryField").entry("inlet")
        assert link.entry_path == "boundaryField/inlet"

    def test_deep_chain(self):
        link = (
            Dictionary.link("constant/cloudProperties")
            .entry("subModels")
            .entry("injectionModels")
            .entry("model1")
            .entry("SOI")
        )
        assert link.entry_path == "subModels/injectionModels/model1/SOI"
        assert link.path == "constant/cloudProperties"


class TestDictionaryLinkIndex:
    def test_index_appends_bracket_notation(self):
        link = Dictionary.link("0/U").entry("internalField").index(2)
        assert "[2]" in link.entry_path

    def test_index_preserves_path(self):
        link = Dictionary.link("0/U").entry("internalField").index(0)
        assert link.path == "0/U"

    def test_reader_resolves_indexed_entry_to_scalar(self, monkeypatch, tmp_path):
        entry = Entry(Dictionary.reader(tmp_path / "dummyDict"), key="internalField")
        entry._value = np.array([10, 20, 30])

        monkeypatch.setattr(
            "flowboost.openfoam.dictionary.DictionaryReader.entry",
            lambda self, entry_path: entry if entry_path == "internalField" else None,
        )

        value = Dictionary.link("0/U").entry("internalField").index(1).reader(tmp_path)

        assert value == 20


class TestDictionaryLinkStr:
    def test_str_no_entry(self):
        link = DictionaryLink("0/U")
        assert "0/U" in str(link)

    def test_str_with_entry(self):
        link = Dictionary.link("0/U").entry("boundaryField")
        s = str(link)
        assert "0/U" in s
        assert "boundaryField" in s
