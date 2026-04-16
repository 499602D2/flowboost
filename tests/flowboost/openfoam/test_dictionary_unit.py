from types import SimpleNamespace

import pytest

from flowboost.openfoam.dictionary import Dictionary, Entry


def test_entry_set_write_dimensioned_updates_cached_name_dimension_and_value(
    monkeypatch, tmp_path
):
    entry = Entry(Dictionary(tmp_path / "physicalProperties"), key="nu")

    monkeypatch.setattr(
        "flowboost.openfoam.dictionary.run_foam_command",
        lambda cmd: SimpleNamespace(returncode=0, stderr="", stdout=""),
    )

    ok = entry.set(
        "nu [0 2 -1 0 0 0 0] 1e-5",
        write_dimensioned=True,
    )

    assert ok is True
    assert entry.name == "nu"
    assert entry.dimension == "[0 2 -1 0 0 0 0]"
    assert entry.value == 1e-5


def test_dictionary_reader_add_failed_nested_entry_leaves_cache_unchanged(
    monkeypatch, tmp_path
):
    reader = Dictionary.reader(tmp_path / "controlDict")
    reader._keywords = []

    monkeypatch.setattr(
        "flowboost.openfoam.dictionary.run_foam_command",
        lambda cmd: SimpleNamespace(returncode=1, stderr="boom", stdout=""),
    )

    added = reader.add("parent/child", 1)

    assert added is None
    assert reader._keywords == []
    assert reader.entry("parent") is None
    assert reader.entry("child") is None
    assert reader.entry("parent/child") is None


def test_dictionary_reader_add_nested_entry_rebuilds_hierarchy_after_success(
    monkeypatch, tmp_path
):
    reader = Dictionary.reader(tmp_path / "controlDict")

    def fake_run_foam_command(cmd):
        cmd = [str(part) for part in cmd]

        if "-add" in cmd:
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        if "-keywords" in cmd and "-entry" not in cmd:
            return SimpleNamespace(returncode=0, stderr="", stdout="parent\n")

        if "-keywords" in cmd and cmd[cmd.index("-entry") + 1] == "parent":
            return SimpleNamespace(returncode=0, stderr="", stdout="child\n")

        if "-value" in cmd and cmd[cmd.index("-entry") + 1] == "parent/child":
            return SimpleNamespace(returncode=0, stderr="", stdout="1")

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "flowboost.openfoam.dictionary.run_foam_command",
        fake_run_foam_command,
    )

    added = reader.add("parent/child", 1)
    resolved = reader.entry("parent/child")

    assert added is not None
    assert reader.entry("child") is None
    assert resolved is not None
    assert resolved.value == 1
    assert reader._keywords is not None
    assert [entry.key for entry in reader._keywords] == ["parent"]
    assert reader._keywords[0].keywords is not None
    assert [entry.key for entry in reader._keywords[0].keywords] == ["child"]


@pytest.mark.parametrize("override", [False, True])
def test_entry_set_non_terminating_dict_raises_explicit_error(tmp_path, override):
    entry = Entry(Dictionary(tmp_path / "controlDict"), key="parent")
    entry.terminating = False

    with pytest.raises(
        NotImplementedError, match="Dictionary-valued entry writes are not implemented"
    ):
        entry.set({"child": 1}, override=override)
