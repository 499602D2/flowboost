from types import SimpleNamespace

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
