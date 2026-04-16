"""Unit tests for Case state/persistence — no OpenFOAM CLI needed."""

from datetime import datetime, timezone

import pytest

from flowboost.openfoam.case import Case, Status


@pytest.fixture
def case(tmp_path):
    d = tmp_path / "test_case"
    d.mkdir()
    return Case(d)


class TestCaseState:
    def test_state_filters_none(self, case):
        state = case.state()
        assert "submitted_at" not in state  # None by default
        assert "model_predictions_by_objective" not in state

    def test_state_preserves_false(self, case):
        case.success = False
        state = case.state()
        assert "success" in state
        assert state["success"] is False

    def test_state_preserves_true(self, case):
        case.success = True
        state = case.state()
        assert state["success"] is True

    def test_state_omits_success_none(self, case):
        assert case.success is None
        state = case.state()
        assert "success" not in state

    def test_state_includes_required_fields(self, case):
        state = case.state()
        assert "name" in state
        assert "id" in state
        assert "path" in state
        assert "status" in state
        assert "created_at" in state


class TestCasePersistRestore:
    def test_round_trip_basic(self, case):
        case.persist_to_file()
        restored = Case.restore_from_file(case.path)

        assert restored.name == case.name
        assert restored.id == case.id
        assert restored.status == case.status

    def test_round_trip_preserves_success_false(self, case):
        case.success = False
        case.persist_to_file()
        restored = Case.restore_from_file(case.path)
        assert restored.success is False

    def test_round_trip_preserves_success_none(self, case):
        assert case.success is None
        case.persist_to_file()
        restored = Case.restore_from_file(case.path)
        assert restored.success is None

    def test_round_trip_preserves_status(self, case):
        case.status = Status.FINISHED
        case.persist_to_file()
        restored = Case.restore_from_file(case.path)
        assert restored.status == Status.FINISHED

    def test_round_trip_preserves_generation_index(self, case):
        case._generation_index = "003.02"
        case.persist_to_file()
        restored = Case.restore_from_file(case.path)
        assert restored._generation_index == "003.02"

    def test_round_trip_preserves_created_at(self, case):
        case.persist_to_file()
        restored = Case.restore_from_file(case.path)
        # fromisoformat may lose timezone info, so compare timestamps
        assert abs(
            restored._created_at.timestamp() - case._created_at.timestamp()
        ) < 1.0

    def test_restore_missing_file_raises(self, tmp_path):
        d = tmp_path / "nonexistent_case"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            Case.restore_from_file(d)

    def test_try_restoring_with_file(self, case):
        case.success = True
        case.persist_to_file()
        restored = Case.try_restoring(case.path)
        assert restored.success is True

    def test_try_restoring_without_file(self, case):
        restored = Case.try_restoring(case.path)
        assert restored.success is None  # fresh Case


class TestCaseMetadata:
    def test_update_metadata_creates_file(self, case):
        case.update_metadata({"key": "value"})
        data = case.read_metadata()
        assert data["key"] == "value"

    def test_update_metadata_with_header(self, case):
        case.update_metadata({"key": "value"}, entry_header="section")
        data = case.read_metadata()
        assert data["section"]["key"] == "value"

    def test_update_metadata_preserves_existing_keys(self, case):
        case.update_metadata({"a": 1}, entry_header="section")
        case.update_metadata({"b": 2}, entry_header="section")
        data = case.read_metadata()
        assert data["section"]["a"] == 1
        assert data["section"]["b"] == 2

    def test_update_metadata_overwrites_existing_key(self, case):
        case.update_metadata({"key": "old"}, entry_header="section")
        case.update_metadata({"key": "new"}, entry_header="section")
        data = case.read_metadata()
        assert data["section"]["key"] == "new"

    def test_read_metadata_returns_none_when_missing(self, case):
        assert case.read_metadata() is None

    def test_update_metadata_no_header(self, case):
        case.update_metadata({"top_level": 42})
        data = case.read_metadata()
        assert data["top_level"] == 42


class TestCaseMarkFailed:
    def test_mark_failed_sets_success(self, case):
        case.mark_failed()
        assert case.success is False

    def test_mark_failed_persists(self, case):
        case.mark_failed()
        restored = Case.restore_from_file(case.path)
        assert restored.success is False


class TestCaseInit:
    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Case(tmp_path / "does_not_exist")

    def test_id_is_deterministic(self, tmp_path):
        d = tmp_path / "det_case"
        d.mkdir()
        c1 = Case(d)
        c2 = Case(d)
        assert c1.id == c2.id

    def test_default_status(self, case):
        assert case.status == Status.NOT_SUBMITTED
        assert case.success is None
