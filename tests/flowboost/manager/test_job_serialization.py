"""Tests for JobV2 serialization round-trip."""

from datetime import datetime, timezone
from pathlib import Path

from flowboost.manager.manager import JobV2


class TestJobV2RoundTrip:
    def test_basic_round_trip(self):
        job = JobV2(id="abc123", name="test_job", wdir=Path("/tmp/case"))
        restored = JobV2.from_dict(job.to_dict())

        assert restored.id == job.id
        assert restored.name == job.name
        assert restored.wdir == job.wdir

    def test_preserves_created_at(self):
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        job = JobV2(id="abc", name="test", wdir=Path("/tmp"), created_at=ts)
        restored = JobV2.from_dict(job.to_dict())
        assert restored.created_at == ts

    def test_wdir_posix_serialization(self):
        job = JobV2(id="abc", name="test", wdir=Path("/home/user/cases/case_01"))
        d = job.to_dict()
        assert d["wdir"] == "/home/user/cases/case_01"
        restored = JobV2.from_dict(d)
        assert restored.wdir == Path("/home/user/cases/case_01")

    def test_to_dict_keys(self):
        job = JobV2(id="abc", name="test", wdir=Path("/tmp"))
        d = job.to_dict()
        assert set(d.keys()) == {"id", "name", "wdir", "created_at"}

    def test_created_at_is_iso_string(self):
        job = JobV2(id="abc", name="test", wdir=Path("/tmp"))
        d = job.to_dict()
        # Should be parseable
        datetime.fromisoformat(d["created_at"])

    def test_runtime_returns_string(self):
        job = JobV2(id="abc", name="test", wdir=Path("/tmp"))
        assert isinstance(job.runtime(), str)
