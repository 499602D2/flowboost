"""Unit tests for Data class — file header parsing and edge cases."""

from pathlib import Path

import pytest

from flowboost.openfoam.data import Data


class TestDiscoverFileHeader:
    def _write_file(self, path: Path, content: str) -> Path:
        path.write_text(content)
        return path

    def test_standard_header(self, tmp_path):
        f = self._write_file(
            tmp_path / "data.dat",
            "# Time\tcpu\tclock\n0.1\t0.5\t1.2\n0.2\t0.6\t1.3\n",
        )
        data = Data(path=tmp_path)
        header = data._discover_file_header(f)
        assert header == ["Time", "cpu", "clock"]

    def test_multiple_comment_lines(self, tmp_path):
        f = self._write_file(
            tmp_path / "data.dat",
            "# Description of output\n# Time\tcpu\tclock\n0.1\t0.5\t1.2\n",
        )
        data = Data(path=tmp_path)
        header = data._discover_file_header(f)
        # Should return the LAST comment line before data
        assert header == ["Time", "cpu", "clock"]

    def test_no_comment_lines(self, tmp_path):
        f = self._write_file(
            tmp_path / "data.dat",
            "0.1\t0.5\t1.2\n0.2\t0.6\t1.3\n",
        )
        data = Data(path=tmp_path)
        header = data._discover_file_header(f)
        assert header is None

    def test_only_comment_lines(self, tmp_path):
        f = self._write_file(
            tmp_path / "data.dat",
            "# Just comments\n# Nothing else\n",
        )
        data = Data(path=tmp_path)
        header = data._discover_file_header(f)
        assert header is None

    def test_custom_comment_prefix(self, tmp_path):
        f = self._write_file(
            tmp_path / "data.dat",
            "% Time\tcpu\n0.1\t0.5\n",
        )
        data = Data(path=tmp_path)
        header = data._discover_file_header(f, comment="%")
        assert header == ["Time", "cpu"]

    def test_custom_delimiter(self, tmp_path):
        f = self._write_file(
            tmp_path / "data.dat",
            "# Time,cpu,clock\n0.1,0.5,1.2\n",
        )
        data = Data(path=tmp_path)
        header = data._discover_file_header(f, delim=",")
        assert header == ["Time", "cpu", "clock"]


class TestFirstTimeDirectory:
    def test_empty_directory_raises(self, tmp_path):
        fo_dir = tmp_path / "postProcessing" / "myFO"
        fo_dir.mkdir(parents=True)
        data = Data(path=tmp_path)
        with pytest.raises(IndexError):
            data._first_time_directory(fo_dir)

    def test_non_numeric_subdir_raises(self, tmp_path):
        fo_dir = tmp_path / "postProcessing" / "myFO"
        fo_dir.mkdir(parents=True)
        (fo_dir / "latest").mkdir()
        data = Data(path=tmp_path)
        with pytest.raises(ValueError):
            data._first_time_directory(fo_dir)

    def test_numeric_dirs_sorted(self, tmp_path):
        fo_dir = tmp_path / "postProcessing" / "myFO"
        fo_dir.mkdir(parents=True)
        (fo_dir / "100").mkdir()
        (fo_dir / "0").mkdir()
        (fo_dir / "50").mkdir()
        data = Data(path=tmp_path)
        assert data._first_time_directory(fo_dir) == "0"


class TestDataInit:
    def test_invalid_backend_raises(self, tmp_path):
        with pytest.raises(ValueError):
            Data(path=tmp_path, dataframe_format="arrow")

    def test_polars_default(self, tmp_path):
        data = Data(path=tmp_path)
        from flowboost.openfoam.data import Backend
        assert data.dataframe_format == Backend.POLARS

    def test_pandas_backend(self, tmp_path):
        data = Data(path=tmp_path, dataframe_format="pandas")
        from flowboost.openfoam.data import Backend
        assert data.dataframe_format == Backend.PANDAS


class TestLoadData:
    def test_single_file_polars(self, tmp_path):
        f = tmp_path / "data.dat"
        f.write_text("# Time\tvalue\n0.1\t1.0\n0.2\t2.0\n")
        data = Data(path=tmp_path)
        df = data.load_data(f)
        assert len(df) == 2
        assert "Time" in df.columns

    def test_single_file_pandas(self, tmp_path):
        f = tmp_path / "data.dat"
        f.write_text("# Time\tvalue\n0.1\t1.0\n0.2\t2.0\n")
        data = Data(path=tmp_path, dataframe_format="pandas")
        df = data.load_data(f)
        assert len(df) == 2
        assert "Time" in df.columns

    def test_backend_override(self, tmp_path):
        """Polars-configured Data can return pandas via override."""
        import pandas as pd

        f = tmp_path / "data.dat"
        f.write_text("# Time\tvalue\n0.1\t1.0\n")
        data = Data(path=tmp_path, dataframe_format="polars")
        df = data.load_data(f, backend="pandas")
        assert isinstance(df, pd.DataFrame)

    def test_multi_file_concat(self, tmp_path):
        f1 = tmp_path / "a.dat"
        f2 = tmp_path / "b.dat"
        f1.write_text("# Time\tvalue\n0.1\t1.0\n")
        f2.write_text("# Time\tvalue\n0.2\t2.0\n")
        data = Data(path=tmp_path)
        df = data.load_data([f1, f2])
        assert len(df) == 2

    def test_custom_separator_produces_same_schema_for_pandas_and_polars(self, tmp_path):
        f = tmp_path / "comma.dat"
        f.write_text("# Time,value,clock\n0.1,1.0,2.0\n0.2,3.0,4.0\n")

        polars_df = Data(path=tmp_path, dataframe_format="polars").load_data(
            f, separator=","
        )
        pandas_df = Data(path=tmp_path, dataframe_format="pandas").load_data(
            f, separator=","
        )

        assert polars_df.columns == ["Time", "value", "clock"]
        assert list(pandas_df.columns) == ["Time", "value", "clock"]
        assert polars_df.to_dicts() == pandas_df.to_dict(orient="records")
