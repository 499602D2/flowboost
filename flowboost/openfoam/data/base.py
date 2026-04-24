import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, Optional, TypeVar, Union, overload

import pandas as pd
import polars as pl

FrameT = TypeVar("FrameT", pd.DataFrame, pl.DataFrame)


class Data(ABC, Generic[FrameT]):
    """ABC for OpenFOAM postProcessing data access.

    Subclasses implement ``_read_file`` and ``_concat`` for their DataFrame
    backend.  Built-in implementations: :class:`PolarsData`, :class:`PandasData`.

    For dictionary access, use :mod:`flowboost.openfoam.dictionary`.
    """

    def __init__(self, path: Path, *, low_memory: bool = False) -> None:
        """
        Args:
            path: Path to the OpenFOAM case directory.
            low_memory: Trade performance for lower memory usage.
        """
        self.path: Path = Path(path)
        self.post_processing_path: Path = self.path / "postProcessing"
        self.low_memory: bool = low_memory

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.path}')"

    # ------------------------------------------------------------------
    # Abstract interface — backends implement these two methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _read_file(self, file: Path, comment: str, separator: str) -> FrameT:
        """Read a single postProcessing output file into a DataFrame."""
        ...

    @abstractmethod
    def _concat(self, frames: list[FrameT]) -> FrameT:
        """Concatenate multiple DataFrames into one."""
        ...

    # ------------------------------------------------------------------
    # Concrete shared logic
    # ------------------------------------------------------------------

    def _for_backend(self, backend: str) -> "Data":
        """Return a sibling instance for the named backend.

        Returns *self* when it already matches the requested backend.
        """
        from . import _BACKENDS

        cls = _BACKENDS.get(backend)
        if cls is None:
            raise ValueError(
                f"Unknown backend '{backend}', expected: {list(_BACKENDS)}"
            )
        if isinstance(self, cls):
            return self
        return cls(path=self.path, low_memory=self.low_memory)

    def postProcessing_directories(self) -> list[Path]:
        """Load all sub-directories within the postProcessing directory."""
        if not self.post_processing_path.exists():
            logging.error(f"No postProcessing dir found for {self.path}")
            return []

        return [f for f in self.post_processing_path.iterdir() if f.is_dir()]

    def postProcessing_directory_names(self) -> list[str]:
        """Load all sub-directory names within the postProcessing directory."""
        return [d.name for d in self.postProcessing_directories()]

    def discover_function_objects(self) -> dict[str, dict[str, list[Path]]]:
        """Discover all function object directories within postProcessing.

        Returns a mapping of the form::

            {
                "functionObjectName": {
                    "0": [file1_path, file2_path],
                    "100": [file1_path],
                }
            }
        """
        if not self.post_processing_path.exists():
            logging.error(
                f"No postProcessing directory found at {self.post_processing_path}"
            )
            return {}

        function_objects = {}
        for function_dir in filter(Path.is_dir, self.post_processing_path.iterdir()):
            time_dirs = self._time_dirs_for_function_object(function_dir)

            function_objects[function_dir.name] = {
                time_dir.name: sorted(
                    (path for path in time_dir.iterdir() if path.is_file()),
                    key=lambda path: path.name,
                )
                for time_dir in time_dirs
            }

        return function_objects

    def _time_dirs_for_function_object(self, fo_folder: Path) -> list[Path]:
        return list(filter(Path.is_dir, fo_folder.iterdir()))

    # -- simple_function_object_reader ------------------------------------

    @overload
    def simple_function_object_reader(
        self,
        function_object_name: str,
        at_time: Optional[str] = ...,
        backend: None = ...,
    ) -> Optional[FrameT]: ...

    @overload
    def simple_function_object_reader(
        self,
        function_object_name: str,
        at_time: Optional[str] = ...,
        *,
        backend: Literal["pandas", "polars"],
    ) -> Optional[Union[pd.DataFrame, pl.DataFrame]]: ...

    def simple_function_object_reader(
        self,
        function_object_name: str,
        at_time: Optional[str] = None,
        backend: Optional[Literal["pandas", "polars"]] = None,
    ) -> Optional[Union[FrameT, pd.DataFrame, pl.DataFrame]]:
        """Load data for a function object in the simplest case.

        Works when the output contains exactly one time directory and one
        output file.  If ``at_time`` is given, the reader targets that
        specific time directory instead.

        Pass ``backend`` to override the instance backend for this call
        (e.g. ``backend="pandas"`` on a Polars-backed reader).

        Returns ``None`` when the function object or time directory is not
        found.  Raises :class:`ValueError` when the output structure is
        ambiguous (multiple time dirs without ``at_time``, or multiple files).

        For more nuanced access:
            1. Discover layout with :meth:`discover_function_objects`.
            2. Load specific files with :meth:`load_data`.
        """
        if backend is not None:
            return self._for_backend(backend).simple_function_object_reader(
                function_object_name,
                at_time=at_time,
            )

        function_objects = self.discover_function_objects()

        if function_object_name not in function_objects:
            logging.error(
                f"Function object '{function_object_name}' not found "
                f"in {self.post_processing_path}"
            )
            return None

        times = function_objects[function_object_name]

        if at_time:
            if at_time not in times:
                logging.error(
                    f"Time '{at_time}' not found for function object "
                    f"'{function_object_name}'."
                )
                return None
            time_dir = at_time
        else:
            if len(times) != 1:
                raise ValueError(
                    f"Multiple time directories found for "
                    f"'{function_object_name}'. Specify `at_time`."
                )
            time_dir = next(iter(times))

        files = times[time_dir]

        if len(files) != 1:
            raise ValueError(
                f"Expected one file, found {len(files)} in "
                f"'{function_object_name}/{time_dir}': use data.load_data()"
            )

        return self.load_data(files[0])

    # -- load_data --------------------------------------------------------

    @overload
    def load_data(
        self,
        files: Union[Path, list[Path]],
        comment: str = ...,
        separator: str = ...,
        backend: None = ...,
    ) -> FrameT: ...

    @overload
    def load_data(
        self,
        files: Union[Path, list[Path]],
        comment: str = ...,
        separator: str = ...,
        *,
        backend: Literal["pandas", "polars"],
    ) -> Union[pd.DataFrame, pl.DataFrame]: ...

    def load_data(
        self,
        files: Union[Path, list[Path]],
        comment: str = "#",
        separator: str = "\t",
        backend: Optional[Literal["pandas", "polars"]] = None,
    ) -> Union[FrameT, pd.DataFrame, pl.DataFrame]:
        """Load data from one or more files.

        If *files* is a list, each file is read into a DataFrame and the
        results are concatenated.

        Pass ``backend`` to override the instance backend for this call.
        """
        if backend is not None:
            return self._for_backend(backend).load_data(
                files,
                comment=comment,
                separator=separator,
            )

        if isinstance(files, Path):
            return self._read_file(files, comment, separator)

        return self._concat([self._read_file(f, comment, separator) for f in files])

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _first_time_directory(self, path: Path) -> str:
        """Return the earliest (by float value) time directory name."""
        all_times = [t.name for t in filter(Path.is_dir, path.iterdir())]
        all_times.sort(key=float)
        return all_times[0]

    def _discover_file_header(
        self, file: Path, comment: str = "#", delim: str = "\t"
    ) -> Optional[list[str]]:
        """Return column names from the last comment line before data."""
        prev_line = None
        with open(file, "r") as f:
            for line in f:
                if not line.startswith(comment):
                    if line and prev_line:
                        return [
                            col.strip() for col in prev_line.strip(comment).split(delim)
                        ]
                    return None
                prev_line = line
        return None

    def _default_column_names(
        self, file: Path, comment: str = "#", delim: str = "\t"
    ) -> Optional[list[str]]:
        with open(file, "r") as f:
            for line in f:
                if line.startswith(comment) or not line.strip():
                    continue

                columns = [col.strip() for col in line.rstrip("\n").split(delim)]
                return [f"column_{i}" for i in range(1, len(columns) + 1)]

        return None

    def _discover_columns(
        self, file: Path, comment: str = "#", delim: str = "\t"
    ) -> Optional[list[str]]:
        header = self._discover_file_header(file, comment=comment, delim=delim)
        if header is not None:
            return header

        return self._default_column_names(file, comment=comment, delim=delim)

    def _discover_file_header_index(
        self, file: Path, comment: str = "#"
    ) -> Optional[int]:
        with open(file, "r") as f:
            for i, line in enumerate(f):
                if not line.startswith(comment):
                    return i
        return None
