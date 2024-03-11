import logging
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
import polars as pl


class Backend(Enum):
    PANDAS = "pandas"
    POLARS = "polars"


class Data:
    """Access and manipulation of OpenFOAM case data.

    Data can be accessed as Pandas or Polars dataframes:
    - https://pandas.pydata.org/
    - https://pola.rs/

    The underlying data loading and processing is always done through Polars.
    Changing the backend thus only affects the format dataframes are returned
    in.

    For dictionary access, use dictionary.Dictionary.
    """

    def __init__(
        self,
        path: Path,
        dataframe_format: Literal["pandas", "polars"] = "polars",
        low_memory: bool = False,
        lazy_backend: bool = False,
    ) -> None:
        """_summary_

        Args:
            path (Path): Path of the case directory.
            dataframe_format (Literal['pandas', 'polars'], \
                optional): Dataframe backend to utilize. Defaults to Backend.POLARS.
            low_memory (bool, optional): Enable low-memory mode, at the \
                expense of performance. Defaults to False.
            lazy_backend (bool, optional): Use the lazy, streaming Polars \
                backend to optimize performance. Only applies when Polars \
                     is the configured backend. Defaults to False.

        Raises:
            NotImplementedError: _description_
        """
        # Path of case directory
        self.path: Path = Path(path)
        self.post_processing_path: Path = self.path / "postProcessing"

        # Dataframe format and memory optimizations
        self.dataframe_format: Backend = Backend(dataframe_format)
        self.low_memory: bool = low_memory
        self.lazy_backend: bool = lazy_backend

        if self.dataframe_format not in Backend:
            raise NotImplementedError(
                f"Format '{dataframe_format}' not supported: should be in {list(Backend)}"
            )

    def time_directories(self):
        # Return all time directories
        # TODO do through case; this serves no purpose here
        pass

    def postProcessing_directories(self) -> list[Path]:
        """ Load a list of all sub-directories within the postProcessing directory.

        Returns:
            Optional[list[Path]]: List of Paths, None only if postProcessing \
                does not exist.
        """
        if not Path(self.path, "postProcessing").exists():
            logging.error(f"No postProcessing dir found for {self.path}")
            return []

        return [f for f in Path(self.path, "postProcessing").iterdir() if f.is_dir()]

    def postProcessing_directory_names(self) -> list[str]:
        """ Load a list of all sub-directories within the postProcessing directory.

        Returns:
            Optional[list[Path]]: List of Paths, None only if postProcessing \
                does not exist.
        """
        return [d.name for d in self.postProcessing_directories()]

    def discover_function_objects(self) -> dict[str, dict[str, list[Path]]]:
        """Discovers all function object directories within postProcessing,
        returning a mapping for function object names to the time directories
        and output files within them.

        More specifically, the output is of the form:

        ```python
        {
            "my function object 1": {
                "time0": [file1_path, file2_path],
                "time1": [file1_path, file2_path]
            }
        }
        ```

        Returns:
            dict[str, dict[str, list[Path]]]: Mapping of function object names to their contents
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
                time_dir.name: list(time_dir.glob("*.*")) for time_dir in time_dirs
            }

        return function_objects

    def _time_dirs_for_function_object(self, fo_folder: Path) -> list[Path]:
        return list(filter(Path.is_dir, fo_folder.iterdir()))

    def simple_function_object_reader(
        self, function_object_name: str, at_time: Optional[str] = None
    ) -> Optional[pl.DataFrame]:
        """ Loads data for a function object in the simplest case, where the
        output only contains one time directory and one output file. If the
        time instance is specified, the reader works for more complicated,
        multi-time output directories (but fails if it is not specified).

        If the file is not found for other reasons, such as due to a missing
        postProcessing directory, a None is returned.

        If these conditions are not met, the reader raises an error due to the
        potential uncertainty about the user's intentions. For more nuanced
        data access:
            1) Discover available files and directories using \
                Data.discover_function_objects().

            2) Load data using Data.load_data(), which additionally offers \
                dataframe concatenation and arbitrary data loading.

        Args:
            function_object_name (str): Name of function object, and the \
                accompanying sub-directory within postProcessing.
            at_time (str, optional): Required if a directory with multiple \
                time sub-directories is provided.

        Returns:
            Any: A loaded Dataframe
        """
        function_objects = self.discover_function_objects()

        if function_object_name not in function_objects:
            logging.error(
                f"Function object '{function_object_name}' not found in {self.post_processing_path}"
            )
            return None

        # Time dirs for this FO
        times = function_objects[function_object_name]

        if at_time:
            if at_time not in times:
                logging.error(
                    f"Time '{at_time}' not found for function object '{function_object_name}'."
                )
                return None
            time_dir = at_time
        else:
            if len(times) != 1:
                raise ValueError(
                    f"Multiple time directories found for '{function_object_name}'. Specify `at_time`."
                )

            time_dir = next(iter(times))

        # Files at this time instance
        files = times[time_dir]

        if len(files) != 1:
            raise ValueError(
                f"Expected one file, found {len(files)} in '{function_object_name}/{time_dir}': use data.load_data()"
            )

        return self.load_data(files[0])

    def load_data(
        self, files: Union[Path, list[Path]], comment: str = "#", separator: str = "\t"
    ) -> pl.DataFrame:
        """
        Loads data from a single file or a list of files. If a list is
        provided, it loads each file into a DataFrame and returns a
        concatenated DataFrame of all files.

        Args:
            files (Path, list[Path]): A path to a single file or a list of \
                file paths.
            comment (string, optional): Character indicating a comment in the \
                function object data file. Defaults to '#'.
            separator (string, optional): Character used as column separator \
                in the function object data file. Defaults to '\\t'.

        Returns:
            A DataFrame containing the data from the file(s). Type depends on \
                Data.dataframe_format (Pandas or Polars).
        """
        if isinstance(files, Path):
            return self._read_fo_to_dataframe(
                files, comment=comment, separator=separator
            )

        dfs = [
            self._read_fo_to_dataframe(file, comment=comment, separator=separator)
            for file in files
        ]

        match self.dataframe_format:
            case Backend.PANDAS:
                return pd.concat(dfs)
            case Backend.POLARS:
                return pl.concat(dfs)

    def fields(self):
        pass

    def read_field(self):
        pass

    def set_backend(self, backend: Literal["pandas", "polars"]):
        if backend not in Backend:
            raise ValueError(f"'{backend}' not supported ({list(Backend)})")

        self.dataframe_format = Backend(backend)

    def _first_time_directory(self, path: Path) -> Optional[str]:
        """Post-processing directory paths are keyed by the first time instance;
        use that preferentially.

        Returns:
            Optional[str]: The first time directory
        """
        all_times = [t.name for t in list(filter(Path.is_dir, path.iterdir()))]
        all_times.sort(key=float)

        return all_times[0]

    def _discover_file_header(
        self, file: Path, comment: str = "#", delim="\t"
    ) -> Optional[list[str]]:
        # Open the file and scan for the header row
        prev_line = None
        with open(file, "r") as f:
            for i, line in enumerate(f):
                if not line.startswith(comment):
                    if line and prev_line:
                        return prev_line.strip(comment).split()
                    else:
                        return None

                prev_line = line

        return None

    def _discover_file_header_index(self, file: Path, comment: str = "#"):
        header_row_index = None

        # Open the file and scan for the header row
        with open(file, "r") as f:
            for i, line in enumerate(f):
                if not line.startswith(comment):
                    header_row_index = i
                    break

        return header_row_index

    def _read_fo_to_dataframe(
        self, file: Path, comment="#", separator="\t"
    ) -> pl.DataFrame:
        """Reads a function object output file to a dataframe according to
        the specified backend. Should not be used for fields, or files where
        the header is not commented out using the `comment` delimiter.

        Args:
            file (Path): Path to file
            comment (str, optional): Comment (and header) prefix. Defaults to "#".
            separator (str, optional): Column separator in a CSV-like file.
        """

        def read_pandas(file) -> pd.DataFrame:
            header = self._discover_file_header_index(file)
            return pd.read_csv(
                file, header=header, comment=comment, low_memory=self.low_memory
            )

        def read_polars(file) -> pl.DataFrame:
            cols = self._discover_file_header(file, comment=comment)
            return pl.scan_csv(
                file,
                comment_prefix=comment,  # OF standard comment prefix
                has_header=False,  # Header is interpreted as a comment!
                separator=separator,
                new_columns=cols,
                low_memory=self.low_memory,
            ).collect(streaming=self.lazy_backend)

        match self.dataframe_format:
            case Backend.PANDAS:
                return read_pandas(file)
            case Backend.POLARS:
                return read_polars(file)
            case _:
                raise NotImplementedError(
                    f"Backend '{self.dataframe_format}' not valid"
                )
