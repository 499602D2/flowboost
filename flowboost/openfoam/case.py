from __future__ import annotations  # pre-3.11 compatibility

import logging
from datetime import datetime, timezone
from enum import Enum
from hashlib import blake2b
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from uuid import uuid4

import tomlkit

from flowboost.openfoam.data import Data
from flowboost.openfoam.dictionary import DictionaryLink, DictionaryReader, Entry
from flowboost.openfoam.interface import FOAM, run_command
from flowboost.optimizer.search_space import Dimension

if TYPE_CHECKING:
    # Lazy imports for function argument typing
    from flowboost.optimizer.objectives import AggregateObjective, Objective

DEFAULT_METADATA: str = "metadata.toml"


class Status(Enum):
    NOT_SUBMITTED = "not_submitted"
    SUBMITTED = "submitted"
    FINISHED = "finished"


class Case:
    def __init__(self, path: Path | str) -> None:
        """Create a new abstraction for an OpenFOAM case.

        When creating a new Case from scratch, the initialization must be
        performed using `Case.clone()`, `Case.from_tutorial()`, or manually.

        Args:
            path (Path | str): Path to an existing case directory.

        Raises:
            FileNotFoundError: If provided path is not found.
        """
        # Identifiers
        self.path: Path = Path(path).resolve().absolute()
        self.name: str = self.path.name
        self.id: str = unique_id(self.path)

        # Case evaluation status. Success is determined by user-specified
        # criteria.
        self.status: Status = Status.NOT_SUBMITTED
        self.success: Optional[bool] = None

        # Data access for a case (through case.data property)
        self._data: Data = Data(path=self.path)

        # Additional attributes that can be configured: not required, but
        # helpful for posterity.
        self._based_on_case: Optional[Path] = None
        self._created_at: datetime = datetime.now(tz=timezone.utc)

        # For model-generated cases, optional attributes.
        # Generation index is of the form "001.01", where the leading integer
        # is the generation, and trailing is index within a generation.
        self._generation_index: Optional[str] = None
        self._model_predictions_by_objective: Optional[dict] = None

        # Execution environment information and submission time
        self._execution_environment: Optional[str] = None
        self._submitted_at: Optional[datetime] = None

        if not self.path.exists():
            raise FileNotFoundError(f"Directory does not exist [{str(self)}]")

    @property
    def data(self) -> Data:
        if self._data.path != self.path:
            self._data = Data(path=self.path)
        return self._data

    def mark_failed(self):
        self.success = False
        self.persist_to_file()

    def clone(
        self,
        clone_to: Path | str,
        add: Optional[list[str]] = None,
        latest_time: bool = False,
        no_scripts: bool = False,
        processor: bool = False,
        start_from: Optional[str] = None,
    ) -> "Case":
        """ Clone this case to a new case directory. The `clone_to` directory
        should not exist: otherwise, the cloning fails.

        For additional information, see the OpenFOAM documentation for `foamCloneCase`.

        Args:
            clone_to (Path | str): Destination case directory. Will be created.
            add (Optional[list[str]]): Copy 1 or more additional files/directories \
                from source case
            latest_time (bool, optional): Clone the latest time directory. Defaults to \
                False.
            no_scripts (bool, optional): Do not copy shell scripts. Defaults to False.
            processor (bool, optional): Copy *processor dirs of a decomposed case. \
                Defaults to False.
            start_from (Optional[str], optional): Set the starting time directory name.\
                Defaults to None.

        Returns:
            Case: Object representing the new case
        """
        assert FOAM.in_env()

        new_case_path = Path(clone_to)
        if new_case_path.exists():
            raise FileExistsError(f"Case directory already exists: '{new_case_path}'")

        cmd = ["foamCloneCase", self.path, new_case_path]

        # Handle optional arguments
        if add:
            cmd.extend(["-add"] + add)
        if latest_time:
            cmd.append("-latestTime")
        if no_scripts:
            cmd.append("-no-scripts")
        if processor:
            cmd.append("-processor")
        if start_from:
            cmd.extend(["-startFrom", start_from])

        run_command(cmd)

        new = Case(path=new_case_path)
        new._based_on_case = self.path
        return new

    def foam_get(self, file: str, target: str = "system"):
        """Run foamGet in the case directory, adding the requested file from
        etc/caseDicts to target directory (defaulting to system).

        For additional details, see OpenFOAM's documentation for foamGet.

        Args:
            file (str): File to add.
        """
        cmd = ["foamGet", "-case", self.path, "-target", target, file]
        run_command(cmd)
        print(f"Loaded '{file}' into {self.path.name}/{target}")

    @staticmethod
    def from_tutorial(tutorial: str, new_case_dir: Path | str) -> "Case":
        """
        Creates a new Case directory as `clone_to`, from an existing
        OpenFOAM tutorial. If new_case_dir exists, the clone will be skipped.

        For additional information, see the OpenFOAM documentation for `foamCloneCase`.

        Args:
            tutorial (str): Tutorial path, relative to /tutorials folder.
                Example: "multicomponentFluid/aachenBomb".
            new_case_dir (Path | str): Path for the new case directory that \
                will be created. E.g. `my/path/aachenBomb_tutorial`.
        """
        if not FOAM.in_env():
            raise ValueError("OpenFOAM not sourced")

        tutorial_case = FOAM.tutorials() / tutorial

        if not tutorial_case.exists():
            raise FileNotFoundError(
                f"Tutorial case path does not exist: '{tutorial_case}'"
            )

        new_case_dir = Path(new_case_dir)
        if new_case_dir.exists():
            logging.warning(
                f"Directory already exists: returning existing Case [{new_case_dir}]"
            )
            return Case(path=new_case_dir)

        cmd = ["foamCloneCase", tutorial_case, new_case_dir]
        run_command(cmd)

        new_case = Case(path=new_case_dir)
        new_case._based_on_case = tutorial_case
        return new_case

    def clean(self, ask_for_confirmation: bool = True):
        """Cleans the case directory by executing `foamCleanCase`.

        Deletes:
        - all files generated during the workflow
        - postProcessing
        - VTK
        - constant/polyMesh directory
        - processor directories from parallel decomposition
        - dynamicCode for run-time compiled code.

        For additional information, see the OpenFOAM documentation for `foamCleanCase`.

        Args:
            ask_for_confirmation (bool): Set to false to override confirmation dialog
        """
        if ask_for_confirmation:
            print(f"Preparing to run foamCleanCase in cwd='{self.path}'")
            inp = input(f"OK to run foamCleanCase for '{self.name}' (y/N): ")
            if inp.lower() != "y":
                print("Aborting")
                return

        run_command(["foamCleanCase"], cwd=self.path)
        print(f"Case directory cleaned: '{self.path}'")

    def list_time_directories(self, omit_dirs: list[str] = ["0"]) -> list[str]:
        """Return a list of all time directories in this case. Does not
        include "0"-folder by default.

        For additional information, see the OpenFOAM documentation for
        `listTimes`.

        Args:
            omit_dirs[list[str]]: List of time directories to omit from output.

        Returns:
            list[str]: List of strings of time directory names (e.g. ["5"]).
        """
        out = run_command(["listTimes"], cwd=self.path)
        times = out.strip().split("\n")

        # FoamListTimes is wonky in the sense that it omits 0 by default, but
        # our case may have a non-0 first time dir.
        if "0" not in omit_dirs and Path(self.path, "0").exists():
            times = ["0"] + times

        if omit_dirs:
            # Not a set intersection to preserve order
            times = [t for t in times if t not in omit_dirs]

        return times

    def remove_time_directories(self):
        """Removes all time directories, except for "0".

        For additional information, see the OpenFOAM documentation for `listTimes -rm`.
        """
        # Get existing dirs
        dirs = self.list_time_directories()

        print("Removing time directories...")
        run_command(["listTimes", "-rm"], cwd=self.path)

        print(f"Removed time directories: {', '.join(dirs)}")

    def dictionary(
        self, read_from: str | DictionaryLink
    ) -> Union[DictionaryReader, Entry]:
        """ Access a FOAM dictionary file for this case using either a DictionaryLink or
        a dictionary path relative to the case directory.

        For additional information, see:
        - flowboost/openfoam/dictionary
        - OpenFOAM documentation for `foamDictionary`

        Args:
            read_from (str | DictionaryLink): A /-separated path to a dictionary file \
                relative to the case directory's root, or a DictionaryLink.

        Returns:
            DictionaryReader: A reader operating on the dictionary.
        """
        if isinstance(read_from, str):
            return DictionaryReader(self.path / read_from)

        return read_from.reader(self.path)

        raise ValueError("No DictionaryLink or path provided, cannot produce a reader")

    def parametrize_configuration(self, dimensions: list[Dimension]) -> dict[str, Any]:
        """
        Parametrize a case configuration for the given list of dimensions. More
        specifically, the function produces a dictionary mapping the name of
        each dimension to the value of the linked entry in the case's config
        dictionary.

        Args:
            dimensions (list[Dimension]): List of dimensions to produce \
                parametrization for

        Raises:
            ValueError: On failure of dictionary reading

        Returns:
            dict[str, Any]: Parametrized configuration keyed by dimension names
        """
        par_dict = {}
        for dim in dimensions:
            if dim.linked_entry is None:
                raise ValueError(
                    f"Cannot parametrize case: '{dim.name}' has no linked dict entry"
                )

            reader = dim.linked_entry.reader(self.path)

            if not reader or isinstance(reader, DictionaryReader):
                raise ValueError(
                    f"Cannot parametrize case: no value for {reader} [{self}]"
                )

            par_dict[dim.name] = reader.value

        return par_dict

    def objective_function_outputs(
        self, objectives: list[Union["Objective", "AggregateObjective"]]
    ) -> dict[str, Any]:
        """Get the post-processed objective function outputs for this case.

        Args:
            objectives (list[&#39;Objective&#39;, &#39;AggregateObjective&#39;])

        Returns:
            dict[str, Any]: Mapping of objective name to output value
        """
        output_mapping = {}

        for obj in objectives:
            out = obj.data_for_case(self, post_processed=True)
            if out is None:
                raise ValueError(f"Objective='{obj.name}' output None for case {self}")

            output_mapping[obj.name] = out

        return output_mapping

    def submission_script(self, glob_with: str = "Allrun*") -> Optional[Path]:
        """Finds an Allrun* -named submission script in the case directory.

        Returns:
            Optional[Path]: Absolute Path to Allrun* script. None if not found.
        """
        script_path = next(self.path.glob(glob_with), None)
        if script_path:
            return Path(script_path).absolute()

        return None

    def serialize_to_parquet(self):
        """Serialize this case and all its contents to a highly portable
        data format.

        TODO:
        - Polars supports direct de/serialization of Parquet data!
            - Easy to do for function objects
            - What about fields?
        """
        raise NotImplementedError("Parquet serialization to implemented")

    def state(self) -> dict:
        """Return a dictionary representing the key attributes and properties
        of this case, useful for further processing into e.g. TOML/JSON.

        Returns:
            dict: State of the object's properties
        """
        state = {
            "name": self.name,
            "id": self.id,
            "path": str(self.path),
            # Submission and failure criteria
            "status": self.status.value,
            "success": self.success,
            # Optional properties
            "created_at": self._created_at.isoformat(),
            "submitted_at": self._submitted_at,
            "generation_index": self._generation_index,
            "based_on_case": str(self._based_on_case) if self._based_on_case else None,
            "model_predictions_by_objective": self._model_predictions_by_objective,
            "execution_environment": self._execution_environment,
        }

        # Remove nones
        return {k: v for k, v in state.items() if v is not None}

    def persist_to_file(self, fname: str = DEFAULT_METADATA):
        """Generate a TOML-based report of this case in the case directory.

        Returns:
            str: _description_
        """
        file_path = Path(self.path, fname)

        if file_path.exists():
            # Check if the metadata file exists and read existing data
            with open(file_path, "r") as toml_file:
                data = tomlkit.load(toml_file)
        else:
            data = tomlkit.document()

        # Get the current state
        new_state = self.state()

        # Update the existing data with the new state
        # This ensures we only update values for existing keys and add new keys
        # without removing anything
        for key, value in new_state.items():
            data[key] = value

        # Write the updated data back to the file
        with open(file_path, "w") as toml_file:
            tomlkit.dump(data, toml_file)

    def update_metadata(
        self,
        update_entries: dict,
        entry_header: Optional[str] = None,
        fname: str = DEFAULT_METADATA,
    ):
        """
        Updates the case metadata file with new entries.

        Args:
            update_entries (dict): A json-like dictionary of entries to update.
            fname (str, optional): File to store in. Defaults to DEFAULT_METADATA.
        """
        file_path = Path(self.path, fname)

        # Ensure the directory exists, to avoid FileNotFoundError
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            # Read the existing data from the file
            with open(file_path, "r") as toml_file:
                data = tomlkit.load(toml_file)
        else:
            # If the file does not exist, initialize an empty TOML document
            data = tomlkit.document()

        if entry_header:
            if entry_header not in data:
                # Create new table
                table = tomlkit.table()
                for key, value in update_entries.items():
                    table.add(key, value)
            else:
                # Access existing table
                table = data.pop(entry_header)
                for key, value in update_entries.items():
                    table[key] = value

            # Write table back to doc
            data[entry_header] = table
        else:
            # No table header provided: do direct write
            for key, value in update_entries.items():
                data[key] = value

        # Write the updated data back to the file
        with open(file_path, "w") as toml_file:
            tomlkit.dump(data, toml_file)

    def read_metadata(
        self, from_file: str = DEFAULT_METADATA
    ) -> Optional[tomlkit.TOMLDocument]:
        file_path = Path(self.path, from_file)
        if not file_path.exists():
            return None

        # Read the existing data from the file
        with open(file_path, "r") as toml_file:
            return tomlkit.load(toml_file)

    @classmethod
    def restore_from_file(
        cls, case_directory: Path | str, fname: str = DEFAULT_METADATA
    ) -> "Case":
        file = Path(case_directory, fname)

        if not file.exists():
            raise FileNotFoundError(f"Case info file not found [{file}]")

        with open(file, mode="r") as toml_file:
            data = tomlkit.load(toml_file)

        # Main properties
        case = cls(path=str(data["path"]))
        case.id = str(data["id"])

        # Status properties
        case.status = Status(data.get("status", "not_submitted"))
        case.success = data.get("success", None)

        # Additional properties that may or may not exist
        case._based_on_case = data.get("based_on_case")
        case._created_at = datetime.fromisoformat(data.get("created_at", ""))
        case._generation_index = data.get("generation_index")
        case._model_predictions_by_objective = data.get(
            "model_predictions_by_objective"
        )
        case._execution_environment = data.get("execution_environment")
        case._submitted_at = data.get("submitted_at")

        return case

    @classmethod
    def try_restoring(cls, case_dir: Path | str, fname: str = DEFAULT_METADATA) -> Case:
        """
        Tries restoring a Case from a directory. If no persistence file is
        found, the Case object is re-initialized.

        Args:
            case_dir (Path | str): Path to try restoring from
            fname (str, optional): File to restore from. Defaults to DEFAULT_METADATA.

        Returns:
            Case: Restored or new case
        """
        if Path(case_dir, fname).exists():
            return cls.restore_from_file(case_dir, fname)

        return cls(case_dir)

    def _delete_all_data(self, skip_familiarity_checks: bool = False):
        """ Permanently deletes a case directory and all its contents.

        By default, the function verifies that the directory contains the
        constant and system folders.

        Args:
            skip_familiarity_checks (bool, optional): Skip verifying that the folder \
                is an OpenFOAM case directory
        """
        # Verify that the directory looks like an OpenFOAM case directory
        if path_is_foam_dir(self.path) or skip_familiarity_checks:
            logging.info(f"Deleting case directory: {str(self)}")
            run_command(command=["rm", "-rf", self.path])
            return True

        logging.error(
            f"Constant and system folders not found in {self.path}: not deleting"
        )
        return False

    def post_evaluation_update(self, serialized_job: dict):
        # Mark as finished
        self.status = Status("finished")

        # TODO Infer success
        logging.warning(
            "Case finished, but not running user-defined success status functions (TODO)"
        )

        # Update metadata on disk
        self.persist_to_file()

        # Stash the job information in metadata
        self.update_metadata(
            update_entries=serialized_job, entry_header="evaluation-information"
        )

    def __str__(self):
        return f"OpenFOAM-Case: '{self.name}' (id={self.id}) [{self.path}]"


def path_is_foam_dir(path: Path | str) -> bool:
    """
    A path is a FOAM case directory if it contains the constant and system
    sub-directories.

    Args:
        path (Path | str): Path to test

    Returns:
        bool: True if path is FOAM directory
    """
    return Path(path, "constant").is_dir() and Path(path, "system").is_dir()


def unique_id(hashable: Any = None) -> str:
    """
    Generate a unique ID for a case. On default Case init, the path is hashed.

    Args:
        hashable (Any, optional): A hashable. Defaults to None.

    Returns:
        str: UID
    """
    if not hashable:
        hashable = uuid4()
    return blake2b(str(hashable).encode()).hexdigest()[0:8]
