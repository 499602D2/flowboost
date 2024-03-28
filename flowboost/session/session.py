import importlib.resources as pkg_resources
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from flowboost.config import config
from flowboost.manager.manager import JobV2, Manager
from flowboost.openfoam.case import Case, path_is_foam_dir, unique_id
from flowboost.openfoam.dictionary import Dictionary, DictionaryLink, DictionaryReader
from flowboost.openfoam.types import FOAMType
from flowboost.optimizer.acquisition_offload import OFFLOAD_SCRIPT
from flowboost.optimizer.backend import DEFAULT_OFFLOAD_RESULT_FNAME
from flowboost.optimizer.interfaces.Ax import Backend
from flowboost.optimizer.search_space import Dimension


class Session:
    def __init__(
        self,
        name: str,
        data_dir: Path,
        archival_dir: Optional[Path] = None,
        dataframe_format: str = "polars",
        backend: str = "AxBackend",
    ):
        """
        Initialize an optimization session.

        Args:
            name (str): Name for this session: can be anything.
            data_dir (Path): Directory in which to prepare new simulations. \
                Will be created.
            archival_dir (Optional[Path], optional): Directory in which to store \
                finished simulations (e.g. an external drive). Defaults to None.
            dataframe_format (str, optional): Dataframe format to use by \
                default for case data access. Can be configured on a per-case \
                basis. Defaults to polars.
            backend (str, optional): Optimization backend to use. Defaults to "Ax".
        """
        self.name: str = name
        self.data_dir: Path = Path(data_dir)
        self.pending_dir: Path = Path(self.data_dir, "cases_pending")
        self.archival_dir: Path = Path(data_dir, "cases_completed")
        self.created_at: datetime = datetime.now(tz=timezone.utc)
        self.dataframe_format: str = dataframe_format

        if archival_dir:
            if archival_dir == self.data_dir:
                raise ValueError("Archival dir must not be same as session data dir")
            self.archival_dir = Path(archival_dir)

        # Create directories
        self._ensure_dirs()

        # Optimizer
        self.backend: Backend = Backend.create(backend)
        self.job_manager: Optional[Manager] = None

        # Template simulation that optimization points are derived from
        self._template_case: Optional[Case] = None
        self._template_case_add_files: Optional[list[str]] = []

        if Path(self.data_dir, config.DEFAULT_CONFIG_NAME).exists():
            # Check if we can restore instead
            logging.info(f"Restoring session ({self.data_dir})")
            self.restore()
        else:
            # New session: persist
            logging.info(f"Session created ({self.data_dir})")
            self.persist()

    def get_all_cases(self, include_failed: bool = True) -> list[Case]:
        """
        Get finished and pending cases in one list.

        Returns:
            list[Case]: List of finished + pending cases
        """
        return (
            self.get_finished_cases(include_failed=include_failed)
            + self.get_pending_cases()
        )

    def get_finished_cases(
        self, include_failed: bool = False, batch_process: bool = False
    ) -> list[Case]:
        """
        A simple property method for loading all finished case objects. This
        read is performed in `Session.archival_dir`.

        Args:
            include_failed (bool, optinoal): If failed cases should be \
                included. Note, that this does not apply to cases with an \
                unclear success status (case.success == None).
            batch_process (bool, optional): If cases should also be batch \
                processed before returning them. All cases, despite their \
                current `success` are always re-evaluated to accommodate for \
                changes in objective functions. Defaults to False.

        Returns:
            list[Case]: _description_
        """
        cases: list[Case] = []
        for p in filter(Path.is_dir, self.archival_dir.iterdir()):
            if path_is_foam_dir(p):
                cases.append(Case.try_restoring(p))

        if batch_process and cases:
            # Executes the post-processing steps while handling potential case
            # failures
            self.backend.batch_process(cases)

        if include_failed:
            return cases

        # Includes cases with unclear evaluation status
        return [c for c in cases if (c.success or c.success is None)]

    def get_failed_cases(self) -> list[Case]:
        """
        Gets all cases from the archival directory that have
        `Case.success == False`.

        Returns:
            list[Case]: _description_
        """
        cases: list[Case] = []
        for p in filter(Path.is_dir, self.archival_dir.iterdir()):
            if path_is_foam_dir(p):
                cases.append(Case.try_restoring(p))

        return [c for c in cases if c.success is False]

    def get_pending_cases(self) -> list[Case]:
        """
        A simple property method for loading all pending case objects. This
        read is performed in `Session.pending_dir`.

        Returns:
            list[Case]: Discovered pending cases
        """
        cases = []
        for p in filter(Path.is_dir, self.pending_dir.iterdir()):
            if path_is_foam_dir(p):
                cases.append(Case.try_restoring(p))

        return cases

    def start(self):
        """
        Start optimization job: pre-process, submit jobs.
        """
        # Template configured and dimensions OK, plus no wonky properties?
        self._verify_search_space_in_template()

        # Initialize the optimizer backend
        self.backend.initialize()

        # Check manager OK
        if not self.job_manager:
            logging.info("No job manager configured: only yielding new cases")

            limit_str = input("Enter number of cases to generate [int]: ")
            limit = FOAMType.try_parse_scalar(limit_str)
            if limit is None or limit <= 0:
                raise ValueError("Provided number is not valid")

            # Run optimizer loop once: data in, cases out
            new_cases = self.loop_optimizer_once(num_new_cases=int(limit))
            logging.info(f"{len(new_cases)} new case(s) prepared")
            self._pretty_print_cases(new_cases)
            return

        logging.info("Entering persistent optimization")
        self.persistent_optimization()

    def persistent_optimization(self):
        if not self.job_manager:
            raise ValueError("Cannot run persistent optimization without a job manager")

        while True:
            free_slots, finished, was_acq_job = self.job_manager.do_monitoring()
            logging.info(f"Manager returned slots={free_slots}, finished={finished}")

            if was_acq_job:
                self.handle_finished_acquisition_job(finished[0])
                continue

            for job in finished:
                logging.info(f"Moving data for finished job {job}")
                case_dest = Path(self.archival_dir, job.wdir.name)
                self.job_manager.move_data_for_job(job=job, dest=case_dest)
                Case(case_dest).post_evaluation_update(job.to_dict())

            logging.info("Entering optimizer loop")
            new_cases = self.loop_optimizer_once(num_new_cases=free_slots)

            for case in new_cases:
                # TODO pass script args automatically
                self.job_manager.submit_case(case)

    def loop_optimizer_once(self, num_new_cases: int) -> list[Case]:
        if self.backend.offload_acquisition:
            # If acquisition offload requested, initialize IPC JSON file
            self.offloaded_optimization(num_new_cases=num_new_cases)
            return []
        else:
            # Otherwise, do locally
            suggestion = self.local_optimization(num_new_cases=num_new_cases)

        if not suggestion:
            return []

        logging.info("Post-processing model suggestions")
        return self._process_optimizer_suggestion(suggestion)

    def local_optimization(self, num_new_cases: int) -> list[dict[Dimension, Any]]:
        if num_new_cases == 0:
            logging.error("Local optimization called with 0 num_new_cases")
            return []

        # Get all finished cases and batch process them while doing so
        finished_cases = self.get_finished_cases(batch_process=True)

        if finished_cases:
            # If any are finished, attach them
            logging.info("Running model update")
            self.backend.tell(self.get_finished_cases(batch_process=True))

        # If there are pending cases, attach them
        self.backend.attach_pending_cases(self.get_pending_cases())

        # Attach failed cases separately
        self.backend.attach_failed_cases(self.get_failed_cases())

        # Ready to get new cases
        logging.info(f"Running acquisition: manager had {num_new_cases} free slot(s)")
        suggestion = self.backend.ask(max_cases=num_new_cases)
        logging.debug(f"Got suggestions: {suggestion}")

        return suggestion

    def offloaded_optimization(self, num_new_cases: int):
        """
        Runs off-loaded model update and acquisition in an off-loaded manner,
        in an external computing environment. The function submits the
        acquisition job, the result of which will be processed later.

        Raises:
            ValueError: _description_
        """
        # Get all finished cases and batch process them while doing so
        finished_cases = self.get_finished_cases(batch_process=True)

        # Pending cases, so we don't generate them twice
        pending_cases = self.get_pending_cases()

        model_snapshot, data_snapshot = self.backend.prepare_for_acquisition_offload(
            finished_cases=finished_cases,
            pending_cases=pending_cases,
            save_in=self.data_dir,
        )

        logging.info("Generated model and data snapshots")

        # 0. Implement an acquisition-only interface
        #    - should accept a path for the output data (json)
        #    - jsons should tell what backend to use (some metadata)
        #    - jsons should include time at which they were created
        #    - anything else?
        # 1. SUBMIT acquisition job script with user's defined job manager
        #    args (using a base script)
        #    - acquisition_offload.sh
        # 2. Define as acquisition job (manager.submit_acquistion...)
        #    - monitor until job has finished (single-job monitoring)
        # 3. Ingress JSON file's contents the backend produced
        #    - note that the keys are dimension name strings: must convert
        #      first (could use _post_process_optimizer_suggestion)

        # - process config
        # - submit shell script
        # - wait until done
        # if not self.config:
        #    raise ValueError("Configuration not provided: TODO")

        if not self.job_manager:
            raise ValueError("No job manager configured: cannot offload acquisition")

        acq_config = {
            "args": {
                "optimizer": self.backend.type,
                "model_snapshot": str(model_snapshot),
                "data_snapshot": str(data_snapshot),
                "num_trials": num_new_cases,
                "output_path": Path(self.data_dir, DEFAULT_OFFLOAD_RESULT_FNAME),
            }
        }

        with pkg_resources.path("flowboost.optimizer", OFFLOAD_SCRIPT) as acq_sh:
            self.job_manager.submit_acquisition(script=acq_sh, config=acq_config)

    def handle_finished_acquisition_job(self, acq_job: JobV2):
        logging.info(f"Handling finished acquisition job: {acq_job}")
        result_json_f = Path(self.data_dir, DEFAULT_OFFLOAD_RESULT_FNAME)
        if not result_json_f.exists():
            raise FileNotFoundError(
                f"Acquisition result JSON not found: {result_json_f}"
            )

        if not self.job_manager:
            raise ValueError("Cannot process acquisition result without a job manager!")

        with open(result_json_f, "r") as json_f:
            data = json.load(json_f)

        # Standard checks
        if data.get("optimizer", "") != self.backend.type:
            raise ValueError(f"Incorrect optimizer type in result JSON: {data}")

        if data.get("status_finished", False) is True:
            logging.info("Backends reported optimization as finished: exiting")
            sys.exit("Optimization finished")

        logging.info(
            f"Loading acquisition result, created at: {data.get('created_at', 'unknown')}"
        )

        # Do post-processing step
        suggestions = self.backend._post_process_suggestion_parametrizations(
            data.get("parametrizations", {})
        )

        logging.info(f"Post-processing model suggestions (count={len(suggestions)})")
        new_cases = self._process_optimizer_suggestion(suggestions)

        for case in new_cases:
            # TODO pass script args automatically
            self.job_manager.submit_case(case)

    def attach_template_case(
        self,
        case: Union[Case, Path],
        add_files: list[str] = [],
        move_to_session_dir: bool = False,
    ):
        """
        Attaches a template case to the session, which new simulations will
        be derived from during the optimization task.

        By default, the case is not copied or otherwise altered to reduce the \
        potential for confusion. If you specify `move_dir_to_session`, the \
        case directory will be moved to the session directory: even then, the \
        case will never be altered, only cloned.

        Optionally, you can specify additional files and directories for the \
        `foamCloneCase` command to copy: this may include a non-0 starting \
        time directory. These files are copied each time the template is \
        used.

        Args:
            case (Case | Path): An OpenFOAM case to use as a template.
            add_files (list[str]): List of file-/directory-names to \
                include whenever the template case is used to derive new \
                simulations.
            move_to_session_dir (bool, optional): Move the entire template \
                case directory to the session data_dir.
        """
        if isinstance(case, Path):
            case = Case(case)

        if move_to_session_dir:
            new_path = self.data_dir / case.name
            shutil.move(case.path, new_path)

            # Case.data automatically updates on first access
            case.path = new_path
            logging.info(f"Moved case to session directory [{new_path}]")

        self._template_case = case
        self._template_case_add_files = add_files
        self.persist()

    def state(self) -> dict:
        """
        Return the session state as a dictionary.
        """
        state = {
            "session": {
                "name": self.name,
                "data_dir": str(self.data_dir),
                "case_dir": str(self.pending_dir),
                "archival_dir": str(self.archival_dir),
                "dataframe_format": self.dataframe_format,
                "created_at": self.created_at.isoformat(),
            },
            "template": {
                "path": str(self._template_case.path) if self._template_case else "",
                "additional_files": self._template_case_add_files,
            },
            "optimizer": {
                "type": self.backend.type,
                "offload_acquisition": self.backend.offload_acquisition,
            },
            "scheduler": {
                "type": self.job_manager.type if self.job_manager else "",
                "job_limit": self.job_manager.job_limit if self.job_manager else 1,
                # "OpenFOAM": self.config.get("scheduler", {}).get("OpenFOAM", {}),
                # "acquisition": self.config.get("scheduler", {}).get("acquisition", {})
            },
        }

        return state

    def persist(self, to_file: str = config.DEFAULT_CONFIG_NAME):
        config.save(self.state(), self.data_dir, to_file)

    def restore(self, from_file: str = config.DEFAULT_CONFIG_NAME):
        data = config.load(self.data_dir, from_file)

        # [session]
        self.name = str(data.get("session", {}).get("name"))
        self.data_dir = Path(data.get("session", {}).get("data_dir"))
        self.pending_dir = Path(data.get("session", {}).get("case_dir"))
        self.archival_dir = Path(data.get("session", {}).get("archival_dir"))
        self.dataframe_format = str(
            data.get("session", {}).get("dataframe_format", "polars")
        )
        self.created_at = datetime.fromisoformat(
            data.get("session", {}).get("created_at")
        )

        # [template]
        self._template_case = Case.try_restoring(
            str(data.get("template", {}).get("path"))
        )
        self._template_case_add_files = [
            str(item) for item in data.get("template", {}).get("additional_files")
        ]

        # [optimizer]
        backend_type = str(data.get("optimizer", {}).get("type", "Ax"))
        self.backend.create(backend_type)
        offload = data.get("optimizer", {}).get("offload_acquisition", False)
        if offload:
            self.backend.offload_acquisition = offload

        # [scheduler]
        scheduler = data.get("scheduler", {}).get("type", "")
        job_limit = data.get("scheduler", {}).get("job_limit", 1)

        if scheduler != "":
            self.job_manager = Manager.create(
                scheduler=scheduler, wdir=self.data_dir, job_limit=job_limit
            )

        logging.info(f"Session restored from {from_file}")

    def _process_optimizer_suggestion(
        self, suggestions: list[dict[Dimension, Any]]
    ) -> list[Case]:
        """
        The optimizer backend is expected to return a list of dictionaries,
        each dicitonary representing a set of modifications to apply to the
        template case.

        The dictionary is expected to be keyed by (search space) Dimensions.

        For the Ax backend, these are called "Trial Parametrizations".

        Args:
            suggestions (list[dict]): Optimizer suggestion

        Returns:
            list[Case]: Suggestions processed into Cases
        """
        if self._template_case is None:
            raise ValueError("Template case is None: cannot generate cases")

        # Get prefix for this batch
        stage_prefix = self._get_next_stage_prefix()

        new_cases: list[Case] = []

        for case_i, suggestion_dict in enumerate(suggestions):
            # Prefix format is `stage_001_01`
            uid = unique_id()
            name = f"stage{stage_prefix:03d}.{case_i+1:02d}_{uid}"

            # Clone template case
            case = self._template_case.clone(
                clone_to=Path(self.pending_dir, name), add=self._template_case_add_files
            )

            case.id = uid
            case._generation_index = f"{stage_prefix:03d}.{case_i+1:02d}"

            # Apply all suggestions
            self._apply_suggestions_to_case(case, suggestion_dict)

            # Persist case
            case.persist_to_file()

            # Update case metadata with the applied suggestions
            suggestion_metadata = self._serializable_suggestion_dict(suggestion_dict)
            header = "optimizer-suggestion"
            case.update_metadata(suggestion_metadata, header)

            new_cases.append(case)

        # Persist session
        self.persist()
        return new_cases

    def _apply_suggestions_to_case(self, case: Case, suggestions: dict[Dimension, Any]):
        """
        Apply a list of optimizer suggestions (dim -> new_value) to a Case.

        Args:
            case (Case): Case to modify
            suggestion (dict[Dimension, Any]): Suggestions to apply

        Raises:
            ValueError: If a Dimension is missing a DictionaryLink
        """
        for dim, new_val in suggestions.items():
            if dim.linked_entry is None:
                raise ValueError(f"Dimension '{dim.name}' not linked to an entry")

            reader = dim.linked_entry.reader(case.path)

            if isinstance(reader, Dictionary):
                raise ValueError(
                    f"Dimension '{dim.name}' linked incorrectly: Entry missing"
                )

            reader.write(new_value=new_val)

    def _serializable_suggestion_dict(
        self, suggestion_dict: dict[Dimension, Any]
    ) -> dict:
        ser_dict = {}
        for dim, value in suggestion_dict.items():
            d = {
                "value": value,
                "dim_type": dim.type,
                "dim_link": str(type(dim.linked_entry)),
            }
            ser_dict[dim.name] = d

        return ser_dict

    def _get_next_stage_prefix(self) -> int:
        # Returns next stage index
        cases = self.get_all_cases(include_failed=True)

        if not cases:
            return 1

        # Separate names by '.' and remove the "stage" prefix
        names = [c.name.split(".")[0].replace("stage", "") for c in cases]

        # Process to ints
        int_names = [int(name) for name in names if name.isdigit()]

        return max(int_names) + 1

    def _ensure_dirs(self):
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        if not self.pending_dir.exists():
            self.pending_dir.mkdir(parents=True, exist_ok=True)

        if not self.archival_dir.exists():
            self.archival_dir.mkdir(parents=True, exist_ok=True)

    def _verify_search_space_in_template(self):
        """
        Verifies all dictionary entries configured in the search space can
        be found in the template case.

        * There must be a configured search space
        * All entries must be present
        * All configured dimensions must be linked
            * ... unless there is an explicit override (TODO)
        """
        if self._template_case is None:
            raise ValueError("No template case configured for session")

        if not self._template_case.path.exists():
            raise FileNotFoundError(f"Template case not found [{self._template_case}]")

        if not self.backend.dimensions:
            raise ValueError(
                "No search space found for backend (session.backend.search_space)"
            )

        # Next, for each dimension, verify the linked entry exists in template
        for dim in self.backend.dimensions:
            if not isinstance(dim.linked_entry, DictionaryLink):
                raise ValueError(
                    f"Dictionary link must be a DictionaryLink (dim='{dim.name}')"
                )

            # Running the reader should yield an Entry
            entry_reader = dim.linked_entry.reader(self._template_case.path)

            # If the type is a DictionaryReader, the user has only provided a
            # link to a foam dictionary file, but no entry
            if isinstance(entry_reader, DictionaryReader):
                logging.error(
                    f"Dim='{dim.name}' not linked to any entry in {entry_reader}"
                )
                logging.error("Missing '.entry()' call?")
                raise ValueError("Cannot continue without a linked dictionary entry")

            # If the reader yields a none, the entry does not exist
            if not entry_reader:
                logging.error(f"Invalid configuration for dimension='{dim.name}'")
                raise ValueError(
                    f"Entry does not exist in template case dict [{dim.linked_entry}]"
                )

        logging.info("Search space correctly linked to template case")

    def _delete_all_data(self):
        """
        Deletes all non-archive data from the session. The removal is only
        carried out for a session that has an external archival
        directory, or an empty data_dir.
        """
        cfg = self.data_dir / config.DEFAULT_CONFIG_NAME

        if not cfg.exists():
            raise ValueError(
                "Will not remove folder unless it contains known directories"
            )

        has_case_data = any(Path(self.pending_dir).iterdir())

        if (self.data_dir == self.archival_dir) and has_case_data:
            raise ValueError("Refusing to delete: case data in /case_data")

        shutil.rmtree(self.data_dir)

        if self.archival_dir.exists():
            logging.warning(f"Not removing archival directory [{self.archival_dir}]")

    @staticmethod
    def _pretty_print_cases(cases: list[Case]):
        print(f"=== New cases ({len(cases)}) ===")
        for i, case in enumerate(cases, 1):
            print(f"[{i}] {str(case)}")
