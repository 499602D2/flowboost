import json
import logging
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flowboost.openfoam.case import Case, Status
from flowboost.utilities.time import td_format

SUPPORTED_SCHEDULERS = ("Local", "SGE")


class Manager(ABC):
    def __init__(self, wdir: Path | str, job_limit: int) -> None:
        """
        An abstract job manager class that can be easily extended.

        Args:
            wdir (Path | str): Working directory of the manager.
            job_limit (int): Maximum number of jobs that can be running.
        """
        # Session working directory
        self.wdir: Path = Path(wdir).resolve().absolute()

        # Limits
        self.job_limit: int = job_limit
        self.monitoring_interval: int = 60

        # Currently tracked jobs
        self.job_pool: set[JobV2] = set()

        # In case user requests acquisition jobs to be off-loaded to the
        # cluster environment, we track these jobs separately.
        self.acquisition_job: Optional[JobV2] = None

        # Default prefix for submitted job names. Used to identify jobs
        # submitted by the optimizer; may be redundant, since we do
        # stateful tracking. Can still be helpful to users, though.
        self.job_prefix: str = "flwbst_"

        # Automatic "type" to avoid mixing and matching persistence
        self.type: str = self.__class__.__name__
        self.persistence_fname: str = f"job_tracking_{self.type}.json"

        # Ensure the manager is available
        if not self._is_available():
            raise ValueError(f"Scheduler '{self.type}' not available")

        # Restore state if available
        if Path(self.wdir, self.persistence_fname).exists():
            logging.info(f"Restoring job manager state ({self.type})")
            self._restore_state(self.wdir)
        else:
            logging.info(f"Created new job manager ({self.type})")
            self._save_state()

    @staticmethod
    @abstractmethod
    def _is_available() -> bool:
        """
        Verify the scheduler is available. This can be done by testing if the
        management commands are in the current PATH.

        Returns:
            bool: Scheduler availability
        """
        pass

    @abstractmethod
    def _submit_job(
        self,
        job_name: str,
        submission_cwd: Path,
        script: Path,
        script_args: dict[str, Any] = {},
    ) -> Optional[str]:
        """
        Submit a job to the execution backend using a shell script. Note, that
        this interface is expected to be capable of submitting arbitrary shell
        scripts: this is for supporting model update and acquisition offloading
        to the external execution environment.

        Args:
            job_name (str): Name of job
            submission_cwd (Path): Working directory the submission is done in
            script (Path): Abslute path to submission shell script (Allrun)
            script_args (dict[str, Any], optional): Additional arguments to be \
                passed to the shell script. Defaults to {}.

        Returns:
            Optional[str]: A job ID that can be used to monitor the submitted \
                job during its evaluation. None if the submission fails.
        """
        pass

    @abstractmethod
    def _cancel_job(self, job_id: str) -> bool:
        """
        Cancels a job that was once submitted. This interface will only be
        called for jobs that are known to be running.

        Args:
            job_id (str): Job to cancel.

        Returns:
            bool: Cancellation success indicator
        """
        pass

    @abstractmethod
    def _job_has_finished(self, job_id: str) -> bool:
        pass

    @staticmethod
    def create(scheduler: str, wdir: Path, job_limit: int) -> "Manager":
        """
        An interface for initializing a new job manager for one of the
        supported job schedulers.

        Currently supported:
            - `Local`: local, uncontainerized evaluation
            - `SGE`: Sun Grid Engine, using the qsub/qstat CLI commands

        Args:
            scheduler (str): Case-insensitive scheduler name (e.g. "SGE")
            wdir (Path): Working directory for session
            job_limit (int): Limit of concurrent jobs

        Raises:
            NotImplementedError: If scheduler is not supported

        Returns:
            Manager: A job manager implementing `Manager` interface
        """
        from flowboost.manager.interfaces.local import Local
        from flowboost.manager.interfaces.sge import SGE

        match scheduler.lower():
            case "local":
                manager = Local(wdir=wdir, job_limit=job_limit)
            case "sge":
                manager = SGE(wdir=wdir, job_limit=job_limit)
            case "slurm":
                raise NotImplementedError("Slurm manager not implemented")
            case _:
                raise NotImplementedError(
                    f"Scheduler '{scheduler}' not in {SUPPORTED_SCHEDULERS}"
                )

        return manager

    def free_slots(self) -> int:
        return self.job_limit - len(self._get_running_jobs())

    def submit_case(self, case: Case, script_args: dict[str, Any] = {}) -> bool:
        """
        Public interface for submitting a Case to the execution environment.

        Args:
            case (Case): Case to submit
            script_args (dict[str, Any], optional): Additional arguments to \
                pass to the Allrun script. Defaults to {}.

        Raises:
            FileNotFoundError: If Allrun-script is not found
            ValueError: If job submission fails

        Returns:
            bool: Submission success status
        """
        script = case.submission_script()
        if not script:
            raise FileNotFoundError(f"Allrun script not found: {case}")

        job_name = f"{self.job_prefix}{case.name}"
        job_id = self._submit_job(
            job_name=job_name,
            submission_cwd=case.path,
            script=script,
            script_args=script_args,
        )

        if not job_id:
            logging.error(f"Job submission failed for case={case} (args={script_args})")
            return False

        job = JobV2(id=job_id, name=f"{self.job_prefix}{case.name}", wdir=case.path)

        self.job_pool.add(job)
        self._save_state()

        logging.info(f"Submitted job: {job}")

        # Update case metadata
        case.status = Status("submitted")
        case.persist_to_file()

        return True

    def submit_acquisition(self, script: Path, config: dict = {}) -> bool:
        if not script.exists():
            raise FileNotFoundError(f"Acquisition script not found [{script}]")

        job_name = f"{self.job_prefix}acquisition"
        acq_args = config.get("args", {})

        job_id = self._submit_job(
            job_name=job_name,
            submission_cwd=self.wdir,
            script=script,
            script_args=acq_args,
        )

        if not job_id:
            logging.error("Error submitting acquisition job")
            return False

        acq_job = JobV2(id=job_id, name=job_name, wdir=self.wdir)
        self.acquisition_job = acq_job

        self._save_state()
        return True

    def cancel_job(self, job: "JobV2") -> bool:
        if job not in self.job_pool:
            logging.error(f"Cannot cancel job: not in job pool ({job})")
            return True

        success = self._cancel_job(job.id)

        if not success:
            logging.error("Cancelling job failed")
            return False

        logging.info(f"Cancelled job: {job}")
        return success

    def do_monitoring(self) -> tuple[int, list["JobV2"], bool]:
        """
        Performs persistent monitoring of the running optimization jobs,
        emitting a tuple that indicates the number of free job slots,
        a list of Jobs that have finished and are ready to be processed, and
        a boolean indicating if the finished job was an off-loaded acquisition
        job.

        Returns:
            tuple[int, list[Job], bool]: Number of free slots, finished jobs, acquisition job
        """
        while True:
            if len(self.job_pool) == 0:
                return (self.job_limit, [], False)

            if self.acquisition_job:
                # Acquisition job is special, during which we do not care what has
                # finished
                if self._job_has_finished(self.acquisition_job.id):
                    logging.info(f"Acquisition job finished ({self.acquisition_job})")
                    job = self.acquisition_job
                    self.acquisition_job = None
                    return (0, [job], True)
                else:
                    logging.info(f"Acquisition still running ({self.acquisition_job})")
                    time.sleep(self.monitoring_interval)
                    continue

            finished_jobs = {
                job for job in self.job_pool if self._job_has_finished(job.id)
            }

            if finished_jobs:
                self.job_pool.difference_update(finished_jobs)
                self._save_state()

                free_slots = self.job_limit - len(self.job_pool)
                return (free_slots, list(finished_jobs), False)

            print(self._status_print())
            logging.info("No jobs finished, monitoring...")
            time.sleep(self.monitoring_interval)

        return (0, [])

    def move_data_for_job(self, job: "JobV2", dest: Path) -> bool:
        if not self._job_has_finished(job.id):
            logging.warning(f"Job is still running, cannot move data ({str(job)})")
            return False

        # Path(dest).mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(str(job.wdir), str(dest))
            logging.info(f"Moved data for job {job.id} to {dest}")
            return True
        except Exception:
            logging.exception(f"Moving data for job failed, dest=[{dest}] ({str(job)})")
            return False

    def _get_running_jobs(self) -> list["JobV2"]:
        """
        Return a list of all currently running job IDs. If the interface is
        not capable of supporting the detection of queued jobs, the parameter
        can be ignored.
        """
        return [j for j in self.job_pool if not self._job_has_finished(j.id)]

    def _status_print(self) -> str:
        if not self.job_pool and not self.acquisition_job:
            return "No jobs are currently being tracked."

        header = "==== Job Tracking Summary ===="
        status_lines = [header]

        if self.acquisition_job:
            status = f"[acq.] {self.acquisition_job}"
            status_lines.append(status)

        jobs_sorted = sorted(self.job_pool, key=lambda job: job.created_at)
        for i, job in enumerate(jobs_sorted, 1):
            status = f"[{i}] {job}"
            status_lines.append(status)

        status_lines.append("=" * len(header))
        return "\n".join(status_lines)

    def _retrieve_running_case(self, job: "JobV2") -> Optional[Case]:
        """
        Provides lazy data access to a case that is still running.

        Args:
            job_id (str): Job ID to look up

        Returns:
            Optional[Case]: Lazily loaded Case, if path exists.
        """
        if not job.wdir.exists():
            logging.warning(f"Job wdir does not exist: still pending? ({str(job)})")
            return None

        return Case(job.wdir)

    def _state(self) -> dict:
        state = {
            "type": self.type,
            "wdir": str(self.wdir),
            "job_limit": self.job_limit,
            "job_prefix": self.job_prefix,
            "monitoring_interval": self.monitoring_interval,
            "job_pool": [job.to_dict() for job in self.job_pool],
            "acquisition_job": None,
        }

        if self.acquisition_job:
            state["acquisition_job"] = asdict(self.acquisition_job)

        return state

    def _save_state(self):
        with open(Path(self.wdir, self.persistence_fname), "w") as f:
            json.dump(self._state(), f, indent=4)

    def _restore_state(self, wdir: Path):
        """
        Restores the manager's existing state (from default tracking file)

        Raises:
            ValueError: If manager types do not match
        """
        if not Path(wdir, self.persistence_fname).exists():
            raise FileNotFoundError(
                f"Manager state not found: {self.persistence_fname}"
            )

        with open(Path(wdir, self.persistence_fname), "rb") as jsonf:
            state = json.load(jsonf)

        if state["type"] != self.type:
            raise ValueError(f"Manager mismatch ({state['type']} != {self.type})")

        # self.wdir = state["wdir"]
        # self.job_limit = state["job_limit"]
        self.job_prefix = state["job_prefix"]
        self.monitoring_interval = state["monitoring_interval"]
        self.job_pool = {
            JobV2.from_dict(job_state) for job_state in state.get("job_pool", [])
        }

        if len(self.job_pool) > 0:
            logging.info(f"Restored {len(self.job_pool)} job(s):")
            for i, job in enumerate(self.job_pool, start=1):
                logging.info(f"\t[{i}] {job}")

        if state.get("acquisition_job", None):
            self.acquisition_job = JobV2(**state["acquisition_job"])
            logging.info(f"Restored acquisition job: {self.acquisition_job}")

        logging.info("Restored job manager")

    @classmethod
    def _construct_scipt_args(cls, args: dict, sep: str = ",") -> str:
        return sep.join(f'{k}="{v}"' for k, v in args.items())


@dataclass(frozen=True)
class JobV2:
    id: str
    name: str
    wdir: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def runtime(self) -> str:
        elapsed = datetime.now(timezone.utc) - self.created_at
        return td_format(elapsed)

    def __str__(self) -> str:
        return f"id='{self.id}' name='{self.name}' running=({self.runtime()})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "wdir": self.wdir.as_posix(),
            "created_at": self.created_at.isoformat(),
        }

    @staticmethod
    def from_dict(data: dict):
        data["wdir"] = Path(data["wdir"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return JobV2(**data)
