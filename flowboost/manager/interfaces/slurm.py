from pathlib import Path

from flowboost.manager.manager import Manager


class Slurm(Manager):
    def __init__(self, wdir: Path | str, job_limit: int) -> None:
        super().__init__(wdir, job_limit)
