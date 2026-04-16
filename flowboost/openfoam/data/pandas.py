from pathlib import Path

import pandas as pd

from .base import Data


class PandasData(Data[pd.DataFrame]):
    """Pandas-backed OpenFOAM data reader.

    Args:
        path: Path to the OpenFOAM case directory.
        low_memory: Trade performance for lower memory usage.
    """

    def _read_file(self, file: Path, comment: str, separator: str) -> pd.DataFrame:
        cols = self._discover_file_header(file, comment=comment, delim=separator)
        return pd.read_csv(
            file,
            comment=comment,
            sep=separator,
            header=None,
            names=cols,
            low_memory=self.low_memory,
        )

    def _concat(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames)
