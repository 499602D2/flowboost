from pathlib import Path

import polars as pl

from .base import Data


class PolarsData(Data[pl.DataFrame]):
    """Polars-backed OpenFOAM data reader.

    Args:
        path: Path to the OpenFOAM case directory.
        low_memory: Trade performance for lower memory usage.
        streaming: Use Polars' streaming engine for collection.
    """

    def __init__(
        self, path: Path, *, low_memory: bool = False, streaming: bool = False
    ) -> None:
        super().__init__(path, low_memory=low_memory)
        self.streaming: bool = streaming

    def _read_file(self, file: Path, comment: str, separator: str) -> pl.DataFrame:
        cols = self._discover_columns(file, comment=comment, delim=separator)
        kwargs: dict = dict(
            comment_prefix=comment,
            has_header=False,
            separator=separator,
            low_memory=self.low_memory,
        )
        if cols is not None:
            kwargs["new_columns"] = cols

        scan = pl.scan_csv(file, **kwargs)
        if self.streaming:
            return scan.collect(engine="streaming")
        return scan.collect()

    def _concat(self, frames: list[pl.DataFrame]) -> pl.DataFrame:
        return pl.concat(frames)
