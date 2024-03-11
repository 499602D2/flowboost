import logging
from pathlib import Path

import polars as pl

from flowboost.openfoam.case import Case


def test_data_loading(data_dir):
    datadir = Path(data_dir)
    case = Case(datadir)
    logging.info(case)

    # Test FO discovery
    print(case)
    dir_names = case.data.postProcessing_directory_names()
    assert dir_names, "Did not discover any postProcessing names"

    # Try reading all
    for fo_name in case.data.postProcessing_directory_names():
        df = case.data.simple_function_object_reader(fo_name)
        assert not df.is_empty(), f"Read an empty dataframe: {df}"
        print(df)

    # Read a dataframe by name
    # print(f"Loading {test_fo}")
    time_df: pl.DataFrame = case.data.simple_function_object_reader("time")
    assert not time_df.is_empty(), f"Read an empty dataframe: {time_df}"
    print(time_df)

    # Last value of 'cpu' column
    last_cpu = time_df.select(pl.last("cpu"))
    print(last_cpu)

    # Entire "clock" column
    cpu_col = time_df.select(pl.col("clock"))
    print(cpu_col)
