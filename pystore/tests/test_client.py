#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file tests the PyStore client.
"""

import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from pystore.client import PyStoreClient
from pystore.tests.utils import ClassToBeWritten
from pystore.utils import make_path


class TestPyStoreClient:

    index = pd.date_range(start="2000", periods=20, freq="d", tz="America/New_York")

    @pytest.fixture
    def path(
        self,
        filepath_to_database: str,
        store_name: str,
        version: str,
    ) -> Path:
        return make_path(filepath_to_database, store_name, version)

    @pytest.fixture
    def store(
        self,
        filepath_to_database: str,
        store_name: str,
        version: str,
    ) -> PyStoreClient:
        return PyStoreClient(
            filepath_to_database=filepath_to_database,
            pystore_name=store_name,
            version=version,
        )

    @pytest.mark.parametrize("filepath_to_database", ["/"])
    @pytest.mark.parametrize("store_name", ["TEST"])
    @pytest.mark.parametrize("version", ["0.0.0"])
    @pytest.mark.parametrize("always_overwrite", [True])
    @pytest.mark.parametrize(
        "name, data, metadata",
        [
            pytest.param(
                "abc",
                pd.DataFrame(
                    {
                        "A": pd.Series(1.0, index=index),
                        "B": pd.Series(0.0, index=index),
                    },
                    dtype=np.float64,
                ),
                None,
                id="Test pd.DataFrame",
            ),
            pytest.param(
                "abc",
                pd.DataFrame(
                    np.nan,
                    index=index,
                    columns=[str(a) for a in range(5)],
                    dtype=np.float64,
                ),
                None,
                id="Test empty pd.DataFrame",
            ),
            pytest.param(
                "abc",
                pd.DataFrame(
                    np.inf,
                    index=index,
                    columns=[str(a) for a in range(5)],
                    dtype=np.float64,
                ),
                None,
                id="Test np.inf pd.DataFrame",
            ),
            pytest.param(
                "abc",
                pd.DataFrame(
                    {
                        "A": pd.Series(1.0, index=index),
                        "B": pd.Series(0.0, index=index),
                        "C": pd.Series(-1.0, index=index),
                    },
                    dtype=np.float64,
                ),
                {"metadata": 10.0},
                id="Metadata is successfully written do database",
            ),
            pytest.param(
                "abc",
                pd.DataFrame(
                    [
                        [ClassToBeWritten(a=1, b=2), ClassToBeWritten(a=4, b=5)],
                        [ClassToBeWritten(a=3, b=6), ClassToBeWritten(a=7, b=8)],
                    ],
                    index=pd.date_range(start="2000", freq="d", periods=2),
                    columns=pd.Index(["ABC", "DEF"], name="example"),
                    dtype=np.float64,
                ),
                {"metadata": 10.0},
                id="pandas objects with non-float entries can be stored via pickle",
            ),
            pytest.param(
                "example",
                pd.DataFrame(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, 1.0, 0.0],
                        [np.nan, 2.0, 0.0],
                        [np.nan, 1.0, np.nan],
                    ],
                    dtype=np.float64,
                    columns=["A", "B", "C"],
                    index=pd.MultiIndex.from_arrays(
                        [
                            pd.date_range(
                                start="2000-01-01", periods=4, tz="America/New_York"
                            ),
                            pd.date_range(
                                start="2000-01-02", periods=4, tz="America/New_York"
                            ),
                        ],
                        names=["start", "end"],
                    ),
                ),
                None,
                id="pandas with MultiIndex index saved with pickle",
            ),
        ],
    )
    def test_write_and_read(
        self,
        path: Path,
        store: PyStoreClient,
        name: str,
        data: Any,
        metadata: Dict,
        always_overwrite: bool,
    ):
        """
        Test that the pandas and non-pandas data is written correctly.
        """
        if path.exists():
            shutil.rmtree(path)

        store.write(
            name=name,
            data=data,
            metadata=metadata,
            always_overwrite=always_overwrite,
        )

        self.written_data_equals_expected_data(
            store=store, name=name, metadata=metadata, expected=data
        )

        if path.exists():
            shutil.rmtree(path)

    @pytest.mark.parametrize("filepath_to_database", ["/"])
    @pytest.mark.parametrize("store_name", ["TEST"])
    @pytest.mark.parametrize("version", ["0.0.0"])
    @pytest.mark.parametrize("name", ["append_example"])
    @pytest.mark.parametrize(
        "metadata, data, data_to_append, expected",
        [
            pytest.param(
                None,
                pd.Series(
                    1.0,
                    dtype=np.float64,
                    index=pd.DatetimeIndex(
                        [
                            pd.Timestamp("2000-01-01 06:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 07:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 08:00:00", tz="UTC"),
                        ],
                        name="end",
                    ),
                ),
                pd.Series(
                    0.0,
                    dtype=np.float64,
                    index=pd.DatetimeIndex(
                        [
                            pd.Timestamp("2000-01-01 09:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 10:00:00", tz="UTC"),
                        ],
                        name="end",
                    ),
                ),
                pd.Series(
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                    dtype=np.float64,
                    index=pd.DatetimeIndex(
                        [
                            pd.Timestamp("2000-01-01 06:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 07:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 08:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 09:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 10:00:00", tz="UTC"),
                        ],
                        name="end",
                    ),
                ),
                id="append works when data is disjoint",
            ),
            pytest.param(
                None,
                pd.Series(
                    np.nan,
                    dtype=np.float64,
                    index=pd.DatetimeIndex(
                        [
                            pd.Timestamp("2000-01-01 06:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 07:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 08:00:00", tz="UTC"),
                        ],
                        name="end",
                    ),
                ),
                pd.Series(
                    0.0,
                    dtype=np.float64,
                    index=pd.DatetimeIndex(
                        [
                            pd.Timestamp("2000-01-01 08:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 09:00:00", tz="UTC"),
                        ],
                        name="end",
                    ),
                ),
                pd.Series(
                    [np.nan, np.nan, np.nan, 0.0],
                    dtype=np.float64,
                    index=pd.DatetimeIndex(
                        [
                            pd.Timestamp("2000-01-01 06:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 07:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 08:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 09:00:00", tz="UTC"),
                        ],
                        name="end",
                    ),
                ),
                id="when data is not disjoint, append gives priority to data already in the database",
            ),
            pytest.param(
                None,
                pd.Series(
                    np.nan,
                    dtype=np.float64,
                    index=pd.DatetimeIndex(
                        [
                            pd.Timestamp("2000-01-01 06:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 07:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 08:00:00", tz="UTC"),
                        ],
                        name="end",
                    ),
                ),
                pd.Series(
                    0.0,
                    dtype=np.float64,
                    index=pd.DatetimeIndex(
                        [
                            pd.Timestamp("1999-01-01 08:00:00", tz="UTC"),
                            pd.Timestamp("1999-01-01 09:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 06:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 07:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 07:59:00", tz="UTC"),
                        ],
                        name="end",
                    ),
                ),
                pd.Series(
                    [np.nan, np.nan, np.nan],
                    dtype=np.float64,
                    index=pd.DatetimeIndex(
                        [
                            pd.Timestamp("2000-01-01 06:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 07:00:00", tz="UTC"),
                            pd.Timestamp("2000-01-01 08:00:00", tz="UTC"),
                        ],
                        name="end",
                    ),
                ),
                id="append no data if all data is before / during / coincident with existing data",
            ),
        ],
    )
    def test_append(
        self,
        path: Path,
        store: PyStoreClient,
        name,
        metadata,
        data,
        data_to_append,
        expected,
    ):

        if path.exists():
            shutil.rmtree(path)

        store.write(name, data, always_overwrite=True)
        store.write(name, data_to_append)

        self.written_data_equals_expected_data(
            store=store, name=name, metadata=metadata, expected=expected
        )

        if path.exists():
            shutil.rmtree(path)

    @staticmethod
    def written_data_equals_expected_data(
        store: PyStoreClient,
        name,
        metadata,
        expected,
    ):
        """
        A helper function for ensuring that written data in the database equals expected data.
        """
        written_data = store.read(name)

        # Assertions that ensure data -> write -> read -> data are the same.
        assert written_data.data.equals(expected)
        assert "_updated" in written_data.metadata.keys()
        assert pd.Timestamp(
            written_data.metadata["end_timestamp"]
        ) == PyStoreClient._get_end_timestamp(expected)
        if metadata is not None:
            for key, value in metadata.items():
                assert written_data.metadata[key] == value
