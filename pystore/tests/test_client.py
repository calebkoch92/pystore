#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file tests the PyStore client.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import shutil

from pystore.client import PyStoreClient
from pystore.utils import make_path


class TestPyStoreClient:

    index = pd.date_range(start="2000", periods=20, freq="d", tz="America/New_York")

    @pytest.mark.parametrize("filepath_to_database", ["~/PYSTORE_TEST"])
    @pytest.mark.parametrize("store_name", ["TEST"])
    @pytest.mark.parametrize("version", ["0.0.0"])
    @pytest.mark.parametrize(
        "name, data, metadata, always_overwrite",
        [
            pytest.param(
                "abc",
                pd.DataFrame(
                    {
                        "A": pd.Series(1.0, index=index),
                        "B": pd.Series(0.0, index=index),
                    },
                ),
                None,
                True,
                id="Test pd.DataFrame",
            )
        ],
    )
    def test_write(
        self,
        name: str,
        data: Any,
        metadata: Dict,
        always_overwrite: bool,
        filepath_to_database: str,
        store_name: str,
        version: str,
    ):
        """
        Test that the pandas and non-pandas data is written correctly.
        """
        path = make_path(filepath_to_database, store_name, version)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()

        store = PyStoreClient(
            filepath_to_database=filepath_to_database,
            pystore_name=store_name,
            version=version,
        )

        store.write(
            name=name, data=data, metadata=metadata, always_overwrite=always_overwrite
        )

        written_data = store.read(name)

        assert written_data.data.equals(data)
        assert written_data.metadata == metadata

        # if path.exists():
        #     shutil.rmtree(path)
