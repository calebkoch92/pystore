#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
PyStore: Flat-file datastore for timeseries data
"""

from functools import cached_property
from typing import Union

import pandas as pd

from . import utils

Tensor = Union[pd.Series, pd.DataFrame]


class Item(object):
    """
    A class object representing data in the database. Item has two key properties:
    """

    def __repr__(self):
        return "PyStore.item <%s/%s>" % (self.collection, self.item)

    def __init__(
        self,
        item: str,
        datastore: str,
        collection: str,
    ):
        """
        Parameters
        -----------
        item: str
            The name of the item as stored in the database.

        datastore: str
            The name of the datastore.

        collection: str
            The name of the collection.

        Note that the file type is inferred from information in the metadata.
        """
        self.datastore = datastore
        self.collection = collection
        self.item = item

        self._metadata_path = utils.make_path(datastore, collection, item)

        if not self._metadata_path.exists():
            raise ValueError(
                "Item `%s` doesn't exist. "
                "Create it using collection.write(`%s`, data, ...)" % (item, item)
            )

        self.metadata = utils.read_metadata(self._metadata_path)
        self.file_type = self.metadata["file_type"]
        self._data_path = utils.make_path(
            datastore, collection, item, "data." + self.file_type
        )

    @cached_property
    def data(self) -> Tensor:
        """
        Return the data from the database.
        """
        if self.file_type == "parquet":
            return pd.read_parquet(self._data_path, engine="fastparquet")
        elif self.file_type == "pickle":
            return pd.read_pickle(self._data_path)
        else:
            ValueError("The file type could not be inferred from the metadata.")
