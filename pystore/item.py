#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
PyStore: Flat-file datastore for timeseries data
"""

import pandas as pd

from . import utils


class Item(object):
    def __repr__(self):
        return "PyStore.item <%s/%s>" % (self.collection, self.item)

    def __init__(
        self,
        item,
        datastore,
        collection,
        snapshot=None,
        filters=None,
        engine="fastparquet",
    ):
        self.engine = engine
        self.datastore = datastore
        self.collection = collection
        self.snapshot = snapshot
        self.item = item

        self._metadata_path = utils.make_path(datastore, collection, item)
        self.metadata = utils.read_metadata(self._metadata_path)

        if not self._metadata_path.exists():
            raise ValueError(
                "Item `%s` doesn't exist. "
                "Create it using collection.write(`%s`, data, ...)" % (item, item)
            )

        if not self.metadata["as_pickle"]:
            self._path = utils.make_path(datastore, collection, item, "data.parquet")
            self.data = pd.read_parquet(self._path, engine=self.engine, filters=filters)
        else:
            self._path = utils.make_path(datastore, collection, item, "data.pickle")
            self.data = pd.read_pickle(self._path)
