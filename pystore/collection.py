#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# PyStore: Flat-file datastore for timeseries data
# https://github.com/ranaroussi/pystore
#
# Copyright 2018-2020 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Union

import pandas as pd

from . import utils
from .item import Item
from .utils import make_path

Tensor = Union[pd.Series, pd.DataFrame]


class Collection(object):
    def __repr__(self):
        return "PyStore.collection <%s>" % self.collection

    def __init__(self, collection, datastore, engine="fastparquet"):
        self.engine = engine
        self.datastore = datastore
        self.collection = collection

    def _item_path(self, item, as_string=False):
        p = utils.make_path(self.datastore, self.collection, item)
        if as_string:
            return str(p)
        return p

    def list_items(self, **kwargs):
        dirs = utils.subdirs(utils.make_path(self.datastore, self.collection))
        if not kwargs:
            return set(dirs)

        matched = []
        for d in dirs:
            meta = utils.read_metadata(
                utils.make_path(self.datastore, self.collection, d)
            )
            del meta["_updated"]

            m = 0
            keys = list(meta.keys())
            for k, v in kwargs.items():
                if k in keys and meta[k] == v:
                    m += 1

            if m == len(kwargs):
                matched.append(d)

        return set(matched)

    def list_items_with_data(self):
        dirs = utils.subdirs(utils.make_path(self.datastore, self.collection))
        return set(
            [
                d
                for d in dirs
                if utils.make_path(
                    self.datastore, self.collection, d, "metadata.json"
                ).exists()
            ]
        )

    def item(self, item, snapshot=None, filters=None):
        return Item(
            item, self.datastore, self.collection, snapshot, filters, engine=self.engine
        )

    def write(
        self,
        item,
        data: Tensor,
        metadata: dict = None,
        overwrite: bool = False,
        as_pickle: bool = False,
    ):

        metadata = metadata or {}

        if utils.path_exists(self._item_path(item)) and not overwrite:
            raise ValueError(
                """
                Item already exists. To overwrite, use `overwrite=True`.
                Otherwise, use `<collection>.append()`"""
            )

        if not as_pickle:
            data_path = make_path(item, "data.parquet")
            data.to_parquet(
                self._item_path(data_path, as_string=True),
                compression="snappy",
                engine=self.engine,
            )
            metadata["as_pickle"] = False
        else:
            data_path = make_path(item, "data.pickle")
            pd.to_pickle(data, self._item_path(data_path, as_string=True))
            metadata["as_pickle"] = True

        metadata_path = make_path(item, "metadata.json")
        utils.write_metadata(
            utils.make_path(self.datastore, self.collection, metadata_path), metadata
        )

    def append(self, item, data, metadata: Dict[str, Any] = None):

        if not utils.path_exists(self._item_path(item)):
            raise ValueError("""Item do not exists. Use `<collection>.write(...)`""")

        current = pd.read_parquet(
            self._item_path(item, as_string=True), engine=self.engine
        )
        combined = current.append(data)

        current = self.item(item)

        self.write(
            item,
            combined,
            metadata=current.metadata | metadata,
            overwrite=True,
        )
