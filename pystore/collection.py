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

import numpy as np
import pandas as pd

from . import utils
from .item import Item
from .utils import make_path

Tensor = Union[pd.Series, pd.DataFrame]
Numeric = [np.dtype("float32"), np.dtype("float64"), np.dtype("int64")]


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
        try:
            return set(
                [
                    d
                    for d in dirs
                    if utils.make_path(
                        self.datastore, self.collection, d, "metadata.json"
                    ).exists()
                ]
            )
        except FileNotFoundError:
            return None

    def item(self, item: str):
        """
        Return an instance of the item.
        """
        return Item(item=item, datastore=self.datastore, collection=self.collection)

    def write(
        self,
        item,
        data: Tensor,
        metadata: dict = None,
        overwrite: bool = False,
    ):

        metadata = metadata or {}
        metadata["file_type"] = self._infer_file_type_from_data(data)

        path = self._item_path(item)
        if path.exists() and not overwrite:
            raise ValueError(
                """
                Item already exists. To overwrite, use `overwrite=True`.
                Otherwise, use `<collection>.append()`"""
            )

        data_path = make_path(item, "data." + metadata["file_type"])
        data_path = self._item_path(data_path, as_string=True)
        if metadata["file_type"] == "parquet":
            data.to_parquet(data_path, compression="snappy", engine=self.engine)
        elif metadata["file_type"] == "pickle":
            pd.to_pickle(data, data_path)

        metadata_path = make_path(item, "metadata.json")
        utils.write_metadata(
            utils.make_path(self.datastore, self.collection, metadata_path), metadata
        )

    def append(self, item: str, data: Tensor, metadata: Dict[str, Any] = None):
        """
        Append data to existing data.
        """
        metadata = metadata or {}
        existing_item = self.item(item)
        self.write(
            item=item,
            data=existing_item.data.append(data),
            metadata=existing_item.metadata | metadata,
            overwrite=True,
        )

    @staticmethod
    def _infer_file_type_from_data(df: Tensor) -> str:
        """
        Infer the correct file type, given the data itself. The general goal is to read/write using parquet, unless
        known problems exist.
        """
        if isinstance(df, pd.Series):
            return "pickle"
        elif df.index.nlevels > 1:
            return "pickle"
        elif any([type(t) not in Numeric for t in df.dtypes]):
            return "pickle"
        else:
            return "parquet"
