#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 08:59:16 2022

@author: caleb_m_koch
"""

import os
from typing import Any, Dict, Union

from agora.pandas import Tensor
from agora.time import TIME_RESOLUTION, TS, TimeRange
from kedro.pipeline.node import Node

from .item import Item
from .store import PyStore
from .utils import set_path


class PyStoreClient:
    """
    This client is main entrypoint for writing/reading data.
    """

    def __init__(self, filepath_to_database: str, pystore_name: str, version: str):
        """
        Initialize PyStore client with database config.
        """
        filepath_to_database = os.path.expanduser(filepath_to_database)
        set_path(filepath_to_database)
        self.store = PyStore(pystore_name)
        self.collection = self.store.collection(version)

    def read(self, name: Union[Node, str]) -> Item:
        """
        Read data from the pystore database.
        """
        name_to_read = name.name if isinstance(name, Node) else name
        return self.collection.item(name_to_read)

    def write(
        self,
        name: str,
        data: Any,
        metadata: Dict[str, Any] = None,
        always_overwrite: bool = False,
    ):
        """
        General purpose entry point for writing and/or appending data, depending on what is most appropriate.
        """
        metadata = metadata or {}

        if always_overwrite or name not in self.collection.list_items_with_data():
            self._write(name=name, data=data, metadata=metadata)
        else:
            self._append(name=name, data=data)

    def _write(
        self,
        name: str,
        data: Any,
        metadata: Dict[str, Any] = None,
    ):
        """
        Write data to dataspace. Importantly, "end_timestamp" is added to metadata to enable
        smart appending.
        """
        metadata = metadata or {}
        metadata["end_timestamp"] = str(self._get_end_timestamp(data))
        path = self.collection._item_path(name)
        if not path.exists():
            path.mkdir()
        self.collection.write(item=name, data=data, metadata=metadata, overwrite=True)

    def _append(self, name: str, data: Any):
        """
        Append data to pre-existing data in the database in a way that is idempotent.
        """

        existing_data = self.read(name=name)
        metadata = existing_data.metadata

        existing_end_timestamp = TS(metadata["end_timestamp"]) + TIME_RESOLUTION
        new_end_timestamp = self._get_end_timestamp(data)

        # Add data to the database only if new entries exist.
        if new_end_timestamp > existing_end_timestamp:
            new_time_range = TimeRange(
                start=existing_end_timestamp, end=new_end_timestamp + TIME_RESOLUTION
            )
            new_metadata = {**metadata, "end_timestamp": str(new_end_timestamp)}
            self.collection.append(
                item=name,
                data=data.pipe(new_time_range.view),
                metadata=new_metadata,
            )

    @staticmethod
    def _get_end_timestamp(df: Tensor) -> TS:
        """
        Get the last timestamp.
        """
        return df.index[-1] if df.index.nlevels == 1 else max(df.index[-1])
