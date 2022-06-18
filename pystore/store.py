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

import os
import shutil

from . import utils
from .collection import Collection


class PyStore(object):
    """
    The main store object used for retrieving data. A "PyStore" contains "Collection"s,
    and "Collection"s contain(s) "Item"s.
    """

    def __repr__(self):
        """
        String representation of PyStore object.
        """
        return "PyStore.datastore <%s>" % self.datastore

    def __init__(self, datastore: str):
        """
        Parameters
        -------------
        datastore: str
            String name of the PyStore database.
        """

        datastore_path = utils.get_path()
        if not utils.path_exists(datastore_path):
            os.makedirs(datastore_path)

        self.datastore = utils.make_path(datastore_path, datastore)

        # Create PyStore metadata, if it doesn't exist
        if not utils.path_exists(self.datastore):
            os.makedirs(self.datastore)
            utils.write_metadata(
                utils.make_path(self.datastore, "metadata.json"),
                {"engine": "fastparquet"},
            )
            self.engine = "fastparquet"
        else:
            # Otherwise retrieve PyStore metadata
            metadata = utils.read_metadata(self.datastore)
            self.engine = metadata["engine"]

        self.collections = self.list_collections()

    def _create_collection(self, collection, overwrite: bool = False):
        """
        Create collection instance.
        """
        # create collection (subdir)
        collection_path = utils.make_path(self.datastore, collection)
        if utils.path_exists(collection_path):
            if overwrite:
                self.delete_collection(collection)
            else:
                raise ValueError(
                    "Collection exists! To overwrite, use `overwrite=True`"
                )

        os.makedirs(collection_path)
        os.makedirs(utils.make_path(collection_path, "_snapshots"))

        self.collections = self.list_collections()
        return Collection(collection, self.datastore)

    def delete_collection(self, collection):
        """
        Delete a collection and all of its items.
        """
        shutil.rmtree(utils.make_path(self.datastore, collection))

        self.collections = self.list_collections()
        return True

    def list_collections(self):
        """
        See the list of collections contained in this PyStore.
        """
        return utils.subdirs(self.datastore)

    def collection(self, collection: str, overwrite: bool = False):
        """
        Get a collection instance, denoted by its string name.
        """
        if collection in self.collections and not overwrite:
            return Collection(collection, self.datastore, self.engine)

        # create collection if it doesn't already exist.
        self._create_collection(collection, overwrite)
        return Collection(collection, self.datastore, self.engine)

    def item(self, collection: str, item: str):
        """
        Allow data retrieval with a (collection: str, item: str) pair.
        """
        return self.collection(collection).item(item)
