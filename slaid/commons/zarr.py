from typing import Tuple
import os

import zarr

from slaid.commons.base import ArrayFactory as BaseArrayFactory


class ArrayFactory(BaseArrayFactory):
    def __init__(self, store: str = None):
        self._store = store

    def empty(self, shape: Tuple[int, int], dtype: str):
        return zarr.creation.empty(shape, dtype=dtype, store=self._store)

    def zeros(self, shape: Tuple[int, int], dtype: str):
        return zarr.creation.zeros(shape, dtype=dtype, store=self._store)


class GroupArrayFactory(BaseArrayFactory):
    def __init__(self, name, store: str = None, mode: str = "a"):
        if store:
            ext = os.path.splitext(store)[1]
            if ext == ".zarr":
                self._store = zarr.DirectoryStore(store)
            elif ext == ".zip":
                self._store = zarr.ZipStore(store, mode=mode)
        else:
            self._store = store
        self.name = name
        self._root = zarr.group(store=self._store)

    def empty(self, shape: Tuple[int, int], dtype: str):
        return self._root.empty(self.name, shape=shape, dtype=dtype)

    def zeros(self, shape: Tuple[int, int], dtype: str):
        return self._root.zeros(self.name, shape=shape, dtype=dtype)
