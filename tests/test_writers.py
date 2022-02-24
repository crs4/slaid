#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import pytest

import slaid.writers.zarr_io as zarr_io


@pytest.mark.parametrize('storage_cls,filename',
                         [(zarr_io.ZarrStorage, 'test.zip'),
                          (zarr_io.ZarrStorage, 'test.zarr')])
@pytest.mark.parametrize('tile_size', [None, 256])
def test_mask_to_zarr(storage_cls, filename, tmp_path, mask, tile_size):
    name = 'test'
    store_path = os.path.join(tmp_path, filename)
    storage = storage_cls(name, store_path)
    array = storage.zeros((10, 10), 'uint8')
    mask.array = array
    if tile_size:
        mask.tile_size = tile_size
    storage.write(mask)
    res = storage.load()
    assert res == mask


@pytest.mark.parametrize('storage_cls, filename',
                         [(zarr_io.ZarrStorage, 'test.zip'),
                          (zarr_io.ZarrStorage, 'test.zarr')])
def test_checks_zarr_path_has_masks(storage_cls, filename, tmp_path):
    name = 'test'
    store_path = os.path.join(tmp_path, filename)
    storage = storage_cls(name, store_path)
    assert storage.mask_exists() is False
    storage.zeros((10, 10), 'uint8')
    assert storage.mask_exists()
