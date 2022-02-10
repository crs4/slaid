#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import subprocess
import unittest

import numpy as np
import pytest
import zarr

from slaid.commons.ecvl import BasicSlide as Slide
from slaid.writers import REGISTRY

DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = '/tmp/test-slaid'
input_ = os.path.join(DIR, 'data/PH10023-1.thumb.tif')
input_basename = os.path.basename(input_)
input_basename_no_ext = 'PH10023-1.thumb'
slide = Slide(input_)
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


def get_input_output(output):
    slide = Slide(input_)
    zarr_group = zarr.open_group(output)
    return slide, zarr_group


def _test_output(feature, output, slide, level, patch_size=1):
    assert output.attrs['filename'] == slide.filename
    assert tuple(output.attrs['resolution']) == slide.dimensions
    level_dims = slide.level_dimensions[level][::-1]
    level_dims = (level_dims[0] // patch_size, level_dims[1] // patch_size)
    assert output[feature].shape == level_dims
    assert output[feature].attrs['extraction_level'] == level
    assert output[feature].attrs[
        'level_downsample'] == slide.level_downsamples[level]


@pytest.mark.parametrize('classifier', ['fixed-batch'])
@pytest.mark.parametrize(
    'model',
    ['slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin'])
@pytest.mark.parametrize('chunk', [None, 256])
def test_classify(classifier, tmp_path, model, chunk):
    label = 'tissue'
    path = str(tmp_path)
    cmd = [
        'classify.py', classifier, '-L', label, '-m', model, '-l', '2', '-o',
        path, input_
    ]
    if chunk:
        cmd += ['--chunk-size', str(chunk)]
    subprocess.check_call(cmd)
    logger.info('running cmd %s', ' '.join(cmd))
    output_path = os.path.join(path, f'{input_basename}.zarr')
    slide, output = get_input_output(output_path)

    _test_output(label, output, slide, 2)
    assert output[label].dtype == 'uint8'


@pytest.mark.parametrize('classifier', ['fixed-batch'])
@pytest.mark.parametrize(
    'model',
    ['slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin'])
def test_classifies_with_no_round(classifier, tmp_path, model):
    path = str(tmp_path)
    label = 'tissue'
    cmd = [
        'classify.py', classifier, '-L', label, '-m', model, '--no-round',
        '-l', '2', '-o', path, input_
    ]
    subprocess.check_call(cmd)
    logger.info('running cmd %s', ' '.join(cmd))
    output_path = os.path.join(path, f'{input_basename}.zarr')
    slide, output = get_input_output(output_path)

    #  assert output[label].dtype == 'float32'
    assert (np.array(output[label]) <= 1).all()


@pytest.mark.skip(reason="to be updated")
class TestSerialPatchClassifier:
    model = 'tests/models/all_one_by_patch.pkl'
    cmd = 'serial'
    feature = 'tumor'

    def test_classifies_with_default_args(self, tmp_path):
        path = str(tmp_path)
        level = 1
        cmd = [
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '-l',
            str(level), input_, '-o', path
        ]
        subprocess.check_call(cmd)
        logger.info('running cmd %s', ' '.join(cmd))
        output_path = os.path.join(path, f'{input_basename}.zarr')
        slide, output = get_input_output(output_path)

        assert output.attrs['filename'] == slide.filename
        assert tuple(output.attrs['resolution']) == slide.dimensions
        print(slide.level_dimensions[level])
        assert output[self.feature].shape == tuple([
            slide.level_dimensions[level][::-1][i] // (128, 128)[i]
            for i in range(2)
        ])
        assert output[self.feature].attrs['extraction_level'] == level
        assert output[self.feature].attrs[
            'level_downsample'] == slide.level_downsamples[level]
        assert output[self.feature].dtype == 'uint8'

    def test_classifies_with_no_round(self, tmp_path):
        path = str(tmp_path)
        level = 1
        cmd = [
            'classify.py', self.cmd, '-f', self.feature, '-m', self.model,
            '-l',
            str(level), '--no-round', input_, '-o', path
        ]
        subprocess.check_call(cmd)
        logger.info('running cmd %s', ' '.join(cmd))
        output_path = os.path.join(path, f'{input_basename}.zarr')
        slide, output = get_input_output(output_path)

        assert output.attrs['filename'] == slide.filename
        assert tuple(output.attrs['resolution']) == slide.dimensions
        assert output[self.feature].shape == tuple([
            slide.level_dimensions[level][::-1][i] // (128, 128)[i]
            for i in range(2)
        ])
        assert output[self.feature].attrs['extraction_level'] == level
        assert output[self.feature].attrs[
            'level_downsample'] == slide.level_downsamples[level]
        assert output[self.feature].dtype == 'float32'
        assert (np.array(output[self.feature]) <= 1).all()


@pytest.mark.skip('patch size too large, TBF')
@pytest.mark.parametrize('classifier', ['fixed-batch'])
@pytest.mark.parametrize('storage', ['zarr', 'zip'])
def test_classifies_with_filter(classifier, slide_with_mask, tmp_path,
                                model_all_ones_path, tmpdir, storage):
    path = f'{tmp_path}.{storage}'
    slide = slide_with_mask(np.ones)
    condition = 'mask>2'
    REGISTRY[storage].dump(slide, path, 'mask')

    cmd = [
        'classify.py',
        classifier,
        '-l',
        '0',
        '-L',
        'test',
        '-m',
        model_all_ones_path,
        '-o',
        str(tmpdir),
        '-F',
        f'"{condition}"',
        '--filter-slide',
        path,
        slide.filename,
    ]
    print(' '.join(cmd))
    subprocess.check_call(cmd)


@pytest.mark.skip(reason="to be updated")
class TestParallelPatchClassifier(TestSerialPatchClassifier):
    cmd = 'parallel'


@pytest.mark.skip(reason="patch prediction without filtering not supported")
@pytest.mark.parametrize('classifier', ['fixed-batch'])
@pytest.mark.parametrize('model',
                         ['slaid/resources/models/tumor_model-level_1.bin'])
def test_classifies_patches(classifier, tmp_path, model):
    label = 'tumor'
    path = str(tmp_path)
    cmd = [
        'classify.py', classifier, '-L', label, '-m', model, '-l', '1', '-o',
        path, input_
    ]
    subprocess.check_call(cmd)
    logger.info('running cmd %s', ' '.join(cmd))
    output_path = os.path.join(path, f'{input_basename}.zarr')
    slide, output = get_input_output(output_path)

    _test_output(label, output, slide, 1, 256)
    assert output[label].dtype == 'uint8'


if __name__ == '__main__':
    unittest.main()
