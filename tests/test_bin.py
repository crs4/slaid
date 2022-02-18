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

DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = '/tmp/test-slaid'
input_ = os.path.join(DIR, 'data/PH10023-1.thumb.tif')
input_basename = os.path.basename(input_)
input_basename_no_ext = 'PH10023-1.thumb'
slide = Slide(input_)
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


def get_input_output(output, slide_path=input_):
    slide = Slide(slide_path)
    zarr_group = zarr.open_group(output)
    return slide, zarr_group


def _test_output(feature, output, slide, level, tile_size=1):
    assert output.attrs['filename'] == slide.filename
    assert tuple(output.attrs['resolution']) == slide.dimensions
    level_dims = slide.level_dimensions[level][::-1]
    level_dims = (level_dims[0] // tile_size, level_dims[1] // tile_size)
    assert output[feature].shape == level_dims
    assert output[feature].attrs['extraction_level'] == level
    assert output[feature].attrs[
        'level_downsample'] == slide.level_downsamples[level]

    assert output[feature].attrs['tile_size'] == tile_size


@pytest.mark.parametrize('classifier', ['fixed-batch'])
@pytest.mark.parametrize(
    'model',
    ['slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin'])
@pytest.mark.parametrize('chunk', [None, 10])
@pytest.mark.parametrize('level', [2])
def test_classify(classifier, tmp_path, model, chunk, level):
    label = 'tissue'
    path = str(tmp_path)
    cmd = [
        'classify.py', classifier, '-L', label, '-m', model, '-l',
        str(level), '-o', path, input_
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


@pytest.mark.parametrize('classifier', ['fixed-batch'])
@pytest.mark.parametrize('storage', ['zip'])
@pytest.mark.parametrize('slide', ['tests/data/patch-8-level.tif'])
def test_classifies_with_filter(classifier, slide, storage, tmp_path):

    tissue_low_res = [
        'classify.py',
        classifier,
        '-l',
        '8',
        '-L',
        'tissue',
        '-m',
        'slaid/resources/models/tissue_model-eddl_2.bin',
        '-o',
        str(tmp_path),
        slide,
    ]
    tissue_high_res = [
        'classify.py',
        classifier,
        '-l',
        '3',
        '-L',
        'tissue-high-res',
        '-m',
        'slaid/resources/models/tissue_model-eddl_2.bin',
        '--filter',
        'tissue>0.8',
        '--filter-slide',
        os.path.join(str(tmp_path), f'{os.path.basename(slide)}.zarr'),
        '-o',
        str(tmp_path),
        slide,
    ]
    tumor = [
        'classify.py',
        classifier,
        '-l',
        '0',
        '-L',
        'tumor',
        '-m',
        'slaid/resources/models/tumor_model-level_1.bin',
        '--filter',
        'tissue>0.8',
        '--filter-slide',
        os.path.join(str(tmp_path), f'{os.path.basename(slide)}.zarr'),
        '-o',
        str(tmp_path),
        slide,
    ]

    subprocess.check_call(tissue_low_res)
    subprocess.check_call(tissue_high_res)
    print(' '.join(tumor))
    subprocess.check_call(tumor)


@pytest.mark.parametrize('classifier', ['fixed-batch'])
@pytest.mark.parametrize('model',
                         ['slaid/resources/models/tumor_model-level_1.bin'])
@pytest.mark.parametrize('slide', ['tests/data/patch-8-level.tif'])
def test_classifies_patches(slide, classifier, tmp_path, model):
    tissue_low_res = [
        'classify.py',
        classifier,
        '-l',
        '8',
        '-L',
        'tissue',
        '-m',
        'slaid/resources/models/tissue_model-eddl_2.bin',
        '-o',
        str(tmp_path),
        slide,
    ]

    label = 'tumor'
    path = str(tmp_path)
    cmd = [
        'classify.py',
        classifier,
        '-L',
        label,
        '-m',
        model,
        '-l',
        '0',
        '-o',
        path,
        '--filter',
        'tissue>0.8',
        '--filter-slide',
        os.path.join(str(tmp_path), f'{os.path.basename(slide)}.zarr'),
        slide,
    ]
    subprocess.check_call(tissue_low_res)
    print(' '.join(cmd))
    subprocess.check_call(cmd)
    logger.info('running cmd %s', ' '.join(cmd))
    output_path = os.path.join(path, f'{os.path.basename(slide)}.zarr')
    slide, output = get_input_output(output_path, slide)

    _test_output(label, output, slide, 0, 256)
    assert output[label].dtype == 'uint8'


@pytest.mark.skipif(not os.path.exists('/usr/include/cudnn.h'),
                    reason='CUDNN not available')
@pytest.mark.parametrize('classifier', ['fixed-batch'])
@pytest.mark.parametrize('mirax_slide',
                         ['tests/data/Mirax2-Fluorescence-2.mrxs'])
@pytest.mark.parametrize('chunk_size', [None, 100])
@pytest.mark.parametrize('batch_size', [None])
def test_real_case_classification(classifier, mirax_slide, chunk_size,
                                  batch_size, tmp_path):
    tissue_low_res = [
        'classify.py',
        classifier,
        '-l',
        '9',
        '-L',
        'tissue',
        '-m',
        'slaid/resources/models/tissue_model-eddl_2.bin',
        '--gpu',
        '0',
        '-o',
        str(tmp_path),
        '--no-round',
        mirax_slide,
    ]
    if chunk_size:
        tissue_low_res.extend(['--chunk-size', str(chunk_size)])

    if batch_size:
        tissue_low_res.extend(['--batch-size', str(batch_size)])
    subprocess.check_call(tissue_low_res)

    group = zarr.open(str(tmp_path / f'{os.path.basename(mirax_slide)}.zarr'))
    output = np.array(group['tissue'])
    expected = np.load(open('tests/data/mask_ar.npy', 'rb'))

    round_to_decimal = 4
    output = np.around(output, round_to_decimal)
    expected = np.around(expected, round_to_decimal)

    assert (output == expected).all()

    label = 'tumor'
    path = str(tmp_path)
    tumor = [
        'classify.py',
        classifier,
        '-L',
        label,
        '-m',
        'slaid/resources/models/tumor_model-level_1.bin',
        '-l',
        '1',
        '--gpu',
        '0',
        '-o',
        path,
        '--filter',
        'tissue>0.5',
        '--filter-slide',
        os.path.join(str(tmp_path), f'{os.path.basename(mirax_slide)}.zarr'),
        '--no-round',
        mirax_slide,
    ]

    if batch_size:
        tumor.extend(['--batch-size', str(batch_size)])

    subprocess.check_call(tumor)

    group = zarr.open(str(tmp_path / f'{os.path.basename(mirax_slide)}.zarr'))
    output = np.array(group[label])
    expected = np.load(open('tests/data/heat.npy', 'rb'))

    round_to_decimal = 5
    output = np.around(output, round_to_decimal)
    expected = np.around(expected, round_to_decimal)


if __name__ == '__main__':
    unittest.main()
