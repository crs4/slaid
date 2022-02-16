import unittest

import numpy as np
import pytest

from slaid.classifiers.fixed_batch import (FilteredPatchClassifier,
                                           FilteredPixelClassifier,
                                           PixelClassifier)
from slaid.commons import Mask
from slaid.commons.base import Filter, Slide
from slaid.commons.dask import init_client
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.commons.openslide import BasicSlide as OpenSlide
from slaid.models.eddl import TissueModel, TumorModel
from slaid.writers.zarr import ZarrStorage
from tests.commons import EddlGreenPatchModel, GreenModel


@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("slide_cls,args", [(Slide, (EcvlSlide, )),
                                            (Slide, (OpenSlide, ))])
@pytest.mark.parametrize("model", [GreenModel()])
@pytest.mark.parametrize("chunk_size", [None, 11, 100])
@pytest.mark.parametrize("slide_path", ["tests/data/test.tif"])
@pytest.mark.parametrize("classifier_cls", [PixelClassifier])
def test_classify_slide(slide, classifier_cls, model, level, chunk_size):
    green_slide = slide
    classifier = classifier_cls(model, "test", chunk_size=chunk_size)
    mask = classifier.classify(green_slide, level=level)

    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    green_zone = int(300 // green_slide.level_downsamples[level])
    assert (mask.array[:green_zone, :] == 100).all()
    assert (mask.array[green_zone:, :] == 0).all()


@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("slide_cls,args", [(Slide, (EcvlSlide, )),
                                            (Slide, (OpenSlide, ))])
@pytest.mark.parametrize("classifier_cls", [FilteredPixelClassifier])
@pytest.mark.parametrize("slide_path", ["tests/data/test.tif"])
@pytest.mark.parametrize("model", [GreenModel()])
def test_classify_with_filter(slide, classifier_cls, level, model):
    green_slide = slide
    filter_level = 2
    filter_downsample = green_slide.level_downsamples[filter_level]
    filter_array = np.zeros(green_slide.level_dimensions[filter_level][::-1])
    ones_row = 10
    ones_col = 10
    filter_array[:ones_row, :ones_col] = 1
    filter_mask = Mask(filter_array, filter_level, filter_downsample,
                       green_slide.level_dimensions)
    classifier = classifier_cls(model, "test", filter_mask >= 1)
    mask = classifier.classify(green_slide, level=level)

    ones_row = round(ones_row * filter_downsample //
                     green_slide.level_downsamples[level])
    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    assert (mask.array[:ones_row, :ones_col] == 100).all()
    assert (mask.array[ones_row:, :ones_col] == 0).all()


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("classifier_cls", [FilteredPixelClassifier])
@pytest.mark.parametrize("slide_cls,args", [(Slide, (EcvlSlide, )),
                                            (Slide, (OpenSlide, ))])
@pytest.mark.parametrize("slide_path", ["tests/data/test.tif"])
@pytest.mark.parametrize("model", [GreenModel()])
def test_classify_with_zeros_as_filter(slide, classifier_cls, level, model):
    green_slide = slide
    filter_level = 2
    filter_downsample = green_slide.level_downsamples[filter_level]
    filter_array = np.zeros(green_slide.level_dimensions[filter_level][::-1])
    filter_mask = Mask(filter_array, filter_level, filter_downsample,
                       green_slide.level_dimensions)

    classifier = classifier_cls(model, "test", filter_mask >= 1)
    mask = classifier.classify(green_slide, level=level)

    assert mask.array.shape == green_slide.level_dimensions[level][::-1]
    assert (mask.array[:, :] == 0).all()


@pytest.mark.parametrize("classifier_cls", [FilteredPatchClassifier])
@pytest.mark.parametrize("slide_cls,args", [(Slide, (EcvlSlide, )),
                                            (Slide, (OpenSlide, ))])
@pytest.mark.parametrize("slide_path", ["tests/data/test.tif"])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("array_factory", [ZarrStorage])
@pytest.mark.parametrize("model", [EddlGreenPatchModel((50, 50))])
def test_classify_slide_by_patches_with_filter_all_zeros(
        classifier_cls, slide, model, level, array_factory, tmp_path):
    green_slide = slide
    filter_array = np.zeros(
        (
            green_slide.level_dimensions[level][1] // model.patch_size[0],
            green_slide.level_dimensions[level][0] // model.patch_size[1],
        ),
        dtype="bool",
    )

    classifier = classifier_cls(model,
                                "test",
                                Filter(None, filter_array),
                                array_factory=array_factory(
                                    'test', store=f'{tmp_path}.zarr'))
    mask = classifier.classify(green_slide, level=level)
    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (
        dims[0] // model.patch_size[0],
        dims[1] // model.patch_size[1],
    )
    assert (np.array(mask.array) == 0).all()


@pytest.mark.parametrize("classifier_cls", [FilteredPatchClassifier])
@pytest.mark.parametrize("slide_cls,args", [(Slide, (EcvlSlide, )),
                                            (Slide, (OpenSlide, ))])
@pytest.mark.parametrize("slide_path", ["tests/data/test.tif"])
@pytest.mark.parametrize("level", [0])
@pytest.mark.parametrize("array_factory", [ZarrStorage])
@pytest.mark.parametrize("model", [EddlGreenPatchModel((50, 50))])
def test_classify_slide_by_patches_with_filter(classifier_cls, slide, level,
                                               model, array_factory, tmp_path):
    green_slide = slide
    filter_array = np.zeros(
        (
            green_slide.level_dimensions[level][1] // model.patch_size[0],
            green_slide.level_dimensions[level][0] // model.patch_size[1],
        ),
        dtype="bool",
    )
    filter_array[1, 1] = True
    classifier = classifier_cls(model,
                                "test",
                                Filter(None, filter_array),
                                array_factory=array_factory(
                                    'test', f'{tmp_path}.zarr'))
    mask = classifier.classify(green_slide, level=level)

    dims = green_slide.level_dimensions[level][::-1]
    assert mask.array.shape == (
        dims[0] // classifier.model.patch_size[0],
        dims[1] // classifier.model.patch_size[1],
    )
    assert (mask.array[0, :] == 0).all()
    assert (mask.array[1, 0] == 0).all()
    assert (mask.array[1, 1] == 100).all()
    assert (mask.array[1, 2:] == 0).all()
    assert (mask.array[2:, :] == 0).all()


@pytest.mark.parametrize("classifier_cls", [FilteredPatchClassifier])
@pytest.mark.parametrize("slide_path", ["tests/data/patch.tif"])
@pytest.mark.parametrize("slide_cls,args", [(Slide, (EcvlSlide, )),
                                            (Slide, (OpenSlide, ))])
@pytest.mark.parametrize(
    "model",
    [
        TumorModel(
            "slaid/resources/models/promort_vgg16_weights_ep_9_vacc_0.85.bin")
    ],
)
@pytest.mark.parametrize("array_factory", [ZarrStorage])
def test_classifies_tumor(slide, classifier_cls, model, array_factory,
                          tmp_path):
    classifier = classifier_cls(model,
                                "test",
                                Filter(None, np.ones((1, 1), dtype="bool")),
                                array_factory=array_factory(
                                    'tumor', f'{tmp_path}.zarr'))
    mask = classifier.classify(slide, level=0, round_to_0_100=False)

    print(mask.array.shape, type(mask.array))
    assert round(float(mask.array[0]), 4) == round(1 - 0.11082522, 4)


@pytest.mark.parametrize("slide_path", ["tests/data/patch.tif"])
@pytest.mark.parametrize("slide_cls,args", [(Slide, (EcvlSlide, )),
                                            (Slide, (OpenSlide, ))])
def test_classifies_tissue(slide, patch_tissue_mask):
    model = TissueModel(
        "slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin")
    classifier = PixelClassifier(model, "tissue")
    mask = classifier.classify(slide, level=0, round_to_0_100=False)
    assert (mask.array == patch_tissue_mask).all()

    downsample = 8
    filter_array = np.zeros(
        (
            patch_tissue_mask.shape[0] // downsample,
            patch_tissue_mask.shape[1] // downsample,
        ),
        dtype="bool",
    )
    filter_array[:2, :2] = 1

    filter_classifier = FilteredPixelClassifier(model, "tissue",
                                                Filter(None, filter_array))
    mask = filter_classifier.classify(
        slide,
        level=0,
        round_to_0_100=False,
    )

    assert (mask.array[:16, :16] == patch_tissue_mask[:16, :16]).all()


if __name__ == "__main__":
    init_client()
    unittest.main()
