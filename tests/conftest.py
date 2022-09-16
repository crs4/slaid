import os
import shutil
from datetime import datetime as dt

import numpy as np
import pytest

from slaid.classifiers.base import BasicClassifier
from slaid.commons import ImageInfo, Mask
from slaid.commons.base import Slide
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.commons.factory import MetaSlideFactory
from slaid.models.factory import Factory
from slaid.runners import DirCache
from tests.commons import DummyModel, EddlGreenPatchModel, GreenModel


@pytest.fixture
def slide_with_mask():

    def _slide_with_mask(create_array_func):
        slide = EcvlSlide("tests/data/patch.tif")
        array = create_array_func(slide.dimensions[::-1])
        slide.masks["mask"] = Mask(
            array,
            0,
            1,
            slide.level_dimensions,
            dt.now(),
            False,
            model="model",
            tile_size=10,
        )
        return slide

    return _slide_with_mask


@pytest.fixture
def array(request):
    return np.ones((10, 10))


@pytest.fixture
def slide_path():
    return 'tests/data/PH10023-1.thumb.tif'


@pytest.fixture
def model_all_ones_path():
    return 'tests/models/all_one_by_patch.pkl'


@pytest.fixture
def patch_path(request):
    return 'tests/data/patch.tif'


@pytest.fixture(params=[EcvlSlide])
def slide_reader(request):
    return request.param


@pytest.fixture
def patch_tissue_mask(request):
    return np.load('tests/data/tissue_mask_prob.npy')


@pytest.fixture
def tumor_model(model_filename):
    return Factory(model_filename, cls_name='TumorModel').get_model()


@pytest.fixture
def tissue_model(model_filename):
    return Factory(model_filename, 'eddl', gpu=None,
                   cls_name='TissueModel').get_model()


@pytest.fixture
def slide_array(cls):
    return cls(
        np.arange(16 * 3).reshape(3, 4, 4),
        ImageInfo.create('bgr', 'yx', 'first'))


@pytest.fixture
def slide(slide_cls, slide_path, args):
    return MetaSlideFactory().get_factory(slide_cls, slide_path,
                                          *args).get_slide()


@pytest.fixture
def green_slide(
    basic_slide_cls,
    image_info,
    tile_size,
    slide_cls=Slide,
):
    slide_path = 'tests/data/test.tif'
    return slide_cls(
        SlideStore(basic_slide_cls(slide_path), tilesize=tile_size),
        image_info)


def green_classifier(classifier_cls):
    model = GreenModel()
    classifier = classifier_cls(model, 'tissue')
    return classifier


@pytest.fixture
def dummy_classifier(classifier_cls):
    model = DummyModel(np.zeros)
    return classifier_cls(model, 'tissue')


@pytest.fixture
def classifier(classifier_cls, model, feature='test'):
    return classifier_cls(model, feature)


@pytest.fixture
def green_slide_and_classifier(backend, slide_cls, image_info):

    slide_path = 'tests/data/test.tif'
    if backend == 'basic':
        return Slide(SlideStore(slide_cls(slide_path)),
                     image_info), BasicClassifier(GreenModel(), 'tissue')
    else:
        return NotImplementedError(backend)


@pytest.fixture
def green_slide_and_patch_classifier(backend, slide_cls, image_info):

    slide_path = 'tests/data/test.tif'
    if backend == 'basic':
        return Slide(SlideStore(slide_cls(slide_path)),
                     image_info), BasicClassifier(
                         EddlGreenPatchModel(patch_size=(10, 10)), 'tissue')
    else:
        return NotImplementedError(backend)


@pytest.fixture
def mask():
    return Mask(np.arange(9).reshape((3, 3)), 0, 1, [(3, 3)])


@pytest.fixture
def patch_array():
    patch_array = np.load(open('tests/data/patch.npy', 'rb'))
    return patch_array


@pytest.fixture
def onnx_path(tmp_path):
    dest = os.path.join(tmp_path, 'model.onnx')
    shutil.copy('slaid/resources/models/tissue_model-eddl-1.1.onnx', dest)
    return dest


@pytest.fixture
def cache(tmp_path):
    return DirCache(os.path.join(tmp_path, 'cache'))
