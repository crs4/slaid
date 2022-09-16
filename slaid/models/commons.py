from typing import List, Tuple

import numpy as np
from pyeddl.tensor import Tensor
import pickle

from slaid.commons import Slide
from slaid.commons.base import Image, ImageInfo
from slaid.models.base import Factory as BaseFactory
from slaid.models.base import Model as BaseModel
from slaid.models.eddl import Model as EddlModel


class Factory(BaseFactory):
    def get_model(self):
        with open(self._filename, "rb") as f:
            return pickle.load(f)


class GreenModel(BaseModel):
    image_info = ImageInfo.create("rgb", "yx", "last")

    def __init__(self, patch_size=None):
        self.patch_size = patch_size

    def predict(self, array: np.array) -> np.array:
        return array[:, 1] / 255


class EddlGreenModel(EddlModel):
    def __init__(self, patch_size=None):
        self.patch_size = patch_size

    @staticmethod
    def _create_net():
        pass

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        array = array[:, 1] / 255
        array = np.array([(0, el) for el in array])
        tensor = Tensor.fromarray(array)
        return [tensor]


class EddlGreenPatchModel(EddlModel):
    def __init__(self, patch_size=(256, 256)):
        self.patch_size = patch_size

    @staticmethod
    def _create_net():
        pass

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        prob_green = [
            np.sum(p[1, :, :] / 255) / (p.shape[1] * p.shape[2]) for p in array
        ]
        array = np.array([(0, prob) for prob in prob_green])
        tensor = Tensor.fromarray(array)
        return [tensor]


class BaseDummyModel(BaseModel):
    def __init__(self, patch_size=None):
        self.patch_size = patch_size
        self.array_predicted = []

    def predict(self, array):
        self.array_predicted.append(array)
        return self._predict(array)

    def _predict(self, array):
        raise NotImplementedError


class DummyModel(BaseDummyModel):
    image_info = ImageInfo.create("bgr", "yx", "last")

    def __init__(self, func, patch_size=None):
        self.func = func
        super().__init__(patch_size)

    def _predict(self, array):
        return self.func(array.shape[0])


class DummySlide(Slide):
    def __init__(
        self,
        level_dimensions: List[Tuple[int, int]],
        level_downsamples: List[Tuple[int, int]],
        ID=None,
    ):
        self._level_dimensions = level_dimensions
        self._level_downsamples = level_downsamples
        self._ID = ID

    @property
    def ID(self):
        return self._ID

    @property
    def dimensions(self):
        return self.level_dimensions[0]

    @property
    def level_dimensions(self):
        return self._level_dimensions

    @property
    def level_downsamples(self):
        return self._level_downsamples

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> Image:
        raise NotImplementedError()

    def get_best_level_for_downsample(self, downsample: int):
        raise NotImplementedError()


class AllOneModel(BaseDummyModel):
    def _predict(self, array):
        return np.ones(array.shape[0], dtype=np.uint8)

    def get_best_level_for_downsample(self, downsample: int):
        raise NotImplementedError()
