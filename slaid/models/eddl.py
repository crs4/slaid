import logging
import os
from abc import ABC, abstractstaticmethod
from typing import List

import numpy as np
import pyeddl.eddl as eddl
import stringcase
from pyeddl.tensor import Tensor

from slaid.commons.base import ImageInfo
from slaid.models import Factory as BaseFactory
from slaid.models import Model as BaseModel

logger = logging.getLogger('eddl-models')
fh = logging.FileHandler('/tmp/eddl.log')
logger.addHandler(fh)


class Factory(BaseFactory):

    def __init__(self, filename: str, gpu: List[int]):
        super().__init__(filename)
        self._gpu = gpu

    def get_model(self):
        basename = os.path.basename(self._filename)
        cls_name = basename.split('-')[0]
        cls_name = stringcase.capitalcase(stringcase.camelcase(cls_name))
        return globals()[cls_name](self._filename, self._gpu)


class Model(BaseModel, ABC):
    patch_size = None
    image_info = ImageInfo(ImageInfo.COLORTYPE.BGR, ImageInfo.COORD.YX,
                           ImageInfo.CHANNEL.FIRST)
    normalization_factor = 1
    index_prediction = 1

    def __init__(self, weight_filename, gpu: List = None):
        self._weight_filename = weight_filename
        self._gpu = gpu
        self._model_ = None

    @property
    def weight_filename(self):
        return self._weight_filename

    def _get_model(self):
        if self._model_ is None:
            self._create_model()
        return self._model_

    def _set_model(self, value):
        self._model_ = value

    _model = property(_get_model, _set_model)

    def __str__(self):
        return self._weight_filename

    def _set_gpu(self, value: List):
        self._gpu = value
        self._model = None

    def _get_gpu(self) -> List:
        return self._gpu

    gpu = property(_get_gpu, _set_gpu)

    def _create_model(self):
        net = self._create_net()
        eddl.build(
            net, eddl.rmsprop(0.00001), ["soft_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_GPU(self.gpu, mem="low_mem")
            if self.gpu else eddl.CS_CPU())
        eddl.load(net, self._weight_filename, "bin")
        self._model_ = net

    @abstractstaticmethod
    def _create_net():
        pass

    def predict(self, array: np.ndarray) -> np.ndarray:
        predictions = self._predict(array)
        temp_mask = []
        for prob_T in predictions:
            output_np = prob_T.getdata()
            temp_mask.append(output_np[:, self.index_prediction])

        flat_mask = np.vstack(temp_mask).flatten()
        return flat_mask

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        tensor = Tensor.fromarray(array / self.normalization_factor)
        prediction = eddl.predict(self._model, [tensor])
        return prediction

    def __getstate__(self):
        return dict(weight_filename=self._weight_filename, gpu=self._gpu)

    def __setstate__(self, state):
        self.__init__(**state)


class TissueModel(Model):
    index_prediction = 1
    image_info = ImageInfo(ImageInfo.COLORTYPE.RGB, ImageInfo.COORD.YX,
                           ImageInfo.CHANNEL.LAST)

    @staticmethod
    def _create_net():
        in_ = eddl.Input([3])
        layer = in_
        layer = eddl.ReLu(eddl.Dense(layer, 50))
        layer = eddl.ReLu(eddl.Dense(layer, 50))
        layer = eddl.ReLu(eddl.Dense(layer, 50))
        out = eddl.Softmax(eddl.Dense(layer, 2))
        net = eddl.Model([in_], [out])
        return net


class TumorModel(Model):
    patch_size = (256, 256)
    normalization_factor = 255
    index_prediction = 1

    @staticmethod
    def _create_net():
        in_size = [256, 256]
        num_classes = 2
        in_ = eddl.Input([3, in_size[0], in_size[1]])
        out = TumorModel._create_VGG16(in_, num_classes)
        net = eddl.Model([in_], [out])
        return net

    @staticmethod
    def _create_VGG16(in_layer, num_classes, seed=1234, init=eddl.HeNormal):
        x = in_layer
        x = eddl.ReLu(init(eddl.Conv(x, 64, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 64, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
        x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
        x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
        x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
        x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)),
                         [2, 2], [2, 2])
        x = eddl.Reshape(x, [-1])
        x = eddl.ReLu(init(eddl.Dense(x, 256), seed))
        x = eddl.Softmax(eddl.Dense(x, num_classes))
        return x


def load_model(weight_filename: str, gpu: List[int] = None) -> Model:
    basename = os.path.basename(weight_filename)
    cls_name = basename.split('-')[0]
    cls_name = stringcase.capitalcase(stringcase.camelcase(cls_name))
    return globals()[cls_name](weight_filename, gpu)
