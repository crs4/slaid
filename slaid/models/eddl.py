import logging
import os
from abc import ABC
from dataclasses import dataclass
from typing import List

import numpy as np
import onnx
import pyeddl.eddl as eddl
import stringcase
from pyeddl.tensor import Tensor

from slaid.commons.base import ImageInfo
from slaid.models import Factory as BaseFactory
from slaid.models import Model as BaseModel

logger = logging.getLogger('eddl-models')
fh = logging.FileHandler('/tmp/eddl.log')
logger.addHandler(fh)


class Model(BaseModel, ABC):
    patch_size = None
    default_image_info = ImageInfo(
        ImageInfo.ColorType.BGR,
        ImageInfo.Coord.YX,
        ImageInfo.Channel.FIRST,
    )
    index_prediction = 1

    def __init__(self,
                 net: eddl.Model,
                 weight_filename: str = None,
                 gpu: List = None,
                 image_info: ImageInfo = None,
                 name: str = None):
        self._net = net
        self._weight_filename = weight_filename
        self._gpu = gpu
        self.image_info = image_info or self.default_image_info
        super().__init__(name)

    @property
    def weight_filename(self):
        return self._weight_filename

    def __str__(self):
        return str(self._weight_filename)

    @property
    def gpu(self) -> List:
        return self._gpu

    @property
    def net(self):
        return self._net

    def predict(self, array: np.ndarray) -> np.ndarray:
        predictions = self._predict(array)
        temp_mask = []
        for prob_T in predictions:
            output_np = prob_T.getdata()
            temp_mask.append(output_np[:, self.index_prediction])

        flat_mask = np.vstack(temp_mask).flatten()
        return flat_mask

    def _predict(self, array: np.ndarray) -> List[Tensor]:
        tensor = Tensor.fromarray(array)
        prediction = eddl.predict(self._net, [tensor])
        return prediction


class TissueModel(Model):
    index_prediction = 1
    default_image_info = ImageInfo(ImageInfo.ColorType.RGB, ImageInfo.Coord.YX,
                                   ImageInfo.Channel.LAST,
                                   ImageInfo.Range._0_255)

    @staticmethod
    def create_net():
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
    index_prediction = 1
    default_image_info = ImageInfo(ImageInfo.ColorType.BGR, ImageInfo.Coord.YX,
                                   ImageInfo.Channel.FIRST,
                                   ImageInfo.Range._0_1)

    @staticmethod
    def create_net():
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


def to_onnx(model: Model, filename: str):
    eddl.save_net_to_onnx_file(model.net, filename)


@dataclass
class Factory(BaseFactory):
    filename: str
    cls_name: str = None
    gpu: List[int] = None
    learn_rate = 1e-5
    list_of_losses: List[str] = None
    list_of_metrics: List[str] = None

    def __post_init__(self):
        self.list_of_losses = self.list_of_losses or ["soft_cross_entropy"]
        self.list_of_metrics = self.list_of_metrics or ["categorical_accuracy"]

    def get_model(self):
        cls_name = self._get_cls_name()
        cls = globals()[cls_name]
        net = cls.create_net()
        self._build_net(net)
        eddl.load(net, self.filename, "bin")
        return globals()[cls_name](net)

    def _build_net(self, net):
        eddl.build(net,
                   eddl.rmsprop(self.learn_rate),
                   self.list_of_losses,
                   self.list_of_metrics,
                   eddl.CS_GPU(self.gpu, mem="low_mem")
                   if self.gpu else eddl.CS_CPU(),
                   init_weights=False)

    def _get_cls_name(self):
        if self.cls_name:
            cls_name = self.cls_name
        else:
            basename = os.path.basename(self.filename)
            cls_name = basename.split('-')[0]
            cls_name = stringcase.capitalcase(stringcase.camelcase(cls_name))
        return cls_name


@dataclass
class OnnxFactory(Factory):

    def get_model(self):
        net = eddl.import_net_from_onnx_file(self.filename)
        self._build_net(net)

        cls_name = self._get_cls_name()
        cls = globals()[cls_name]

        image_info = self._update_image_info(cls.default_image_info)
        return cls(net,
                   image_info=image_info,
                   name=os.path.basename(self.filename))

    def _update_image_info(self, image_info: ImageInfo) -> ImageInfo:

        image_info = ImageInfo(color_type=image_info.color_type,
                               coord=image_info.coord,
                               channel=image_info.channel,
                               pixel_range=image_info.pixel_range)
        onnx_model = onnx.load(self.filename)
        for prop in onnx_model.metadata_props:
            if prop.key == "Image.BitmapPixelFormat":
                color_type = prop.value[:3].lower()
                image_info.color_type = ImageInfo.ColorType(color_type)
            if prop.key == "Image.NominalPixelRange":
                pixel_range = prop.value.split('_', 1)[1]
                image_info.pixel_range = ImageInfo.Range(pixel_range)
        return image_info
