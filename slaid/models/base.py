import abc

import numpy as np

from slaid.commons import ImageInfo


class Model(abc.ABC):
    image_info = ImageInfo.create('bgr', 'yx', 'first')

    def __init__(self, name: str = None):
        self.name = name

    @abc.abstractmethod
    def predict(self, array: np.array) -> np.array:
        pass


class Factory(abc.ABC):

    def __init__(self, filename):
        self._filename = filename

    @abc.abstractmethod
    def get_model(self) -> Model:
        pass
