from typing import Tuple

import numpy as np
from openslide import open_slide
from PIL import Image as PIL_Image

from slaid.commons.base import Image as BaseImage, ImageInfo
from slaid.commons.base import BasicSlide as BaseSlide


class Image(BaseImage):
    IMAGE_INFO = ImageInfo.create("rgb", "yx", "last")

    def __init__(self, image: PIL_Image):
        self._image = image

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._image.size)

    def to_array(self, image_info: ImageInfo = None) -> np.ndarray:
        array = np.array(self._image)  # yxc, RGB
        array = array[:, :, :3]
        if image_info is not None:
            array = self.IMAGE_INFO.convert(array, image_info)
        return array


class BasicSlide(BaseSlide):
    IMAGE_INFO = Image.IMAGE_INFO

    def __init__(self, filename: str):
        super().__init__(filename)
        self._slide = open_slide(filename)  # not serializable...
        self._dimensions = self._slide.dimensions
        self._level_dimensions = self._slide.level_dimensions
        self._level_downsamples = self._slide.level_downsamples

    def __eq__(self, other):
        return self._filename == other.filename and self.masks == other.masks

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._dimensions

    @property
    def filename(self):
        return self._filename

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> BaseImage:
        return Image(self._slide.read_region(location, level, size))

    def get_best_level_for_downsample(self, downsample: int):
        return self._slide.get_best_level_for_downsample(downsample)

    @property
    def level_dimensions(self):
        return self._level_dimensions

    @property
    def level_downsamples(self):
        return self._level_downsamples
