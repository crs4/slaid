# NAPARI LAZY OPENSLIDE
#  Copyright (c) 2020, Trevor Manz
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of napari-lazy-openslide nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
from typing import List, Tuple

import numpy as np
from pyecvl.ecvl import Image as EcvlImage
from pyecvl.ecvl import OpenSlideImage

from slaid.commons import Image as BaseImage
from slaid.commons import ImageInfo

import slaid.commons.base as base

logger = logging.getLogger("ecvl")


class Image(BaseImage):
    IMAGE_INFO = ImageInfo.create("rgb", "yx", "first")

    def __init__(self, image: EcvlImage):
        self._image = image

    def to_array(self, image_info: ImageInfo = None):
        # FIXME
        array = np.array(self._image)
        array = array.transpose(0, 2, 1)

        if image_info is not None:
            array = self.IMAGE_INFO.convert(array, image_info)
        return array

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._image.dims_


class BasicSlide(base.BasicSlide):
    IMAGE_INFO = Image.IMAGE_INFO

    def __init__(self, filename: str):
        super().__init__(filename)
        self._slide = OpenSlideImage(filename)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return tuple(self._slide.GetLevelsDimensions()[0])

    def read_region(
        self, location: Tuple[int, int], level, size: Tuple[int, int]
    ) -> Image:
        return Image(self._slide.ReadRegion(level, location + size))

    def get_best_level_for_downsample(self, downsample: int):
        return self._slide.GetBestLevelForDownsample(downsample)

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        return [tuple(d) for d in self._slide.GetLevelsDimensions()]

    @property
    def level_downsamples(self):
        return self._slide.GetLevelDownsamples()


def load(filename: str):
    slide = BasicSlide(filename)
    return slide
