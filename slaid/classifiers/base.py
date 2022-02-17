import abc
import logging
from datetime import datetime as dt
from typing import Tuple

import numpy as np
from progress.bar import Bar

from slaid.commons import Filter, Mask, NapariSlide
from slaid.commons.base import ImageInfo
from slaid.models import Model
from slaid.commons.base import ArrayFactory, NumpyArrayFactory

logger = logging.getLogger('classify')
fh = logging.FileHandler('/tmp/base-classifier.log')
logger.addHandler(fh)


class InvalidChunkSize(Exception):
    pass


class Classifier(abc.ABC):
    MASK_CLASS = Mask

    def __init__(self,
                 model: "Model",
                 label: str,
                 array_factory: ArrayFactory = None):
        self.model = model
        self.label = label
        self.array_factory = array_factory or NumpyArrayFactory()

        try:
            self._patch_size = self.model.patch_size
            self._image_info = model.image_info
        except AttributeError as ex:
            logger.error(ex)
            raise ex
            self._patch_size = None
            self._image_info = ImageInfo(ImageInfo.COLORTYPE('rgb'),
                                         ImageInfo.COORD('yx'),
                                         ImageInfo.CHANNEL('last'))

    @abc.abstractmethod
    def classify(self,
                 slide: NapariSlide,
                 level,
                 threshold: float = None,
                 batch_size: int = 8,
                 round_to_0_100: bool = True) -> Mask:
        pass

    def _get_mask(self,
                  slide,
                  array,
                  level,
                  downsample,
                  round_to_0_100,
                  tile_size=1):

        return self.MASK_CLASS(array,
                               level,
                               downsample,
                               slide.level_dimensions,
                               round_to_0_100,
                               model=str(self.model),
                               tile_size=tile_size)

    @staticmethod
    def _round_to_0_100(array: np.ndarray, round_: bool) -> np.ndarray:
        return (array * 100).astype('uint8') if round_ else array

    @staticmethod
    def _threshold(array: np.ndarray, threshold: float) -> np.ndarray:
        if threshold is not None:
            array[array >= threshold] = 1
            array[array < threshold] = 0
            return array.astype('uint8')
        return array

    def _predict(self, array):
        if array.size == 0:
            return np.empty((0, ))
        return self.model.predict(array)


class BasicClassifier(Classifier):

    def __init__(self,
                 model: "Model",
                 feature: str,
                 array_factory: ArrayFactory = None,
                 _filter: Filter = None,
                 chunk: Tuple[int, int] = None):
        super().__init__(model, feature, array_factory)
        self.chunk = chunk
        self._filter = _filter

    def set_filter(self, _filter: Filter):
        self._filter = _filter

    def classify(self,
                 slide: NapariSlide,
                 threshold: float = None,
                 level: int = 2,
                 round_to_0_100: bool = True) -> Mask:

        slide_array = slide[level]
        chunk = self.chunk or slide_array.size
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'

        if self._patch_size:
            predictions = self._classify_patches(slide_array, chunk, threshold,
                                                 round_to_0_100, dtype)
        else:
            predictions = self._classify_pixels(slide_array, chunk, threshold,
                                                round_to_0_100, dtype)
        return self._get_mask(slide, predictions, level,
                              slide.level_downsamples[level], round_to_0_100)

    def _classify_pixels(self, slide_array, chunk, threshold, round_to_0_100,
                         dtype):
        if self._filter:
            if (self._filter.array == 0).all():
                return self.array_factory.zeros(slide_array.size, dtype=dtype)

            self._filter.rescale(slide_array.size)
            _filter = self._filter.array
        else:
            _filter = np.ones(slide_array.size, dtype='bool')

        predictions = self.array_factory.empty((0, slide_array.size[1]),
                                               dtype=dtype)
        with Bar('Processing', max=_filter.shape[0] // chunk[0] or 1) as bar:
            for x in range(0, _filter.shape[0], chunk[0]):
                row = np.empty((min(chunk[0], _filter.shape[0] - x), 0),
                               dtype=dtype)
                for y in range(0, _filter.shape[1], chunk[1]):
                    filter_block = _filter[x:x + chunk[0], y:y + chunk[1]]

                    res = np.zeros(filter_block.shape, dtype='float32')
                    if (filter_block == True).any():
                        block = slide_array[x:x + chunk[0],
                                            y:y + chunk[1]].convert(
                                                self.model.image_info)
                        to_predict = block[filter_block]
                        prediction = self._predict(to_predict.array)
                        res[filter_block] = prediction
                    res = self._threshold(res, threshold)
                    res = self._round_to_0_100(res, round_to_0_100)
                    row = self._append(row, res, 1)
                predictions = self._append(predictions, row, 0)
                bar.next()

        return predictions


#

    @staticmethod
    def _append(array, values, axis):
        import zarr
        if isinstance(array, np.ndarray):
            return np.append(array, values, axis)
        elif isinstance(array, zarr.core.Array):
            array.append(values, axis)
            return array

    def _classify_patches(self, slide_array, chunk, threshold, round_to_0_100,
                          dtype):
        predictions = self.array_factory.empty(
            (0, slide_array.size[1] // self._patch_size[1]), dtype=dtype)
        _filter = self._filter if self._filter else np.ones(
            (slide_array.size[0] // self._patch_size[0],
             slide_array.size[1] // self._patch_size[1]),
            dtype='bool')
        for i in range(2):
            if chunk[i] % self._patch_size[i] > 0 and chunk[
                    i] < slide_array.size[i]:
                raise InvalidChunkSize(
                    f'Invalid chunk size {chunk}: should be a multiple of {self._patch_size}'
                )
        with Bar('Processing', max=slide_array.size[0] // chunk[0]
                 or 1) as progress_bar:
            for x in range(0, slide_array.size[0], chunk[0]):
                row_size = min(chunk[0], slide_array.size[0] - x)
                row_size = row_size - (row_size % self._patch_size[0])
                if not row_size:
                    break
                row = np.empty((row_size // self._patch_size[0], 0),
                               dtype=dtype)

                for y in range(0, slide_array.size[1], chunk[1]):
                    col_size = min(chunk[1], slide_array.size[1] - y)
                    col_size = col_size - (col_size % self._patch_size[1])
                    if not col_size:
                        break

                    filter_block = _filter[
                        x // self._patch_size[0]:(x + row_size) //
                        self._patch_size[0],
                        y // self._patch_size[1]:(y + col_size) //
                        self._patch_size[1]]
                    res = np.zeros(filter_block.shape, dtype='float32')

                    if (filter_block == True).any():
                        chunked_array = slide_array[x:x + row_size,
                                                    y:y + col_size].convert(
                                                        self.model.image_info)
                        patches, channel_first = chunked_array.get_blocks(
                            self._patch_size)
                        to_predict = patches[:, filter_block,
                                             ...] if channel_first else patches[
                                                 filter_block, :, ...]
                        to_predict = to_predict.reshape(
                            (to_predict.shape[0] * to_predict.shape[1], ) +
                            to_predict.shape[2:])
                        if to_predict.shape[0] > 0:
                            prediction = self._predict(to_predict)
                            res[filter_block] = prediction
                            res = self._threshold(res, threshold)
                            res = self._round_to_0_100(res, round_to_0_100)

                    row = self._append(row, res, 1)
                predictions = self._append(predictions, row, 0)
                progress_bar.next()
        return predictions

    def _get_slide_array(self, slide, level):
        return slide[level].convert(self.model.image_info)


def append_array(array, values, axis):
    import zarr
    if isinstance(array, np.ndarray):
        return np.append(array, values, axis)
    elif isinstance(array, zarr.core.Array):
        array.append(values, axis)
        return array
