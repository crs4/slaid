import logging
from dataclasses import dataclass
from datetime import datetime as dt
from functools import partial
from typing import Tuple, Union

import numpy as np

from slaid.classifiers.base import Classifier as BaseClassifier
from slaid.classifiers.base import append_array
from slaid.commons import Filter, Mask
from slaid.commons.base import ArrayFactory, ImageInfo, Slide
from slaid.models import Model

logger = logging.getLogger()


@dataclass
class BatchIterator:
    batch_size: int
    channel_first: bool

    def __post_init__(self):
        self._buffer: np.ndarray = np.empty((3,
                                             0) if self.channel_first else (0,
                                                                            3))

    @property
    def buffer(self):
        return self._buffer

    def iter(self) -> np.ndarray:
        max_index = self._buffer_size() - self._buffer_size() % self.batch_size
        for i in range(0, max_index, self.batch_size):
            batch = self._buffer[:, i:i + self.
                                 batch_size] if self.channel_first else self._buffer[
                                     i:i + self.batch_size, :]
            yield batch
        self._buffer = self._buffer[:,
                                    max_index:] if self.channel_first else self._buffer[
                                        max_index:, :]

    def append(self, array: np.ndarray):
        self._buffer = np.concatenate([self._buffer, array])

    def _buffer_size(self):
        return self._buffer.shape[
            1] if self.channel_first else self._buffer.shape[0]


class Classifier(BaseClassifier):

    def _predict_by_batch(self, batch_iterator: BatchIterator,
                          all_buffer: bool) -> np.ndarray:

        predictions = []
        for batch in batch_iterator.iter():
            predictions.append(self._predict(batch))

        if all_buffer:
            predictions.append(self._predict(batch_iterator.buffer))
        predictions = np.concatenate(predictions) if predictions else np.empty(
            (0, ))
        return predictions


class FilteredClassifier(Classifier):

    def __init__(self,
                 model: "Model",
                 feature: str,
                 _filter: Filter,
                 array_factory: ArrayFactory = None):
        super().__init__(model, feature, array_factory)
        self._filter = _filter


class PixelClassifier(Classifier):

    def __init__(self,
                 model: "Model",
                 feature: str,
                 array_factory: ArrayFactory = None,
                 chunk_size: int = None):
        super().__init__(model, feature, array_factory)
        self.chunk_size = chunk_size

    def classify(self,
                 slide: Slide,
                 level,
                 threshold: float = None,
                 batch_size: int = 8,
                 round_to_0_100: bool = True) -> Mask:

        slide_array = slide[level]
        row_size = self.chunk_size if self.chunk_size else slide_array.size[0]
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'

        res = self.array_factory.empty(slide_array.size, dtype=dtype)

        channel_first = self.model.image_info.CHANNEL == ImageInfo.CHANNEL.FIRST
        batch_iterator = BatchIterator(batch_size, channel_first)
        row_splitter = RowSplitter(slide_array.size[1])

        for row_idx in range(0, slide_array.size[0], row_size):
            row = slide_array[row_idx:row_idx + row_size, :].convert(
                self.model.image_info).array
            row = row.reshape(3, -1) if channel_first else row.reshape(-1, 3)

            batch_iterator.append(row)
            predictions = self._predict_by_batch(batch_iterator, False)
            row_splitter.append(predictions)
            self._set_rows(res, row_splitter, threshold, round_to_0_100)

        remaining_predictions = self._predict_by_batch(batch_iterator, True)
        row_splitter.append(remaining_predictions)
        self._set_rows(res, row_splitter, threshold, round_to_0_100)

        return self._get_mask(slide, res, level,
                              slide.level_downsamples[level], round_to_0_100)

    def _set_rows(self, array, row_splitter: "RowSplitter", threshold: float,
                  round_to_0_100: bool):
        try:
            row_index, rows = row_splitter.split()
        except RowSplitter.RowsNotFound:
            ...
        else:
            rows = self._threshold(rows, threshold)
            rows = self._round_to_0_100(rows, round_to_0_100)
            array[row_index:row_index + rows.shape[0], :] = rows


class FilteredPatchClassifier(FilteredClassifier):

    def classify(self,
                 slide: Slide,
                 level: int,
                 threshold: float = None,
                 batch_size: int = 8,
                 round_to_0_100: bool = True) -> Mask:
        if not self._patch_size:
            raise RuntimeError(f'invalid patch size {self._patch_size}')

        slide_array = slide[level]

        filter_array = self._filter.array
        patch_coords = np.argwhere(filter_array) * self._patch_size
        patch_coords = list(
            filter(partial(self._remove_borders, slide_array),
                   iter(patch_coords)))

        n_patches = len(patch_coords)
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
        res = self.array_factory.zeros(
            (slide_array.size[0] // self._patch_size[0],
             slide_array.size[1] // self._patch_size[1]),
            dtype=dtype)

        predictions = np.empty(n_patches)
        patches = []
        for patch_coord in patch_coords:
            x, y = patch_coord

            patch = slide_array[x:x + self._patch_size[0],
                                y:y + self._patch_size[1]].convert(
                                    self.model.image_info).array
            patches.append(patch)

        for index in range(0, n_patches, batch_size):

            to_predict = np.array(patches[index:index + batch_size])
            prediction = self._predict(to_predict)
            predictions[index:index + batch_size] = prediction

        predictions = self._threshold(predictions, threshold)
        predictions = self._round_to_0_100(predictions, round_to_0_100)
        res.set_mask_selection(filter_array, predictions)

        return self._get_mask(slide,
                              res,
                              level,
                              slide.level_downsamples[level],
                              round_to_0_100,
                              tile_size=self._patch_size[0])

    def _remove_borders(self, slide_array: np.ndarray,
                        coord: np.ndarray) -> bool:
        return coord[0] <= (slide_array.size[0] -
                            self._patch_size[0]) and coord[1] <= (
                                slide_array.size[1] - self._patch_size[1])


class FilteredPixelClassifier(FilteredClassifier):

    def classify(self,
                 slide: Slide,
                 level: int = 2,
                 threshold: float = None,
                 batch_size: int = 4096,
                 round_to_0_100: bool = True) -> Mask:
        if self._patch_size:
            raise RuntimeError(
                f'Not expecting patch size, found {self._patch_size}')

        slide_array = slide[level]
        filter_array = self._filter.array
        zoom_factor = (
            slide_array.size[0] // filter_array.shape[0],
            slide_array.size[1] // filter_array.shape[1],
        )
        tile_size = zoom_factor
        patch_coords = np.argwhere(filter_array) * tile_size
        n_patches = len(patch_coords)
        predictions = np.empty(n_patches)
        patches = []
        for patch_coord in patch_coords:
            x, y = patch_coord

            patch = slide_array[x:x + tile_size[0],
                                y:y + tile_size[1]].convert(
                                    self.model.image_info).array
            patches.append(patch.reshape(-1, 3))

        to_predict = np.concatenate(patches) if patches else np.empty(
            (0, 3), dtype='uint8')
        channel_first = (
            self.model.image_info.CHANNEL == ImageInfo.CHANNEL.FIRST)
        batch_iterator = BatchIterator(batch_size, channel_first)
        batch_iterator.append(to_predict)

        predictions = self._predict_by_batch(batch_iterator, True)

        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
        res = self.array_factory.zeros(slide_array.size, dtype=dtype)
        patch_area = tile_size[0] * tile_size[1]

        for i in range(n_patches):
            patch_coord = patch_coords[i]
            x, y = patch_coord
            i = i * patch_area
            patch = predictions[i:i + patch_area].reshape(tile_size)
            patch = self._threshold(patch, threshold)
            patch = self._round_to_0_100(patch, round_to_0_100)
            res[x:x + tile_size[0], y:y + tile_size[1]] = patch

        return self._get_mask(slide, res, level,
                              slide.level_downsamples[level], round_to_0_100)


class RowSplitter:

    def __init__(self, col_size: int):
        self._col_size = col_size
        self._buffer = np.empty((0), dtype='float32')
        self._row_index = 0

    class RowsNotFound(Exception):
        ...

    def append(self, data: np.ndarray):
        self._buffer = np.append(self._buffer, data)

    def split(self) -> Tuple[int, np.ndarray]:
        n_rows = self._buffer.size // self._col_size
        if n_rows:
            rows = np.array(self._buffer[:n_rows * self._col_size]).reshape(
                n_rows, self._col_size)
            self._buffer = self._buffer[n_rows * self._col_size:]
            row_index = self._row_index
            self._row_index += n_rows
            return (row_index, rows)
        else:
            raise RowSplitter.RowsNotFound()
