import logging
from datetime import datetime as dt

import dask.array as da
import numpy as np
from dask import delayed

from slaid.classifiers.base import BasicClassifier
from slaid.commons import BasicSlide, Filter, NapariSlide
from slaid.commons.dask import Mask
from slaid.models.base import Model

logger = logging.getLogger('dask')
logger.setLevel(logging.DEBUG)


class Classifier(BasicClassifier):
    MASK_CLASS = Mask

    def __init__(self, model: Model, feature: str, compute_mask: bool = False):
        super().__init__(model, feature)
        self.compute_mask = compute_mask

    def classify(self,
                 slide: NapariSlide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 round_to_0_100: bool = True,
                 chunk=None) -> Mask:
        logger.info('patch size %s', self._patch_size)
        if self._patch_size:
            predictions = self._classify_patches(slide, self._patch_size,
                                                 level, filter_, threshold,
                                                 round_to_0_100, chunk)
        else:
            predictions = self._classify_batches(slide, level, filter_, chunk)
        predictions = self._threshold(predictions, threshold)
        predictions = self._round_to_0_100(predictions, round_to_0_100)
        mask = self._get_mask(slide, predictions, level,
                              slide.level_downsamples[level], round_to_0_100)

        if self.compute_mask:
            mask.compute()
        return mask

    def _classify_batches(self, slide, level, filter_, chunk):
        slide_array = self._get_slide_array(slide, level)
        if filter_ is None:
            return self._classify_batches_no_filter(slide_array, chunk)

        else:
            return self._classify_batches_with_filter(slide, slide_array,
                                                      level, filter_, chunk)

    def _predict_batch(self, array, model):
        n_px = array.shape[0] * array.shape[1]
        p = model.predict(array.reshape((n_px, 3))).reshape(array.shape[:2])
        return p

    def _classify_batches_no_filter(self, slide_array, _chunk):
        prediction = slide_array.array.map_blocks(self._predict_batch,
                                                  delayed(self._model),
                                                  meta=np.array(
                                                      (), dtype='float32'),
                                                  drop_axis=2)
        return prediction

    def _classify_batches_with_filter(self, slide, slide_array, level, filter_,
                                      chunk):

        size = slide_array.size
        scale = (size[0] // filter_.array.shape[0],
                 size[1] // filter_.array.shape[1])

        filter_array = da.from_array(filter_.array, chunks=20)
        chunks = []
        for i, _chunk in enumerate(filter_array.chunks):
            chunks.append([c * scale[i] for c in _chunk])
        filter_array = da.map_blocks(
            lambda x, scale: x.repeat(scale[0], 0).repeat(scale[1], 1),
            filter_array,
            scale,
            meta=np.array([], dtype='float32'),
            chunks=chunks)

        predictions = da.map_blocks(self._predict_with_filter,
                                    filter_array,
                                    delayed(self.model),
                                    slide.filename,
                                    type(slide._slide),
                                    self.model.image_info,
                                    level,
                                    meta=np.array([], dtype='float32'))

        return predictions

    def _predict_with_filter(self,
                             filter_array,
                             model,
                             slide_filename,
                             slide_cls,
                             image_info,
                             level,
                             block_info=None):
        res = np.zeros(filter_array.shape, dtype='float32')
        if (filter_array == 0).all():
            return res
        loc = block_info[0]['array-location'][::-1]

        slide = slide_cls(slide_filename)
        data = slide.read_region((loc[0][0], loc[1][0]), level,
                                 (loc[0][1] - loc[0][0],
                                  loc[1][1] - loc[1][0])).to_array(image_info)

        predictions = model.predict(data[filter_array])
        res[filter_array] = predictions

        return res

    def _classify_patches(self,
                          slide: BasicSlide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          round_to_0_100: bool = True,
                          chunk=None) -> Mask:
        slide_array = self._get_slide_array(slide, level).array

        shape_1 = slide_array.shape[1] - (slide_array.shape[1] %
                                          self._patch_size[0])
        shape_2 = slide_array.shape[2] - (slide_array.shape[2] %
                                          self._patch_size[1])
        slide_array = slide_array[:, :shape_1, :shape_2]

        chunks = []
        for axis in range(2):
            slide_chunks = np.array(
                slide_array.chunks[axis + 1]) // self._patch_size[axis]
            chunks.append(slide_chunks.tolist())

        if filter_ is None:
            predictions = da.map_blocks(self._predict_patches,
                                        slide_array,
                                        delayed(self._model),
                                        drop_axis=0,
                                        meta=np.array([], dtype='float32'),
                                        chunks=chunks)
        else:
            chunks = 20
            #  chunks = chunks if min(filter_.array.shape +
            #                         chunks) > chunks[0] else 'auto'
            filter_array = da.from_array(filter_.array, chunks=chunks)
            predictions = da.map_blocks(self._predict_patch_with_filter,
                                        filter_array,
                                        delayed(self.model),
                                        slide.filename,
                                        type(slide._slide),
                                        level,
                                        meta=np.array([], dtype='float32'))

        return predictions

    def _predict_patch_with_filter(self,
                                   filter_array,
                                   model,
                                   slide_filename,
                                   slide_cls,
                                   level,
                                   block_info=None):
        predictions = np.zeros(filter_array.shape, dtype='float32')
        if np.count_nonzero(filter_array) == 0:
            return predictions
        loc = block_info[0]['array-location'][::-1]
        slide = slide_cls(slide_filename)
        pos = (loc[0][0] * self._patch_size[0],
               loc[1][0] * self._patch_size[1])
        size = ((loc[0][1] - loc[0][0]) * self._patch_size[0],
                (loc[1][1] - loc[1][0]) * self._patch_size[1])
        data = slide.read_region(pos, level,
                                 size).to_array(self.model.image_info)

        data = np.concatenate([
            np.split(row, data.shape[2] // self._patch_size[1], 2)
            for row in np.split(data, data.shape[1] // self._patch_size[0], 1)
        ])
        filtered_predictions = model.predict(data[filter_array.flatten()])
        predictions[filter_array] = filtered_predictions
        return predictions

    def _predict_patches(self, chunk, model):
        final_shape = (chunk.shape[1] // self._patch_size[0],
                       chunk.shape[2] // self._patch_size[1])
        data = np.concatenate([
            np.split(row, final_shape[1], 2)
            for row in np.split(chunk, final_shape[0], 1)
        ])
        predictions = model.predict(data).reshape(final_shape)
        return predictions

    @staticmethod
    def _get_zeros(size, dtype):
        return da.zeros(size, dtype=dtype)

    @staticmethod
    def _concatenate(seq, axis):
        #  seq = [el for el in seq if el.size]
        return da.concatenate(seq, axis)

    @staticmethod
    def _reshape(array, shape):
        return da.reshape(array, shape)


def _rescale(array, scale, block_info=None):
    #  logger.debug('scale %s', scale)
    #  logger.debug('array.shape %s', array.shape)
    res = np.zeros((array.shape[0] * scale[0], array.shape[1] * scale[1], 1),
                   dtype='bool')
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            res[x * scale[0]:x * scale[0] + scale[0],
                y * scale[1]:y * scale[1] + scale[1]] = array[x, y]
    #  res = np.expand_dims(res, 2)
    #  logger.debug('res.shape %s', res.shape)
    return res
