import logging
from datetime import datetime as dt

import dask.array as da
import numpy as np
from dask import delayed
from pyecvl.ecvl import OpenSlideRead

from slaid.classifiers.base import BasicClassifier
from slaid.classifiers.base import Batch as BaseBatch
from slaid.classifiers.base import BatchIterator, Filter, Model
from slaid.classifiers.base import Patch as BasePatch
from slaid.commons import Slide
from slaid.commons.dask import Mask
from slaid.models.eddl import Model as EddlModel
from slaid.models.eddl import load_model

logger = logging.getLogger('dask')


class Patch(BasePatch):
    pass
    #  @property
    #  def array(self):
    #      return delayed(super().array)


class Batch(BaseBatch):
    pass
    #  @property
    #  def array(self):
    #      return delayed(super().array)


class Classifier(BasicClassifier):
    MASK_CLASS = Mask

    #  lock = threading.Lock()

    @property
    def model(self):
        return load_model(self._model.weight_filename,
                          self._model.gpu) if isinstance(
                              self._model, EddlModel) else self._model

    @staticmethod
    def _get_batch_iterator(slide, level, n_batch, color_type, coords,
                            channel):
        return BatchIterator(slide, level, n_batch, color_type, coords,
                             channel, Batch)

    def _classify_batches(self, batches: BatchIterator, threshold: float,
                          round_to_0_100: bool) -> Mask:
        predictions = []
        for batch in batches:
            predictions.append(
                da.from_delayed(delayed(self._classify_batch)(batch, threshold,
                                                              round_to_0_100),
                                batch.size,
                                dtype='uint8'
                                if threshold or round_to_0_100 else 'float32'))
        return self._concatenate(predictions, axis=0)

    def _classify_patches(self,
                          slide: Slide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          n_patch: int = 25,
                          round_to_0_100: bool = True) -> Mask:
        dimensions = slide.level_dimensions[level][::-1]
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
        patch_indexes = filter_ if filter_ is not None else np.ndindex(
            dimensions[0] // patch_size[0], dimensions[1] // patch_size[1])
        patches_to_predict = [
            Patch(slide, p[0], p[1], level, patch_size) for p in patch_indexes
        ]

        # adding fake patches, workaround for
        # https://github.com/deephealthproject/eddl/issues/236
        #  for _ in range(patch_to_add):
        #      patches_to_predict.append(patches_to_predict[0])

        predictions = []
        for i in range(0, len(patches_to_predict), n_patch):
            patches = patches_to_predict[i:i + n_patch]
            input_array = da.stack([
                da.from_delayed(p.array(),
                                shape=(p.size[0], p.size[1], 3),
                                dtype=dtype) for p in patches
            ])
            predictions.append(
                da.from_delayed(self._classify_array(input_array, threshold,
                                                     round_to_0_100),
                                shape=(len(patches), ),
                                dtype=dtype))
        if predictions:
            predictions = da.concatenate(predictions)

        logger.debug('predictions %s', predictions)
        predictions = predictions.compute()
        res = np.zeros(
            (dimensions[0] // patch_size[0], dimensions[1] // patch_size[1]),
            dtype=dtype)
        for i, p in enumerate(predictions):
            patch = patches_to_predict[i]
            res[patch.row, patch.column] = p
        return da.array(res, dtype=dtype)

    @staticmethod
    def _get_zeros(size, dtype):
        return da.zeros(size, dtype=dtype)

    @staticmethod
    def _concatenate(seq, axis):
        seq = [el for el in seq if el.size]
        return da.concatenate(seq, axis)

    @staticmethod
    def _reshape(array, shape):
        return da.reshape(array, shape)

    #  def _classify_array(self, array, threshold, round_to_0_100) -> np.ndarray:
    #      with self.lock:
    #          print('locked classify array')
    #          return super()._classify_array(array, threshold, round_to_0_100)


def _classify_batch(slide_path: str, model_path: str, level: int):
    pass


@delayed
def get_array(slide_path, level, dims):
    image = OpenSlideRead(slide_path, level, dims)
    return np.array(image)


#  def classify(
#      slide_path: str,
#      model_path: str,
#      gpu: List[int],
#      level: int,
#      threshold: float,
#      round_to_0_100: bool,
#      n_batch: int,
#  ):
#      model = delayed(load_model)(model_path, gpu)
#      dimensions_0 = slide.dimensions[::-1]
#      dimensions = slide.level_dimensions[level][::-1]
#      step = dimensions_0[0] // n_batch
#      predictions = []
#      for i in range(0, dimensions_0[0], step):
#          array = da.from_delayed(get_array(
#              slide_path, level,
#              (0, i, dimensions[1], step // slide.level_dimensions[level][1])),
#                                  shape=(3, dimensions[1], step //
#                                         slide.level_dimensions[level][1]),
#                                  dtype='uint8')
#          array = array.transpose((1, 2, 0))
#          array = array[:, :, ::-1]
#          n_px = array.shape[0] * array.shape[1]
#          array_reshaped = array.reshape((n_px, 3))
#          prediction = da.from_delayed(model.predict(array_reshaped),
#                                       shape=(n_px, ),
#                                       dtype='float32')
#          prediction.reshape(array.shape[0], array.shape[1])
#          predictions.append(
#              da.from_delayed(model.predict(array),
#                              shape=(n_px, ),
#                              dtype='float32'))
#
#      return da.concatenate(predictions, 0)
#  def classify(self,
#               slide: Slide,
#               filter_=None,
#               threshold: float = None,
#               level: int = 2,
#               n_batch: int = 1,
#               round_to_0_100: bool = True,
#               n_patch=25) -> Mask:
#      dimensions_0 = slide.dimensions[::-1]
#      dimensions = slide.level_dimensions[level][::-1]
#      step = dimensions_0[0] // n_batch
#      predictions = []
#      for i in range(0, dimensions_0[0], step):
#          array = da.from_delayed(
#              get_array(slide.filename, level,
#                        (0, i, dimensions[1],
#                         step // slide.level_dimensions[level][1])),
#              shape=(3, dimensions[1],
#                     step // slide.level_dimensions[level][1]),
#              dtype='uint8')
#          array = array.transpose((1, 2, 0))
#          array = array[:, :, ::-1]
#          n_px = array.shape[0] * array.shape[1]
#          array_reshaped = array.reshape((n_px, 3))
#          prediction = da.from_delayed(self.model.predict(array_reshaped),
#                                       shape=(n_px, ),
#                                       dtype='float32')
#          prediction.reshape(array.shape[0], array.shape[1])
#          predictions.append(prediction)
#
#      res = da.concatenate(predictions, 0)
#      return self._get_mask(res, level, slide.level_downsamples[level],
#                            dt.now(), round_to_0_100)

#  def _classify_batches(self, batches: BatchIterator, threshold: float,
#                        round_to_0_100: bool) -> Mask:
#      return classify(self.slide.filename, self.model.weight_filename, None,
#                      batches.level, True, batches.n_batch)
#  predictions = []
#  for batch in batches:
#      predictions.append(
#          da.from_delayed(self._classify_batch(batch, threshold,
#                                               round_to_0_100),
#                          batch.size,
#                          dtype='uint8'
#                          if threshold or round_to_0_100 else 'float32'))
#  return self._concatenate(predictions, axis=0)
