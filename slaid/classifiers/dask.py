import logging

import dask.array as da
import numpy as np
from dask import delayed
from dask.distributed import Client

from slaid.classifiers.base import BasicClassifier, Filter
from slaid.commons import BasicSlide, Slide
from slaid.commons.dask import Mask
from slaid.models.base import Model
from slaid.models.eddl import Model as EddlModel
from slaid.models.eddl import load_model

logger = logging.getLogger('dask')
logger.setLevel(logging.DEBUG)


class Classifier(BasicClassifier):
    MASK_CLASS = Mask

    def __init__(self, model: Model, feature: str, compute_mask: bool = False):
        super().__init__(model, feature)
        self.compute_mask = compute_mask

    def classify(self,
                 slide: Slide,
                 filter_=None,
                 threshold: float = None,
                 level: int = 2,
                 round_to_0_100: bool = True,
                 max_MB_prediction=None) -> Mask:
        mask = super().classify(slide, filter_, threshold, level,
                                round_to_0_100, max_MB_prediction)
        if self.compute_mask:
            mask.compute()
        return mask

    def _predict(self, array):
        n_px = array.shape[0] * array.shape[1]
        p = self._model.predict(array.reshape(
            (n_px, 3))).reshape(array.shape[:2])
        return p
        #  return da.from_delayed(self._delayed_model.predict(area.array),
        #                         shape=(area.array.shape[0], ),
        #                         dtype='float32')

    @delayed
    def _get_delayed_model(self):
        return load_model(self._model.weight_filename,
                          self._model.gpu) if isinstance(
                              self._model, EddlModel) else self._model

    def _classify_batches_no_filter(self, slide_array, max_MB_prediction):
        prediction = slide_array.array.map_blocks(self._predict,
                                                  meta=np.array(
                                                      (), dtype='float32'),
                                                  drop_axis=2)
        return prediction

    #  @delayed
    #  def _classify_batches_no_filter(self, slide_array, level, n_batch):
    #      return super()._classify_batches_no_filter(slide_array, level, n_batch)
    #  return da.rechunk(predictions)

    #  def _classify_batches_with_filter(self, slide_array, filter_,
    #                                    scale_factor):
    #      model = load_model(self._model.weight_filename, False) if isinstance(
    #          self._model, EddlModel) else self._model
    #      res = np.zeros((slide_array.shape[0], slide_array.shape[1]),
    #                     dtype='float32')
    #      step = 2
    #      with Bar('pixels', max=len(filter_.indices) / step) as progress:
    #          for pixels in toolz.partition_all(step, filter_.indices):
    #              logger.debug('pixels %s', pixels)
    #              areas = []
    #              for pixel in pixels:
    #                  pixel = pixel * scale_factor
    #                  area = slide_array[pixel[0]:pixel[0] + scale_factor,
    #                                     pixel[1]:pixel[1] + scale_factor, :3]
    #                  n_px = area.shape[0] * area.shape[1]
    #                  areas.append(np.array(area.reshape((n_px, 3))))
    #
    #              logger.debug('np.concatenate(areas, 0) %s',
    #                           np.concatenate(areas, 0))
    #              predictions = model.predict(np.concatenate(areas, 0))
    #              for i, pixel in enumerate(pixels):
    #                  idx = i * scale_factor
    #                  pixel = pixel * scale_factor
    #                  prediction = predictions[idx:idx + scale_factor].reshape(
    #                      scale_factor, scale_factor)
    #                  res[pixel[0]:pixel[0] + scale_factor - 1,
    #                      pixel[1]:pixel[1] + scale_factor - 1] = prediction
    #              progress.next()
    #
    #      return res
    #  @delayed
    #  def _classify_batches_with_filter(self, slide_array, filter_,
    #                                    scale_factor):
    #      res = np.zeros(slide_array.size, dtype='float32')
    #      model = load_model(self._model.weight_filename,
    #                         self._model.gpu) if isinstance(
    #                             self._model, EddlModel) else self._model
    #      for pixel in filter_:
    #          pixel = pixel * scale_factor
    #          area = slide_array[pixel[0]:pixel[0] + scale_factor,
    #                             pixel[1]:pixel[1] + scale_factor]
    #          n_px = area.size[0] * area.size[1]
    #          prediction = model.predict(area.reshape(n_px).array)
    #          res[pixel[0]:pixel[0] + scale_factor,
    #              pixel[1]:pixel[1] + scale_factor] = prediction.reshape(
    #                  (scale_factor, scale_factor))
    #      return res
    #
    #  def _classify_batches(self, slide, level, n_batch, threshold,
    #                        round_to_0_100, filter_):
    #      slide_array = slide[level]
    #      model = delayed(self.model)
    #      if filter_ is not None:
    #          scale_factor = round(slide.level_downsamples[
    #              filter_.mask.extraction_level]) // round(
    #                  slide.level_downsamples[level])
    #
    #          predictions = da.from_delayed(self._classify_batches_with_filter(
    #              slide_array, filter_, scale_factor),
    #                                        shape=slide_array.size,
    #                                        dtype='float32')
    #  predictions = da.from_delayed(
    #      self._classify_batches_with_filter(slide_array, filter_,
    #                                         scale_factor),
    #      shape=(slide_array.shape[0], slide_array.shape[1]),
    #  dtype='float32')
    #  return da.rechunk(predictions)
    #  res = np.zeros((slide_array.shape[0], slide_array.shape[1]),
    #                 dtype='float32')
    #  for pixel in filter_:
    #      pixel = pixel * scale_factor
    #
    #      area = slide_array[pixel[0]:pixel[0] + scale_factor,
    #                         pixel[1]:pixel[1] + scale_factor, :3]
    #      n_px = area.shape[0] * area.shape[1]
    #      area_reshaped = area.reshape((n_px, 3))
    #      prediction = da.from_delayed(self.model.predict(area),
    #                                   shape=(area_reshaped.shape[0], ),
    #                                   dtype='float32')
    #      prediction = prediction.reshape((scale_factor, scale_factor))
    #      res[pixel[0]:pixel[0] + scale_factor,
    #          pixel[1]:pixel[1] + scale_factor] = prediction
    #
    #  predictions = []
    #  area_by_row = defaultdict(list)
    #  for pixel in filter_:
    #      pixel = pixel * scale_factor
    #
    #      area = slide_array[pixel[0]:pixel[0] + scale_factor,
    #                         pixel[1]:pixel[1] + scale_factor, :3]
    #      n_px = area.shape[0] * area.shape[1]
    #      area_reshaped = area.reshape((n_px, 3))
    #      area_by_row[pixel[0]].append((pixel[1], area_reshaped))
    #
    #  for row in range(0, slide_array.shape[0], scale_factor):
    #      step = scale_factor if row + scale_factor < slide_array.shape[
    #          0] else slide_array.shape[0] - row
    #
    #      row_predictions = []
    #      if row not in area_by_row:
    #          row_predictions = da.zeros((step, slide_array.shape[1]),
    #                                     dtype='float32')
    #      else:
    #          areas = area_by_row[row]
    #          prev_y = 0
    #          for y, area in areas:
    #              pad_0 = y - prev_y
    #              if pad_0 > 0:
    #                  row_predictions.append(
    #                      da.zeros((scale_factor, pad_0),
    #                               dtype='float32'))
    #
    #              prediction = da.from_delayed(model.predict(area),
    #                                           shape=(area.shape[0], ),
    #                                           dtype='float32')
    #              row_predictions.append(
    #                  prediction.reshape(scale_factor, scale_factor))
    #              prev_y = y + scale_factor
    #
    #          y = slide_array.shape[1]
    #          pad_0 = y - prev_y
    #          if pad_0:
    #              row_predictions.append(
    #                  da.zeros((scale_factor, pad_0), dtype='float32'))
    #
    #          row_predictions = da.concatenate(row_predictions, 1)
    #
    #      predictions.append(row_predictions)
    #  predictions = da.concatenate(predictions, 0)
    #  logger.debug('predictions shape %s', predictions.shape)
    #  return da.rechunk(predictions)

    #  res = da.concatenate(res)
    #  logger.debug(
    #      'dif shapes %s',
    #      res.shape[0] - slide_array.shape[0] * slide_array.shape[1])
    #  assert res.shape[0] == slide_array.shape[0] * slide_array.shape[1]
    #  logger.debug('res shape %s, slide_array shape %s', res.shape,
    #               slide_array.shape)
    #  return res.reshape((slide_array.shape[0], slide_array.shape[1]))

    #  else:
    #      predictions = []
    #      step = slide_array.size[0] // n_batch
    #      logger.debug('n_batch %s, step %s', n_batch, step)
    #      with Bar('batches', max=n_batch) as bar:
    #          for i in range(0, slide_array.size[0], step):
    #              bar.next()
    #              area = slide_array[i:i + step, :]
    #              n_px = area.size[0] * area.size[1]
    #              area_reshaped = area.reshape((n_px, ))
    #
    #              prediction = da.from_delayed(
    #                  model.predict(area_reshaped.array),
    #                  shape=(area_reshaped.size[0], ),
    #                  dtype='float32')
    #              prediction = prediction.reshape(area.size[0], area.size[1])
    #              predictions.append(prediction)
    #      return da.concatenate(predictions, 0)

    def _classify_patches(self,
                          slide: BasicSlide,
                          patch_size,
                          level,
                          filter_: Filter,
                          threshold,
                          n_patch: int = 25,
                          round_to_0_100: bool = True) -> Mask:
        dimensions = slide.level_dimensions[level][::-1]
        dtype = 'uint8' if threshold or round_to_0_100 else 'float32'
        patches_to_predict = filter_ if filter_ is not None else np.ndindex(
            dimensions[0] // patch_size[0], dimensions[1] // patch_size[1])

        slide_array = slide[level]
        model = delayed(self.model)
        predictions = []
        patches_to_predict = list(patches_to_predict)
        for i in range(0, len(patches_to_predict), n_patch):
            patches = patches_to_predict[i:i + n_patch]
            input_array = da.stack([
                slide_array[p[0]:p[0] + self._patch_size[0],
                            p[1]:p[1] + self._patch_size[1]].array
                for p in patches
            ])
            predictions.append(
                da.from_delayed(model.predict(input_array),
                                shape=(len(patches), ),
                                dtype=dtype))
        if predictions:
            predictions = da.concatenate(predictions)

        predictions = predictions.compute()
        res = np.zeros(
            (dimensions[0] // patch_size[0], dimensions[1] // patch_size[1]),
            dtype='float32')
        for i, p in enumerate(predictions):
            patch = patches_to_predict[i]
            res[patch[0], patch[1]] = p
        return da.array(res, dtype='float32')

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


def _classify_batch(slide_path: str, model_path: str, level: int):
    pass
