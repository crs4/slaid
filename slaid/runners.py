import abc
import logging
import os
import re
from dataclasses import dataclass
from importlib import import_module
from typing import List

import numpy as np
from clize import parameters

import slaid.commons.ecvl as ecvl
from slaid.classifiers import BasicClassifier
from slaid.classifiers.fixed_batch import (FilteredPatchClassifier,
                                           FilteredPixelClassifier,
                                           PixelClassifier)
from slaid.commons import ImageInfo
from slaid.commons.base import Filter
from slaid.models.factory import Factory as ModelFactory
from slaid.writers import REGISTRY as STORAGE
from slaid.writers.zarr import ZarrStorage

DEFAULT_BATCH_SIZE = 10000


class SlideFactory:

    def __init__(self,
                 filename: str,
                 basic_slide_module: str,
                 slide_module: str,
                 image_info: ImageInfo = None):
        self._filename = filename.rstrip('/')
        self._basic_slide_module = basic_slide_module
        self._slide_module = slide_module
        self._image_info = image_info

    def get_slide(self):
        basic_slide_cls = import_module(
            f'slaid.commons.{self._basic_slide_module}').BasicSlide
        slide_cls = import_module(f'slaid.commons.{self._slide_module}').Slide

        slide_ext_with_dot = os.path.splitext(self._filename)[-1]
        slide_ext = slide_ext_with_dot[1:]
        try:
            basic_slide = STORAGE[slide_ext].load(self._filename)
        except KeyError:
            basic_slide = basic_slide_cls(self._filename)
        return slide_cls(basic_slide)


@dataclass
class Runner(abc.ABC):
    input_path: str
    output_dir: str
    model_name: str
    level: int
    label: str
    writer: str
    threshold: float = None
    gpu: List[int] = None
    overwrite_output_if_exists: bool = False
    no_round: bool = False
    filter_slide: str = None
    slide_reader: str = None
    batch_size: int = None

    def __post_init__(self):
        self.gpu = _convert_gpu_params(self.gpu)
        self.model = ModelFactory(self.model_name, gpu=self.gpu).get_model()

        _prepare_output_dir(self.output_dir)
        self._classifier = None

    @abc.abstractproperty
    def classifier(self):
        ...

    def run(self):
        classifiled_slides = []
        for slide in _get_slides(self.input_path, self.slide_reader):
            output_path = os.path.join(
                self.output_dir,
                f'{os.path.basename(slide.filename)}.{self.writer}')

            storage = ZarrStorage(self.classifier.label, output_path)
            self.classifier.array_factory = storage
            mask = self.classifier.classify(slide,
                                            level=self.level,
                                            threshold=self.threshold,
                                            round_to_0_100=not self.no_round,
                                            batch_size=self.batch_size)
            slide.masks[self.label] = mask
            storage.write(mask)
            storage.add_metadata({
                'filename': slide.filename,
                'resolution': slide.dimensions
            })

            classifiled_slides.append(slide)
            print(output_path)
        return self.classifier, classifiled_slides


@dataclass
class FilteredRunner(Runner):
    _filter: str = None

    def __post_init__(self):
        super().__post_init__()
        self._filter_obj = self._process_filter()

    def _process_filter(self) -> Filter:
        if self._filter:
            return self._convert_condition()

    def _convert_condition(self) -> Filter:
        operator_mapping = {
            '>': '__gt__',
            '>=': '__ge__',
            '<': '__lt__',
            '<=': '__le__',
            '==': '__eq__',
            '!=': '__ne__',
        }

        condition = self._filter.replace('"', '')
        parsed = re.match(
            r"(?P<mask>\w+)\s*(?P<operator>[<>=!]+)\s*(?P<value>\d+\.*\d*)",
            condition).groupdict()
        label = parsed['mask']
        value = float(parsed['value'])

        mask = ZarrStorage(label, self.filter_slide).load()
        operator = operator_mapping[parsed['operator']]
        return getattr(mask, operator)(value)


@dataclass
class SerialRunner(FilteredRunner):
    chunk_size: int = None

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = BasicClassifier(
                self.model,
                self.label,
                chunk=(self.chunk_size,
                       self.chunk_size) if self.chunk_size else None,
                _filter=self._filter)
        return self._classifier


@dataclass
class FilteredPatchRunner(FilteredRunner):

    def __post_init__(self):
        super().__post_init__()
        if self.model.patch_size is None:
            raise RuntimeError(
                f'model {self.model_name} does not work with patch')

        if self._filter is None:
            raise NotImplementedError(
                'Prediction patch based without filtering not implemented.')
        self.batch_size = self.batch_size or 10

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = FilteredPatchClassifier(self.model, self.label,
                                                       self._filter_obj)
        return self._classifier


@dataclass
class PixelRunner(Runner):
    chunk_size: int = None

    def __post_init__(self):
        super().__post_init__()
        self.batch_size = self.batch_size or DEFAULT_BATCH_SIZE

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = PixelClassifier(self.model,
                                               self.label,
                                               chunk_size=self.chunk_size)

        return self._classifier


@dataclass
class FilteredPixelRunner(FilteredRunner):

    def __post_init__(self):
        super().__post_init__()
        self.batch_size = self.batch_size or DEFAULT_BATCH_SIZE

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = FilteredPixelClassifier(
                self.model, self.label, _filter=self._filter_obj)

        return self._classifier


def basic(input_path: str,
          *,
          model: (str, 'm'),
          level: (int, 'l'),
          output_dir: (str, 'o'),
          label: (str, 'L'),
          threshold: (float, 't') = None,
          gpu: (int, parameters.multi()) = None,
          writer: ('w', parameters.one_of(*list(STORAGE.keys()))) = list(
              STORAGE.keys())[0],
          _filter: 'F' = None,
          overwrite_output_if_exists: 'overwrite' = False,
          no_round: bool = False,
          filter_slide: str = None,
          slide_reader: ('r', parameters.one_of('ecvl', 'openslide')) = 'ecvl',
          batch_size: ('b', int) = None,
          chunk_size: int = None):
    return SerialRunner(input_path,
                        model_name=model,
                        level=level,
                        output_dir=output_dir,
                        label=label,
                        threshold=threshold,
                        gpu=gpu,
                        writer=writer,
                        _filter=_filter,
                        overwrite_output_if_exists=overwrite_output_if_exists,
                        no_round=no_round,
                        filter_slide=filter_slide,
                        slide_reader=slide_reader,
                        batch_size=batch_size,
                        chunk_size=chunk_size).run()


def fixed_batch(input_path: str,
                *,
                model: (str, 'm'),
                level: (int, 'l'),
                output_dir: (str, 'o'),
                label: (str, 'L'),
                threshold: (float, 't') = None,
                gpu: (int, parameters.multi()) = None,
                writer: ('w', parameters.one_of(*list(STORAGE.keys()))) = list(
                    STORAGE.keys())[0],
                _filter: 'F' = None,
                overwrite_output_if_exists: 'overwrite' = False,
                no_round: bool = False,
                filter_slide: str = None,
                slide_reader: ('r', parameters.one_of('ecvl',
                                                      'openslide')) = 'ecvl',
                chunk_size: int = None,
                batch_size: ('b', int) = None):
    kwargs = dict(input_path=input_path,
                  model_name=model,
                  level=level,
                  output_dir=output_dir,
                  label=label,
                  threshold=threshold,
                  gpu=gpu,
                  writer=writer,
                  _filter=_filter,
                  overwrite_output_if_exists=overwrite_output_if_exists,
                  no_round=no_round,
                  filter_slide=filter_slide,
                  slide_reader=slide_reader,
                  batch_size=batch_size)
    _model = ModelFactory(model, gpu=gpu).get_model()
    if _filter:
        kwargs['_filter'] = _filter
        if _model.patch_size:
            cls = FilteredPatchRunner
        else:
            cls = FilteredPixelRunner
    else:
        kwargs.pop('_filter')
        kwargs.pop('filter_slide')
        cls = PixelRunner
        kwargs['chunk_size'] = chunk_size

    cls(**kwargs).run()


def _convert_gpu_params(gpu: List[int]) -> List[int]:
    if gpu:
        res = np.zeros(max(gpu) + 1, dtype='uint8')
        np.put(res, gpu, 1)
        return list(res)
    return gpu


def _prepare_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def _get_slides(input_path, slide_reader):

    inputs = [
        os.path.abspath(os.path.join(input_path, f))
        for f in os.listdir(input_path)
    ] if os.path.isdir(input_path) and os.path.splitext(
        input_path)[-1][1:] not in STORAGE.keys() else [input_path]
    logging.info('processing inputs %s', inputs)
    for f in inputs:
        yield SlideFactory(f, slide_reader, 'base').get_slide()

    def _get_slide(path, slide_reader):
        return SlideFactory(path, slide_reader, 'base').get_slide()
