import abc
import os
import pkgutil
from typing import Dict, Tuple

from slaid.commons.base import BasicSlide, Image, Mask

REGISTRY = {}


class Storage(abc.ABC):

    def __init_subclass__(cls, _name, **kwargs):
        super().__init_subclass__(**kwargs)
        for key in _name.split(','):
            REGISTRY[key] = cls

    @abc.abstractmethod
    def write(self, mask: Mask):
        ...

    @abc.abstractmethod
    def add_metadata(self, metadata: Dict):
        ...

    @abc.abstractmethod
    def load(self) -> Mask:
        ...

    @abc.abstractmethod
    def mask_exists(self) -> bool:
        ...


def _get_slide_metadata(slide: BasicSlide) -> dict:
    return {'filename': slide.filename, 'resolution': slide.dimensions}


def _dump_masks(path: str,
                slide: BasicSlide,
                overwrite: bool,
                func: str,
                only_mask: str = None,
                **kwargs):
    masks = {only_mask: slide.masks[only_mask]} if only_mask else slide.masks
    for name, mask_ in masks.items():
        getattr(mask_, func)(os.path.join(path, name), overwrite, **kwargs)


class ReducedSlide(BasicSlide):

    def __init__(self, filename: str):
        self._filename = os.path.abspath(filename)
        self.masks: Dict[str, Mask] = {}

    def __eq__(self, other):
        return self._filename == other.filename and self.masks == other.masks

    @property
    def dimensions(self) -> Tuple[int, int]:
        raise NotImplementedError

    @property
    def filename(self):
        return self._filename

    def read_region(self, location: Tuple[int, int],
                    size: Tuple[int, int]) -> Image:
        raise NotImplementedError

    def get_best_level_for_downsample(self, downsample: int):
        raise NotImplementedError

    @property
    def level_dimensions(self):
        raise NotImplementedError

    @property
    def level_downsamples(self):
        raise NotImplementedError


"""
Following code allows to automagically populate the REGISTRY
with all Storage subclasses defined in submodules.

"""
__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module = loader.find_module(module_name).load_module(module_name)
    globals()[module_name] = _module
