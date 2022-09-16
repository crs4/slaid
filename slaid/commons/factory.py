import abc
import logging
from dataclasses import dataclass

from slaid.commons import BasicSlide, ImageInfo, Slide

logger = logging.getLogger("slaid.commons.factory")


class BaseSlideFactory(abc.ABC):
    @abc.abstractmethod
    def get_slide(self) -> Slide:
        ...


class MetaSlideFactory:
    _registry = {}

    @staticmethod
    def register(cls_to_create: Slide):
        def _register(cls_factory: BaseSlideFactory):
            MetaSlideFactory._registry[cls_to_create] = cls_factory

        return _register

    def get_factory(self, cls_name: str, *args) -> BaseSlideFactory:
        return self._registry[cls_name](*args)


@MetaSlideFactory.register(Slide)
@dataclass
class SlideFactory(BaseSlideFactory):
    slide_filename: str
    slide_reader_cls: BasicSlide

    def get_slide(self) -> Slide:
        return Slide(self.slide_reader_cls(self.slide_filename))
