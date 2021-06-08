import logging
import os
from datetime import datetime as dt
from tempfile import NamedTemporaryFile, TemporaryDirectory

import zarr

from slaid.commons import BasicSlide, Mask
from slaid.commons.ecvl import BasicSlide as EcvlSlide
from slaid.writers import Storage, _dump_masks, _get_slide_metadata

logger = logging.getLogger(__file__)


class ZarrDirectoryStorage(Storage, _name='zarr'):
    @staticmethod
    def dump(slide: BasicSlide,
             output_path: str,
             mask: str = None,
             overwrite: bool = False,
             **kwargs):
        logger.info('dumping slide to zarr on path %s', output_path)
        group = zarr.open_group(output_path)
        if not group.attrs:
            group.attrs.update(_get_slide_metadata(slide))
        _dump_masks(output_path, slide, overwrite, 'to_zarr', mask, **kwargs)

    @staticmethod
    def empty(shape, dtype):
        temp_dir = TemporaryDirectory(suffix='.zarr')
        return zarr.open(temp_dir.name, shape=shape, dtype=dtype)

    @staticmethod
    def load(path: str) -> BasicSlide:
        logger.info('loading slide from zarr at path %s', path)
        group = zarr.open_group(path)
        try:
            slide = EcvlSlide(group.attrs['filename'])
        except BasicSlide.InvalidFile:
            # FIXME: workaround for cwl
            slide = EcvlSlide(os.path.basename(group.attrs['filename']))
        for name, value in group.arrays():
            try:
                logger.info('loading mask %s, %s', name, value.attrs.asdict())
                kwargs = value.attrs.asdict()
                if 'datetime' in kwargs:
                    kwargs['datetime'] = dt.fromtimestamp(kwargs['datetime'])
                slide.masks[name] = Mask(value, **kwargs)
            except Exception as ex:
                logger.error('skipping mask %s, exception: %s ', name, ex)
                raise ex
        return slide

    @staticmethod
    def mask_exists(path: str, mask: 'str') -> bool:
        if not os.path.exists(path):
            return False
        group = zarr.open_group(path)
        return mask in group.array_keys()


class ZarrZipStorage(ZarrDirectoryStorage, _name='zarr-zip'):
    @staticmethod
    def load(path: str) -> BasicSlide:
        logger.info('loading slide from zarr at path %s', path)
        storage = zarr.storage.ZipStore(path, mode='r')
        group = zarr.open_group(storage)
        try:
            slide = EcvlSlide(group.attrs['filename'])
        except BasicSlide.InvalidFile:
            # FIXME: workaround for cwl
            slide = EcvlSlide(os.path.basename(group.attrs['filename']))
        for name, value in group.arrays():
            try:
                logger.info('loading mask %s, %s', name, value.attrs.asdict())
                kwargs = value.attrs.asdict()
                if 'datetime' in kwargs:
                    kwargs['datetime'] = dt.fromtimestamp(kwargs['datetime'])
                slide.masks[name] = Mask(value, **kwargs)
            except Exception as ex:
                logger.error('skipping mask %s, exception: %s ', name, ex)
        return slide

    @staticmethod
    def dump(slide: BasicSlide,
             output_path: str,
             mask: str = None,
             overwrite: bool = False,
             **kwargs):
        # FIXME duplicated code
        logger.info('dumping slide to zarr on path %s', output_path)
        storage = zarr.storage.ZipStore(output_path)
        group = zarr.open_group(storage)
        if not group.attrs:
            group.attrs.update(_get_slide_metadata(slide))

        for name, mask_ in slide.masks.items():
            group[name] = mask_.array

            for attr, value in mask_._get_attributes().items():
                logger.info('writing attr %s %s', attr, value)
                group[name].attrs[attr] = value
        storage.close()

    @staticmethod
    def empty(shape, dtype):
        temp_file = NamedTemporaryFile(suffix='zarr-zip')
        return zarr.open(temp_file.name, shape=shape, dtype=dtype)
