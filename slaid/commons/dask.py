import logging
from tempfile import TemporaryDirectory

import dask
import dask.array as da
import tiledb
import zarr
from dask.distributed import Client, progress

from slaid.commons import Mask as BaseMask
from slaid.commons.base import NapariSlide as BaseSlide
from slaid.commons.base import SlideArray

logger = logging.getLogger()


def init_client(address=None, processes=False, **kwargs):
    if address:
        return Client(address, **kwargs)
    dask.config.config['distributed']['comm']['timeouts']['connect'] = '20s'
    return Client(processes=processes, **kwargs)


class Mask(BaseMask):

    def compute(self):
        if isinstance(self.array, da.Array):
            self.array = self.array.compute(rerun_exceptions_locally=True)

    def to_tiledb(self, path: str, overwrite: bool = False, **kwargs):
        if overwrite:
            try:
                tiledb.remove(path)
            except tiledb.libtiledb.TileDBError as ex:
                logger.error(ex)
        da.to_tiledb(self.array, path, compute=True, **kwargs)
        self._write_meta_tiledb(path)

    #  FIXME: duplicate code
    def to_zarr(self, group, name: str, overwrite: bool = False):
        if overwrite and name in group:
            del group[name]
        if isinstance(self.array, da.Array):
            # workaround since directly store dask
            # array to zarr group does not seem to work
            with TemporaryDirectory() as tmp_dir:
                array = da.to_zarr(self.array,
                                   tmp_dir,
                                   compute=True,
                                   overwrite=True,
                                   return_stored=True)
                progress(array)
                print('tmp_dir', tmp_dir)
                array = zarr.open(tmp_dir, mode='r')
                group[name] = array
                array = group[name]
        else:
            array = group.array(name, self.array)
        for attr, value in self._get_attributes().items():
            logger.info('writing attr %s %s', attr, value)
            array.attrs[attr] = value


class Slide(BaseSlide):

    def _read_from_store(self, dataset):
        return da.from_zarr(self._store, component=dataset["path"])

    def _create_slide(self, dataset):
        return DaskSlideArray(self._read_from_store(dataset),
                              self._slide.IMAGE_INFO).convert(self.image_info)


class DaskSlideArray(SlideArray):

    @property
    def array(self):
        return self._array
