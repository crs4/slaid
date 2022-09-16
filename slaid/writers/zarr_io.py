import logging
from typing import Dict

from slaid.commons import Mask
from slaid.commons.zarr import GroupArrayFactory
from slaid.writers import Storage

logger = logging.getLogger(__file__)


class ZarrStorage(Storage, GroupArrayFactory, _name="zarr,zip"):
    def add_metadata(self, metadata: Dict):
        for k, v in metadata.items():
            self._root.attrs[k] = v

    def write(self, mask: Mask):
        array = self._root[self.name]
        for attr, value in mask.get_attributes().items():
            logger.info("writing attr %s %s", attr, value)
            array.attrs[attr] = value

    def load(self) -> Mask:
        array = self._root[self.name]
        kwargs = array.attrs.asdict()
        kwargs["array"] = array
        kwargs.pop("dzi_sampling_level")
        return Mask(**kwargs)

    def mask_exists(self) -> bool:
        return len(list(self._root.arrays())) > 0
