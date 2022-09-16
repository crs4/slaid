import logging
import os
from urllib.parse import urlparse
from urllib.request import urlretrieve

model_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "resources/models"
)

logger = logging.getLogger()


def retrieve_model(uri: str, dest_dir: str = model_dir) -> str:
    local_path: str

    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        logger.info("retrieving remote model from %s", uri)
        local_path = os.path.join(dest_dir, os.path.basename(parsed.path))
        if not os.path.exists(local_path):
            urlretrieve(uri, local_path)
    elif parsed.scheme in {"file", ""}:
        local_path = uri
    else:
        raise UnsupportedScheme(parsed.scheme)

    return local_path


class UnsupportedScheme(Exception):
    ...
