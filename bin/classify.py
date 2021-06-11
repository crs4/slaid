#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import clize
import pkg_resources
import tiledb

from slaid.runners import run

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s '
                    '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)


def set_model(func, model):
    def wrapper(*args, **kwargs):
        return func(*args, model=model, **kwargs)

    return wrapper


def load_config_file(config_file: str, backend: str):
    if config_file is None:
        return
    if backend == 'tiledb':
        config = tiledb.Config.load(config_file)
        tiledb.Ctx(config)
        tiledb.VFS(config)


if __name__ == '__main__':

    model = os.environ.get("SLAID_MODEL")
    if model:
        model = pkg_resources.resource_filename('slaid',
                                                f'resources/models/{model}')
        run = set_model(run, model)

    clize.run(run)
