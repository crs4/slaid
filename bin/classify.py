#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import clize
import pkg_resources

from slaid.runners import fixed_batch

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s '
                    '[%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)


def set_model(func, model):

    def wrapper(*args, **kwargs):
        return func(*args, model=model, **kwargs)

    return wrapper


if __name__ == '__main__':

    model = os.environ.get("SLAID_MODEL")
    if model:
        model = pkg_resources.resource_filename('slaid',
                                                f'resources/models/{model}')
        fixed_batch = set_model(fixed_batch, model)

    clize.run({'fixed-batch': fixed_batch})
