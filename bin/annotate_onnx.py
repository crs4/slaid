#!/usr/bin/env python
# -*- coding: utf-8 -*-
import onnx
from clize import parameters, run


def main(model_path: str,
         *,
         pixel_format: parameters.one_of('Bgr8', 'Rgb8') = None,
         pixel_range: parameters.one_of('Normalized_0_1', 'Normalized_1_1',
                                        'NominalRange_0_255') = None):
    """
    :param model_path: path to the onnx file
    :type model_path: str
    """
    model = onnx.load(model_path)
    metadata = {
        'Image.BitmapPixelFormat': pixel_format,
        'Image.NominalPixelRange': pixel_range
    }
    for prop in model.metadata_props:
        new_value = metadata.pop(prop.key)
        prop.value = new_value

    for k, v in metadata.items():
        prop = model.metadata_props.add()
        prop.key = k
        prop.value = v

    onnx.save(model, model_path)


if __name__ == '__main__':
    run(main)
