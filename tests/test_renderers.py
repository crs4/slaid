import json
import unittest

import cloudpickle as pickle
import numpy as np
from PIL import Image
from test_commons import DummySlide

from slaid.classifiers import KarolinskaFeature
from slaid.commons import Patch
from slaid.renderers import BasicFeatureTIFFRenderer, to_json, to_pickle


class BasicFeatureTIFFRendererTest(unittest.TestCase):
    def test_render_patch(self):
        features = {KarolinskaFeature.CANCER_PERCENTAGE: 1}
        patch = Patch(DummySlide('slide', (100, 100)), (0, 0), (10, 10),
                      features)
        renderer = BasicFeatureTIFFRenderer()
        output = '/tmp/patch.tiff'
        renderer.render_patch(output, patch)
        image = Image.open(output)
        data = np.array(image)
        self.assertEqual(data.shape, (10, 10, 4))
        self.assertTrue((data[:, :, 0] == 255).all())


class ToPickleTest(unittest.TestCase):
    def test_render(self):
        # given
        slide = DummySlide('slide', (256, 256))

        # when
        pickled = pickle.loads(to_pickle(slide))

        # then
        self.assertTrue(slide.patches.dataframe.equals(pickled['features']))
        self.assertEqual(slide.patches.patch_size, pickled['patch_size'])
        self.assertEqual(slide.patches.extraction_level,
                         pickled['extraction_level'])


class ToJsonTest(unittest.TestCase):
    def test_np_array(self):
        array = np.zeros((10, 10))
        jsoned_array = json.loads(to_json(array))
        self.assertTrue(np.array_equal(array, jsoned_array))

    def test_slide(self):
        #  given
        slide = DummySlide('s', (10, 20), patch_size=(10, 10))
        prob = 10
        slide.patches.add_feature('prob', prob)

        #  when
        jsoned = json.loads(to_json(slide))
        #  then
        self.assertEqual(jsoned['filename'], slide.ID)
        self.assertEqual(tuple(jsoned['patch_size']), slide.patch_size)
        self.assertEqual(len(slide.patches), len(jsoned['features']))

        for f in jsoned['features']:
            self.assertEqual(len(f),
                             len(slide.patches.features) + 2)  # features +x +y
            self.assertEqual(
                slide.patches.get_patch((f['x'], f['y'])).features['prob'],
                f['prob'])
