import os
import unittest
from PIL import Image

import hanashi.processor.page_processor as processor


class PageProcessorTest(unittest.TestCase):
    def test_onepiece(self):

        filename = os.path.join(os.path.dirname(__file__), 'resources/onepiece.jpg')

        masks, lines, rectangles = processor.process(filename)
        text = processor.extract_text(masks)

        self.assertAlmostEqual(len(text), 10, delta=3)