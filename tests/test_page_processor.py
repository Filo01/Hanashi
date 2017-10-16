import os
import unittest

import pytesseract

import hanashi.processor.page_processor as processor


class PageProcessorTest(unittest.TestCase):
    def test_processor(self):

        filename = os.path.join(os.path.dirname(__file__), 'resources/onepiece.jpg')

        masks = processor.process(filename)
        n = 0
        for mask in masks:
            s = (pytesseract.image_to_string(mask[2])).strip()
            if s != "":
                n += 1
        self.assertAlmostEqual(n, 10, delta=3)
