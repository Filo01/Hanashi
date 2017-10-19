import cv2

import os
import unittest

import hanashi.processor.page_processor as processor
import hanashi.processor.image_helper as img_helper


def test_image(filename, expected_filename):
    masks, lines, rectangles = processor.process(filename)
    text, masks = processor.extract_text(masks)
    img = cv2.imread(filename, 0)
    final = processor.apply_masks(img, masks)
    expected = cv2.imread(expected_filename, 0)

    mse, ssim = img_helper.compare_images(cv2.imread(filename,0), expected)
    mse1, ssim1 = img_helper.compare_images(final, expected)
    mse_percentage = mse1 / mse *100
    return mse_percentage, len(masks)


class PageProcessorTest(unittest.TestCase):

    def test_onepunch(self):
        filename = os.path.join(os.path.dirname(__file__), 'resources/onepunch.jpg')
        expected_filename = os.path.join(os.path.dirname(__file__), 'resources/onepunch_expected.jpg')
        expected_mask_number = 6
        error_percentage, mask_number = test_image(filename, expected_filename)
        self.assertAlmostEqual(mask_number, expected_mask_number, delta=3)
        self.assertAlmostEqual(error_percentage, 0, delta=1)

    def test_onepiece(self):
        filename = os.path.join(os.path.dirname(__file__), 'resources/onepiece.jpg')
        expected_filename = os.path.join(os.path.dirname(__file__), 'resources/onepiece_expected.jpg')
        expected_mask_number = 10
        error_percentage, mask_number = test_image(filename, expected_filename)
        self.assertAlmostEqual(mask_number, expected_mask_number, delta=3)
        self.assertAlmostEqual(error_percentage, 0, delta=1)

    def test_image(self):
        filename = os.path.join(os.path.dirname(__file__), 'resources/image.jpg')
        expected_filename = os.path.join(os.path.dirname(__file__), 'resources/image_expected.png')
        expected_mask_number = 15
        error_percentage, mask_number = test_image(filename, expected_filename)
        self.assertAlmostEqual(mask_number, expected_mask_number, delta=3)
        self.assertAlmostEqual(error_percentage, 0, delta=1)