import unittest

from hanashi.model.rectangle import Rectangle


class RectangleTest(unittest.TestCase):
    def test_contains(self):
        rect1 = Rectangle(0, 0, 50, 50)
        rect2 = Rectangle(10, 10, 20, 20)

        self.assertTrue(rect2 in rect1)
        self.assertFalse(rect1 in rect2)

    def test_contains_with_overlapping_rectangles(self):
        rect1 = Rectangle(0, 0, 50, 50)
        rect2 = Rectangle(10, 40, 20, 20)
        border_rect = Rectangle(10, 40, 10, 10)

        self.assertFalse(rect2 in rect1)
        self.assertTrue(border_rect in rect1)

    def test_overlap(self):
        rect1 = Rectangle(0, 0, 50, 50)
        rect2 = Rectangle(10, 10, 100, 20)
        self.assertTrue(rect2.overlaps_with(rect1))
