import unittest

from hanashi.model.quadtree import Quadtree
from hanashi.model.rectangle import Rectangle


class QuadtreeTest(unittest.TestCase):
    def test_get_index(self):
        bounds = Rectangle(0, 0, 400, 400)
        quadtree = Quadtree(0, bounds)

        rect1 = Rectangle(100, 10, 10, 10)
        rect2 = Rectangle(190, 10, 20, 20)

        quadtree.insert(rect1)
        quadtree.insert(rect2)
        self.assertEqual(1, quadtree.get_index(rect1))
        self.assertEqual(-1, quadtree.get_index(rect2))

    def test_retrieve(self):
        bounds = Rectangle(0, 0, 1000, 1000)
        quadtree = Quadtree(0, bounds)
        for x in range(10):
            for y in range(10):
                quadtree.insert(Rectangle(x * 20, y * 20, 10, 10))

        rect3 = Rectangle(150, 10, 10, 10)

        rectangles = []
        quadtree.retrieve(rectangles, rect3)

        self.assertEqual(len(rectangles), 33)
