import unittest

from hanashi.model.ufarray import UFarray


class UFarrayTest(unittest.TestCase):
    def test_length(self):
        elements = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        uf = UFarray(len(elements))
        self.assertEqual(len(uf.P), len(elements))

    def test_union(self):
        elements = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        uf = UFarray(len(elements))
        unions = [(0, 2), (4, 3), (1, 3)]
        for u in unions:
            uf.setLabel(max(u[0], u[1]))
            uf.union(u[0], u[1])
        uf.flatten()
        root_dict = uf.root_dictionary()

        self.assertEqual(root_dict[0], [0, 2])
        self.assertEqual(root_dict[1], [1, 3, 4])
