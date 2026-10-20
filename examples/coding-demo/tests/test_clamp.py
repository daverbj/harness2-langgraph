import unittest

from clamp import clamp


class ClampTests(unittest.TestCase):
    def test_below_minimum(self):
        self.assertEqual(clamp(-3, 0, 10), 0)

    def test_above_maximum(self):
        self.assertEqual(clamp(20, 0, 10), 10)

    def test_inside_range(self):
        self.assertEqual(clamp(4, 0, 10), 4)

    def test_inclusive_endpoints(self):
        self.assertEqual(clamp(0, 0, 10), 0)
        self.assertEqual(clamp(10, 0, 10), 10)

    def test_reversed_bounds(self):
        with self.assertRaises(ValueError):
            clamp(4, 10, 0)
