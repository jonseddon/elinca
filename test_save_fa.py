"""
Basic unittests to check logic in save_fa.py
"""
import unittest
from save_fa import knots_to_beaufort


class TestKnotsToBeaufort(unittest.TestCase):
    def test_calm(self):
        self.assertEqual(0, knots_to_beaufort(0.5))

    def test_light_air_lowest(self):
        self.assertEqual(1, knots_to_beaufort(1.0))

    def test_light_air_mid(self):
        self.assertEqual(1, knots_to_beaufort(2.2))

    def test_light_breeze(self):
        self.assertEqual(2, knots_to_beaufort(4.))

    def test_gentle_breeze(self):
        self.assertEqual(3, knots_to_beaufort(7.))

    def test_moderate_breeze(self):
        self.assertEqual(4, knots_to_beaufort(11.))

    def test_fresh_breeze(self):
        self.assertEqual(5, knots_to_beaufort(17.))

    def test_strong_breeze(self):
        self.assertEqual(6, knots_to_beaufort(22.))

    def test_near_gale(self):
        self.assertEqual(7, knots_to_beaufort(28.))

    def test_gale(self):
        self.assertEqual(8, knots_to_beaufort(34.))

    def test_severe_gale(self):
        self.assertEqual(9, knots_to_beaufort(41.))

    def test_storm(self):
        self.assertEqual(10, knots_to_beaufort(48.))

    def test_violent_storm(self):
        self.assertEqual(11, knots_to_beaufort(56.))

    def test_huricane(self):
        self.assertEqual(12, knots_to_beaufort(64.))


if __name__ == '__main__':
    unittest.main()
