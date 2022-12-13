import unittest


class MockTest(unittest.TestCase):
    def test_mock(self):
        x = 1
        y = 2
        assert x + y == 3
