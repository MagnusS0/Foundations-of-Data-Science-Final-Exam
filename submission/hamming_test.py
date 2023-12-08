import unittest
import numpy as np
from hamming_code import Hamming, Support


class TestSupport(unittest.TestCase):

    def test_create_random_vector(self):
        vector = Support.create_random_vector()
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), 4)
        self.assertTrue(np.all(np.isin(vector, [0, 1])))

    def test_check_input(self):
        # Test with valid integer
        self.assertTrue(np.array_equal(Support.check_input(1010, 4), np.array([1, 0, 1, 0])))

        # Test with valid list
        self.assertTrue(np.array_equal(Support.check_input([1, 0, 1, 0], 4), np.array([1, 0, 1, 0])))

        # Test with valid string
        self.assertTrue(np.array_equal(Support.check_input("1010", 4), np.array([1, 0, 1, 0])))

        # Test with invalid input type
        with self.assertRaises(TypeError):
            Support.check_input({1, 0, 1, 0}, 4)

        # Test with invalid length
        with self.assertRaises(ValueError):
            Support.check_input("10101", 4)

        # Test with invalid elements
        with self.assertRaises(ValueError):
            Support.check_input("12", 2)

    def test_bitflip_rand(self):
        # Test with valid input and bit flip count
        vector = np.array([1, 0, 1, 0, 1, 0, 1])
        flipped = Support.bitflip_rand(np.array([1, 0, 1, 0, 1, 0, 1]), 1)
        self.assertEqual(len(flipped), 7)
        # Check if the flipped vector is different from the original
        self.assertFalse(np.array_equal(vector, flipped))
        # Additionally, you can check if exactly one bit was flipped
        diff = np.sum(np.abs(vector - flipped))
        self.assertEqual(diff, 1)

        # Test flipping more than 2 bits
        with self.assertRaises(ValueError):
            Support.bitflip_rand(vector, 3)

    def test_bitflip_specific(self):
        vector = np.array([1, 0, 1, 0, 1, 0, 1])
        flipped = Support.bitflip_specific(np.array([1, 0, 1, 0, 1, 0, 1]), 0)
        self.assertEqual(len(flipped), 7)
        self.assertNotEqual(vector[0], flipped[0])

        # Test flipping a bit out of range
        with self.assertRaises(ValueError):
            Support.bitflip_specific(vector, 7)

class TestHamming(unittest.TestCase):

    def test_encoder(self):
        input_vector = np.array([1, 0, 1, 0])
        codeword = Hamming.encoder(input_vector)
        self.assertEqual(len(codeword), 7)

    def test_parity_check(self):
        codeword = np.array([1, 0, 1, 0, 1, 0, 1])
        # Test with no errors
        Hamming.parity_check(codeword)

    def test_decoder(self):
        codeword = np.array([1, 0, 1, 0, 1, 0, 1])
        # Test decoding a correct codeword
        Hamming.decoder(codeword)

if __name__ == '__main__':
    unittest.main()