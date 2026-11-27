import unittest

from image_processing import ArchitectureLayer, output_resolution, random_architecture


class ArchitecturePracticeTests(unittest.TestCase):
    def test_convolution_uses_its_filter_count_for_output_channels(self):
        layer = ArchitectureLayer("Convolution", 3, 2, "VALID", output_channels=8)
        self.assertEqual(output_resolution(28, 30, 3, layer), (13, 14, 8))

    def test_pooling_preserves_channels(self):
        layer = ArchitectureLayer("Max pooling", 2, 2, "SAME")
        self.assertEqual(output_resolution(13, 14, 8, layer), (7, 7, 8))

    def test_generated_network_has_valid_successive_shapes(self):
        input_shape, layers = random_architecture(seed=7)
        current = input_shape
        for layer in layers:
            current = output_resolution(*current, layer)
            self.assertGreaterEqual(min(current), 1)


if __name__ == "__main__":
    unittest.main()
