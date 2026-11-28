import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

from app.image_processing import EDGE_FILTERS, apply_operation, make_samples
from test_operation_view import player_data


class EdgeDetectionTests(unittest.TestCase):
    def test_default_detectors_find_edges_and_preserve_sign(self):
        image = np.tile([0, 0, 255, 255, 255], (5, 1)).astype(np.uint8)
        expected = {
            "Edge detection": np.tile([-765, 765, 0], (3, 1)),
            "Vertical edge detection": np.tile([1020, 1020, 0], (3, 1)),
        }
        for operation in EDGE_FILTERS:
            with self.subTest(operation=operation):
                result = apply_operation(image, operation)
                np.testing.assert_allclose(result.output, expected[operation])
                reversed_result = apply_operation(255 - image, operation)
                np.testing.assert_allclose(reversed_result.output, -result.output)
                constant = apply_operation(np.full((5, 5), 100), operation)
                np.testing.assert_array_equal(constant.output, np.zeros((3, 3)))

    def test_custom_kernel_stride_and_padding_use_raw_pixels_and_sum(self):
        image = make_samples(8)["Horizontal gradient"]
        kernel = np.array([[.25, -1, 2], [0, -2, 1], [-1, 0, .5]])
        for operation in EDGE_FILTERS:
            with self.subTest(operation=operation):
                result = apply_operation(image, operation, kernel=kernel, padding="same", stride=2)
                self.assertEqual(result.output.shape, (4, 4))
                self.assertEqual(result.divisor, 1)
                self.assertFalse(result.bipolar_input)
                np.testing.assert_array_equal(result.incoming[:8, :8], image)
                for row, col in np.ndindex(result.output.shape):
                    patch = result.incoming[row * 2:row * 2 + 3, col * 2:col * 2 + 3]
                    self.assertAlmostEqual(result.output[row, col], float((patch * kernel).sum()))

    def test_editable_detectors_keep_separate_applied_filters(self):
        app = AppTest.from_string("""
from app.cnn_teaching_lab import playground
from app.image_processing import make_samples
playground(make_samples(8)["Horizontal gradient"], False, 25, False)
""", default_timeout=30).run()
        applied_kernels = {}
        for operation, prefix in [("Edge detection", "edge"), ("Vertical edge detection", "vertical_edge")]:
            app.selectbox[0].select(operation).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.number_input), 9)
            self.assertEqual(len(app.selectbox), 2)
            frame = player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]
            np.testing.assert_array_equal(frame["kernel"], EDGE_FILTERS[operation])
            np.testing.assert_array_equal(frame["input"], make_samples(8)["Horizontal gradient"])
            self.assertTrue(frame["heatmap"])
            self.assertFalse(frame["bipolarInput"])
            app.number_input(key=f"{prefix}_weight_0_0").set_value(2.5)
            np.testing.assert_array_equal(
                player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]["kernel"], EDGE_FILTERS[operation],
            )
            next(button for button in app.button if button.label == "Run convolution").click().run()
            self.assertFalse(app.exception)
            frame = player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]
            self.assertEqual(frame["kernel"][0][0], 2.5)
            self.assertEqual(frame["divisor"], 1)
            applied_kernels[operation] = frame["kernel"]
        for operation in EDGE_FILTERS:
            app.selectbox[0].select(operation).run()
            self.assertFalse(app.exception)
            np.testing.assert_array_equal(
                player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]["kernel"], applied_kernels[operation],
            )
        app.toggle(key="lab_show_only_output").set_value(True).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("iframe")), 0)
        self.assertEqual(len(app.get("image")), 2)


if __name__ == "__main__":
    unittest.main()
