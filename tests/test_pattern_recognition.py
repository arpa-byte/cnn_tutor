import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

from image_processing import PATTERN_FILTER, apply_operation, pattern_input
from test_operation_view import player_data


OPERATION = "Convolution for pattern recognition"


class PatternRecognitionTests(unittest.TestCase):
    def test_slide_example_heatmap(self):
        result = apply_operation(pattern_input(), OPERATION, kernel=PATTERN_FILTER)
        expected = np.array([
            [7, -1, 1, 3, 5, -1, 3],
            [-1, 9, -1, 3, -1, 1, -1],
            [1, -1, 9, -3, 1, -1, 5],
            [3, 3, -3, 5, -3, 3, 3],
            [5, -1, 1, -3, 9, -1, 1],
            [-1, 1, -1, 3, -1, 9, -1],
            [3, -1, 5, 3, 1, -1, 7],
        ]) / 9
        np.testing.assert_allclose(result.output, expected)

    def test_custom_filter_stride_padding_and_raw_sum(self):
        image = pattern_input()
        kernel = np.array([[.5, -2, 3], [0, 1, -.25], [-1, 0, 2]])
        averaged = apply_operation(image, OPERATION, kernel=kernel, stride=2, padding="same")
        raw = apply_operation(image, OPERATION, kernel=kernel, stride=2, padding="same",
                              average_products=False)
        np.testing.assert_allclose(raw.output, averaged.output * 9)
        self.assertEqual(raw.output.shape, (5, 5))
        self.assertEqual(raw.original_bounds, (1, 1, 9, 9))
        for row, col in np.ndindex(raw.output.shape):
            patch = raw.incoming[row * 2:row * 2 + 3, col * 2:col * 2 + 3]
            self.assertAlmostEqual(raw.output[row, col], float((patch * kernel).sum()))

    def test_invalid_filter_is_rejected(self):
        for kernel in [None, np.ones((2, 3)), np.full((3, 3), np.nan)]:
            with self.subTest(kernel=kernel):
                with self.assertRaises(ValueError):
                    apply_operation(pattern_input(), OPERATION, kernel=kernel)

    def test_edit_randomize_and_run_apply_only_submitted_filter(self):
        app = AppTest.from_string("""
from cnn_teaching_lab import playground
from image_processing import make_samples
playground(make_samples(8)["Horizontal gradient"], True, 25, False)
""", default_timeout=30).run()
        app.selectbox[0].select(OPERATION).run()
        self.assertFalse(app.exception)
        initial = player_data(app.get("iframe")[0].proto.srcdoc)
        self.assertEqual(len(initial["frames"][0]["output"]), 7)
        app.number_input(key="pattern_weight_0_0").set_value(2.5)
        self.assertEqual(player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]["kernel"][0][0], 1)
        next(button for button in app.button if button.label == "Run convolution").click().run()
        self.assertFalse(app.exception)
        applied = player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]
        self.assertEqual(applied["kernel"][0][0], 2.5)
        first_run = applied["runId"]
        next(button for button in app.button if button.label == "Run convolution").click().run()
        self.assertGreater(player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]["runId"], first_run)
        app.button(key="pattern_randomize").click().run()
        self.assertFalse(app.exception)
        random_weights = np.array([widget.value for widget in app.number_input]).reshape(3, 3)
        self.assertTrue(np.isin(random_weights, [-1, 0, 1]).all())
        self.assertEqual(player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]["kernel"][0][0], 2.5)
        next(button for button in app.button if button.label == "Run convolution").click().run()
        np.testing.assert_array_equal(
            player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]["kernel"], random_weights,
        )
        app.selectbox[1].select("Sidebar image (threshold at 128)")
        app.toggle(key="pattern_padding").set_value(True)
        app.selectbox(key="pattern_stride").select(2)
        app.checkbox(key="pattern_average").set_value(False)
        next(button for button in app.button if button.label == "Run convolution").click().run()
        self.assertFalse(app.exception)
        frame = player_data(app.get("iframe")[0].proto.srcdoc)["frames"][0]
        self.assertEqual(frame["divisor"], 1)
        self.assertEqual(frame["stride"], 2)
        self.assertEqual(frame["bounds"], [0, 0, 8, 8])
        self.assertEqual(len(frame["output"]), 4)
        app.toggle(key="lab_show_only_output").set_value(True).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("iframe")), 0)
        self.assertEqual(len(app.get("image")), 2)


if __name__ == "__main__":
    unittest.main()
