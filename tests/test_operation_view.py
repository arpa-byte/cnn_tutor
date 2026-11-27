import json
from pathlib import Path
import subprocess
import unittest

import numpy as np

from image_processing import KERNELS, Layer, apply_operation, run_network
from operation_view import network_html, operation_html


def player_data(html):
    return json.loads(html.split("const lesson = ", 1)[1].split(";\nconst $", 1)[0])


class NetworkPlaybackTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(16, dtype=float).reshape(4, 4)
        self.layers = [
            Layer(name="Edges", preprocess="Normalize 0–1", activation="ReLU",
                  pooling="Max pooling"),
            Layer(name="Smooth", kernel="Gaussian blur", activation="None"),
        ]
        self.stages = run_network(self.image, self.layers)

    def test_frames_follow_calculated_operations_and_layer_handoff(self):
        data = player_data(network_html(self.stages, True, False, self.layers[1].id))
        self.assertEqual(data["selectedLayer"], 1)
        self.assertEqual([frame["step"] for frame in data["frames"]],
                         ["Preprocessing", "Convolution", "Activation", "Pooling", "Convolution"])
        self.assertEqual([(layer["start"], layer["end"]) for layer in data["layers"]],
                         [(0, 3), (4, 4)])
        for frame, expected in zip(data["frames"], [
            self.stages[0].preprocessed, self.stages[0].convolution,
            self.stages[0].activated, self.stages[0].output, self.stages[1].output,
        ]):
            np.testing.assert_allclose(frame["output"], expected)
        next_input = np.array(data["frames"][4]["input"])
        np.testing.assert_allclose(next_input[1:-1, 1:-1], data["frames"][3]["output"])
        self.assertEqual(data["frames"][4]["bounds"], [1, 1, 2, 2])

    def test_stride_and_asymmetric_padding_match_window_coordinates(self):
        layer = Layer(kernel="Sharpen", stride=2, activation="None")
        stage = run_network(self.image, [layer])[0]
        frame = player_data(network_html([stage], False, True))["frames"][0]
        self.assertEqual(frame["bounds"], [0, 0, 4, 4])
        padded = np.array(frame["input"])
        for row, col in np.ndindex(stage.output.shape):
            patch = padded[row * 2:row * 2 + 3, col * 2:col * 2 + 3]
            self.assertAlmostEqual(float((patch * KERNELS["Sharpen"]).sum()),
                                   frame["output"][row][col])

    def test_empty_operation_is_a_terminal_frame(self):
        for layer in [
            Layer(padding="valid", activation="None"),
            Layer(pooling="Max pooling", pool_size=4),
        ]:
            with self.subTest(layer=layer):
                stages = run_network(np.ones((2, 2)), [layer, Layer()])
                data = player_data(network_html(stages, False, False))
                self.assertEqual(len(data["layers"]), 1)
                self.assertEqual(data["frames"][-1]["output"], [])
                self.assertIn("Later layers have not run", data["frames"][-1]["error"])

    def test_custom_layer_names_cannot_end_embedded_script(self):
        self.layers[0].name = "</script><h1>Example</h1>"
        html = network_html(self.stages, True, False)
        self.assertNotIn(self.layers[0].name, html)
        self.assertEqual(player_data(html)["layers"][0]["name"], self.layers[0].name)

    def run_controls(self, html, scenario):
        script = Path(__file__).with_name("player_controls.js")
        result = subprocess.run(["node", str(script), scenario], input=html, text=True,
                                capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_network_controls_and_layer_transitions(self):
        self.run_controls(network_html(self.stages, True, True, self.layers[1].id), "network")

    def test_empty_frame_stops_playback_without_canvas_errors(self):
        stages = run_network(self.image, [
            Layer(pooling="Max pooling", pool_size=4), Layer(padding="valid"), Layer(),
        ])
        self.run_controls(network_html(stages, True, False), "empty")

    def test_playground_controls_still_work(self):
        result = apply_operation(self.image, "Gaussian smoothing", padding="same")
        self.run_controls(operation_html(result, True, False), "playground")

    def test_heatmap_colors_filter_display_and_playback(self):
        image = np.array([[-1, 1, -1, 1, -1], [0, -1, 0, 1, 0], [-1, 1, -1, 1, -1]], float)
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 9
        result = apply_operation(
            image, "Convolution for pattern recognition", kernel=kernel,
        )
        self.run_controls(operation_html(result, True, True), "heatmap")


if __name__ == "__main__":
    unittest.main()
