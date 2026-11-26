"""Numerical operations for the CNN teaching lab, independent of Streamlit."""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np


KERNELS = {
    "Edge detect (vertical)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "Edge detect (horizontal)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "Box blur": np.ones((3, 3), float) / 9,
    "Gaussian blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16,
    "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
    "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
}
PREPROCESSING = [
    "None", "Normalize 0–1", "Invert (255 − pixel)",
    "Threshold at 128", "Contrast × 1.6",
]
ACTIVATIONS = ["None", "ReLU", "Sigmoid", "Tanh"]
POOLING = ["None", "Max pooling", "Average pooling", "Min pooling"]
PLAYGROUND_OPERATIONS = {
    "Gaussian smoothing": "Reduce noise by replacing each pixel with a weighted average of its neighbors. "
                          "The center has the most weight; the 3 × 3 kernel weights sum to 1.",
    "Downsampling (nearest neighbor)": "Make the grid smaller by keeping every nth row and column. "
                                      "The factor n is the distance between retained pixels; no averaging is done.",
    "Max pooling": "Replace each window with its largest pixel value, keeping the brightest local feature.",
    "Min pooling": "Replace each window with its smallest pixel value, keeping the darkest local feature.",
    "Average pooling": "Replace each window with its mean pixel value, summarizing the local brightness.",
    "Normalize 0–1": "Rescale using (pixel − image minimum) / (image maximum − image minimum). "
                     "A constant image becomes zero. Outputs are floating-point values from 0 to 1.",
    "Invert (255 − pixel)": "Replace every pixel with 255 − pixel, swapping light and dark.",
}


@dataclass
class Layer:
    name: str = "Edge detector"
    preprocess: str = "None"
    kernel: str = "Edge detect (vertical)"
    padding: str = "same"
    stride: int = 1
    activation: str = "ReLU"
    pooling: str = "None"
    pool_size: int = 2
    pool_stride: int = 2
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Stage:
    layer: Layer
    incoming: np.ndarray
    preprocessed: np.ndarray
    padded: np.ndarray
    convolution: np.ndarray
    activated: np.ndarray
    output: np.ndarray


@dataclass
class OperationResult:
    operation: str
    incoming: np.ndarray
    output: np.ndarray
    original_bounds: tuple[int, int, int, int]
    window_size: int
    stride: int
    kernel: np.ndarray | None = None


def pad_window(
    a: np.ndarray, shape: tuple[int, int], stride: int, padding: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if stride < 1 or padding not in ("same", "valid"):
        raise ValueError("A window needs a positive stride and same/valid padding.")
    ph = pw = 0
    if padding == "same":
        oh, ow = ((dim + stride - 1) // stride for dim in a.shape)
        ph = max((oh - 1) * stride + shape[0] - a.shape[0], 0)
        pw = max((ow - 1) * stride + shape[1] - a.shape[1], 0)
    work = np.pad(a, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    return work, (ph // 2, pw // 2, a.shape[0], a.shape[1])


def make_samples(n: int = 16) -> dict[str, np.ndarray]:
    """Generate the same ten 8-bit patterns at any positive square size."""
    if n < 1:
        raise ValueError("Image size must be positive.")
    y, x = np.mgrid[:n, :n] * (16 / n)
    samples = {
        "Horizontal gradient": np.tile(np.linspace(0, 255, n), (n, 1)),
        "Vertical gradient": np.tile(np.linspace(0, 255, n)[:, None], (1, n)),
        "Checkerboard": ((x // 2 + y // 2) % 2) * 255,
        "Bright square": np.where((x > 3) & (x < 12) & (y > 3) & (y < 12), 245, 20),
        "Diagonal line": np.where(np.abs(x - y) <= 1, 255, 10),
        "Cross": np.where((np.abs(x - 7.5) <= 1) | (np.abs(y - 7.5) <= 1), 255, 15),
        "Circle": np.where((x - 7.5) ** 2 + (y - 7.5) ** 2 < 28, 240, 15),
        "Stripes": ((x // 2) % 2) * 220 + 15,
        "Corners": np.where(((x < 4) | (x > 11)) & ((y < 4) | (y > 11)), 255, 15),
        "Deterministic noise": np.random.default_rng(8).integers(0, 256, (n, n)),
    }
    return {name: image.astype(np.uint8) for name, image in samples.items()}


def preprocess(a: np.ndarray, method: str) -> np.ndarray:
    a = a.astype(float)
    if method == "Normalize 0–1":
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    if method == "Invert (255 − pixel)":
        return 255 - a
    if method == "Threshold at 128":
        return np.where(a >= 128, 255.0, 0.0)
    if method == "Contrast × 1.6":
        return np.clip((a - 128) * 1.6 + 128, 0, 255)
    if method != "None":
        raise ValueError(f"Unknown preprocessing: {method}")
    return a.copy()


def convolve(a: np.ndarray, k: np.ndarray, padding: str, stride: int) -> tuple[np.ndarray, np.ndarray]:
    kh, kw = k.shape
    work, _ = pad_window(a, (kh, kw), stride, padding)
    oh = (work.shape[0] - kh) // stride + 1
    ow = (work.shape[1] - kw) // stride + 1
    if oh < 1 or ow < 1:
        return np.empty((0, 0)), work
    windows = np.lib.stride_tricks.sliding_window_view(work, (kh, kw))[::stride, ::stride]
    return np.einsum("ijmn,mn->ij", windows, k), work


def activate(a: np.ndarray, kind: str) -> np.ndarray:
    if kind == "ReLU":
        return np.maximum(a, 0)
    if kind == "Sigmoid":
        return 1 / (1 + np.exp(-np.clip(a, -30, 30)))
    if kind == "Tanh":
        return np.tanh(a)
    if kind != "None":
        raise ValueError(f"Unknown activation: {kind}")
    return a


def pool(a: np.ndarray, kind: str, size: int, stride: int, padding: str = "valid") -> np.ndarray:
    if kind not in POOLING or size < 1 or stride < 1:
        raise ValueError("Pooling needs a known method and positive window/stride.")
    if kind == "None" or a.size == 0:
        return a
    work, _ = pad_window(a, (size, size), stride, padding)
    if min(work.shape) < size:
        return np.empty((0, 0))
    windows = np.lib.stride_tricks.sliding_window_view(work, (size, size))[::stride, ::stride]
    if kind == "Max pooling":
        return windows.max(axis=(-2, -1))
    if kind == "Min pooling":
        return windows.min(axis=(-2, -1))
    return windows.mean(axis=(-2, -1))


def apply_operation(
    image: np.ndarray, operation: str, *, size: int = 2,
    stride: int = 1, padding: str = "valid", factor: int = 2,
) -> OperationResult:
    """Keep the displayed windows and their computed outputs on the same coordinates."""
    if operation not in PLAYGROUND_OPERATIONS:
        raise ValueError(f"Unknown playground operation: {operation}")
    source = image.astype(float)
    kernel = None
    if operation == "Gaussian smoothing":
        kernel = KERNELS["Gaussian blur"]
        size = 3
        output, work = convolve(source, kernel, padding, stride)
        _, bounds = pad_window(source, (size, size), stride, padding)
    elif operation in POOLING:
        output = pool(source, operation, size, stride, padding)
        work, bounds = pad_window(source, (size, size), stride, padding)
    else:
        work = source
        bounds = (0, 0, *source.shape)
        size = 1
        stride = 1
        if operation == "Downsampling (nearest neighbor)":
            if factor < 1:
                raise ValueError("Downsampling factor must be positive.")
            stride = factor
            output = source[::factor, ::factor]
        else:
            output = preprocess(source, operation)
    return OperationResult(operation, work, output, bounds, size, stride, kernel)


def default_network() -> list[Layer]:
    return [
        Layer(name="Smooth noise", kernel="Gaussian blur", activation="None"),
        Layer(name="Find vertical edges"),
        Layer(name="Refine edges", kernel="Sharpen"),
        Layer(name="Keep strong features", kernel="Identity", pooling="Max pooling"),
        Layer(name="Smooth features", kernel="Gaussian blur", activation="None"),
        Layer(name="Find horizontal edges", kernel="Edge detect (horizontal)"),
        Layer(name="Find local texture", kernel="Emboss"),
        Layer(name="Summarize regions", kernel="Identity", pooling="Average pooling"),
        Layer(name="Refine small features", kernel="Sharpen"),
        Layer(name="Final feature map", kernel="Box blur", activation="None"),
    ]


def run_network(image: np.ndarray, layers: list[Layer]) -> list[Stage]:
    stages = []
    source = image.astype(float)
    for layer in layers:
        if source.size == 0:
            break
        pre = preprocess(source, layer.preprocess)
        conv, padded = convolve(pre, KERNELS[layer.kernel], layer.padding, layer.stride)
        act = activate(conv, layer.activation)
        output = pool(act, layer.pooling, layer.pool_size, layer.pool_stride)
        stages.append(Stage(layer, source, pre, padded, conv, act, output))
        source = output
    return stages
