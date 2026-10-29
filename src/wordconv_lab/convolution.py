"""Explicit local feature operators.

The implementation uses the cross-correlation convention commonly called
convolution in neural-network libraries. Kernels are randomly initialized and
UNTRAINED. A high demo score is not evidence of linguistic or aesthetic quality.
"""
from __future__ import annotations

import numpy as np


def _finite_array(value: object, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or any(size <= 0 for size in array.shape):
        raise ValueError(f"{name} must be nonempty with {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def horizontal_convolution(
    x: np.ndarray, kernel: np.ndarray, bias: np.ndarray | None = None
) -> np.ndarray:
    """Apply an ordered shared kernel independently to each line.

    x: (lines, positions, features). kernel: (odd_width, features, outputs).
    Zero padding occurs at each LINE boundary, never at a flattened poem boundary.
    """
    x = _finite_array(x, 3, "x")
    kernel = _finite_array(kernel, 3, "kernel")
    width, feature_dim, outputs = kernel.shape
    if width % 2 != 1 or feature_dim != x.shape[2]:
        raise ValueError("kernel width must be odd and its feature dimension must match")
    if bias is None:
        bias = np.zeros(outputs)
    bias = _finite_array(bias, 1, "bias")
    if bias.shape != (outputs,):
        raise ValueError("bias dimension must match outputs")
    radius = width // 2
    padded = np.pad(x, ((0, 0), (radius, radius), (0, 0)))
    result = np.empty((x.shape[0], x.shape[1], outputs))
    for position in range(x.shape[1]):
        window = padded[:, position : position + width, :]
        result[:, position, :] = np.einsum("lwf,wfo->lo", window, kernel) + bias
    return result


class LocalConvScorer:
    """Untrained horizontal + same-couplet aligned feature scorer.

    score_parts exposes the two contributions so the demo can be inspected.
    Calling code must use exactly the same candidate set and legality filter for
    every compared scorer. This class does not enforce poetic constraints.
    """

    trained = False
    status = "random_untrained_engineering_only"

    def __init__(
        self, feature_dim: int, channels: int = 4, kernel_width: int = 3, seed: int = 0
    ) -> None:
        if feature_dim <= 0 or channels <= 0 or kernel_width <= 0 or kernel_width % 2 != 1:
            raise ValueError("dimensions must be positive and kernel width must be odd")
        self.feature_dim = feature_dim
        self.channels = channels
        self.seed = seed
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(feature_dim * kernel_width)
        self.horizontal_kernel = rng.normal(0, scale, (kernel_width, feature_dim, channels))
        self.pair_kernel = rng.normal(0, 1 / np.sqrt(2 * feature_dim), (2, feature_dim, channels))
        self.horizontal_readout = rng.normal(0, 1 / np.sqrt(channels), channels)
        self.pair_readout = rng.normal(0, 1 / np.sqrt(channels), channels)

    def _validate(self, x: np.ndarray) -> np.ndarray:
        x = _finite_array(x, 3, "x")
        if x.shape[0] != 2 or x.shape[2] != self.feature_dim:
            raise ValueError("scorer expects exactly one couplet and its configured feature dimension")
        return x

    def score_parts(self, x: np.ndarray) -> dict[str, float]:
        x = self._validate(x)
        horizontal = np.tanh(horizontal_convolution(x, self.horizontal_kernel))
        # Only corresponding positions within THIS couplet are mixed vertically.
        paired = np.tanh(np.einsum("lpf,lfo->po", x, self.pair_kernel))
        horizontal_score = float(horizontal.mean(axis=(0, 1)) @ self.horizontal_readout)
        pair_score = float(paired.mean(axis=0) @ self.pair_readout)
        return {
            "horizontal": horizontal_score,
            "aligned_pair": pair_score,
            "total": horizontal_score + pair_score,
        }

    def score(self, x: np.ndarray) -> float:
        return self.score_parts(x)["total"]

    def parameter_count(self) -> int:
        return sum(array.size for array in (
            self.horizontal_kernel, self.pair_kernel,
            self.horizontal_readout, self.pair_readout,
        ))
