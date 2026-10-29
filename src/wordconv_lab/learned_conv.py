"""Trainable NumPy version of the declared local couplet objective.

This module learns source-restoration rankings, not a literary-quality label.
Features are a character embedding plus FOUR FIXED task/position features. No
feature depends on a completed candidate's distant output, so width-three
models remain compatible with the exact local-factor solver after training.

Input ids have shape (batch, candidates, two lines, line length). Vocabulary
identity and train-only construction are caller responsibilities; id 0 is the
frozen zero unknown embedding. Width one provides the predeclared near-matched
non-horizontal-locality ablation, retaining aligned upper/lower interactions.
"""
from __future__ import annotations

import json
import math
from numbers import Integral
from pathlib import Path
from typing import Mapping

import numpy as np

from .convolution import LocalConvScorer


class LearnedLocalScorer:
    embedding_dim = 8
    feature_dim = 12
    format_version = "versematrix.learned-local-scorer.v1"

    def __init__(self, vocab_size: int, kernel_width: int = 3,
                 channels: int = 12, seed: int = 0):
        for value, label in ((vocab_size, "vocab_size"), (channels, "channels")):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if isinstance(kernel_width, bool) or kernel_width not in (1, 3):
            raise ValueError("only the declared width-one/three comparisons are supported")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.vocab_size, self.channels = int(vocab_size), int(channels)
        self.kernel_width, self.seed = int(kernel_width), int(seed)
        # Identical vocabulary and seed give identical starting embeddings in
        # both arms, regardless of their core width/channel geometry.
        embedding = np.random.default_rng(seed).normal(0.0, 0.1, (vocab_size, self.embedding_dim))
        embedding[0] = 0.0
        base = LocalConvScorer(self.feature_dim, channels, kernel_width, seed=seed + 1)
        self.params = {"embedding": embedding}
        for name in ("horizontal_kernel", "pair_kernel", "horizontal_readout", "pair_readout"):
            self.params[name] = np.array(getattr(base, name), copy=True)
        self.adam_m = {name: np.zeros_like(array) for name, array in self.params.items()}
        self.adam_v = {name: np.zeros_like(array) for name, array in self.params.items()}
        self.step_count = 0
        self.last_optimizer = None

    def _shapes(self):
        return {
            "embedding": (self.vocab_size, self.embedding_dim),
            "horizontal_kernel": (self.kernel_width, self.feature_dim, self.channels),
            "pair_kernel": (2, self.feature_dim, self.channels),
            "horizontal_readout": (self.channels,),
            "pair_readout": (self.channels,),
        }

    def _check_parameters(self, *, optimizer=False):
        if set(self.params) != set(self._shapes()):
            raise ValueError("parameter names disagree with declared architecture")
        collections = [("parameter", self.params)]
        if optimizer:
            collections.extend((("first moment", self.adam_m), ("second moment", self.adam_v)))
        for label, values in collections:
            if set(values) != set(self._shapes()):
                raise ValueError(f"invalid {label} keys")
            for name, shape in self._shapes().items():
                array = values[name]
                if not isinstance(array, np.ndarray) or array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"invalid finite {label}: {name}")
            if np.any(values["embedding"][0] != 0.0):
                raise ValueError(f"unknown embedding {label} must remain exactly zero")
        if optimizer and any(np.any(v < 0) for v in self.adam_v.values()):
            raise ValueError("Adam second moments cannot be negative")
        if isinstance(self.step_count, bool) or not isinstance(self.step_count, Integral) or self.step_count < 0:
            raise ValueError("invalid optimizer step count")

    @staticmethod
    def _position_features(length, mask_start):
        if length not in (5, 7):
            raise ValueError("the restoration task supports five/seven-character lines")
        if isinstance(mask_start, bool) or not isinstance(mask_start, Integral) or not 0 <= mask_start <= length - 2:
            raise ValueError("mask_start must locate a complete two-character lower-line span")
        features = np.zeros((2, length, 4), dtype=np.float64)
        features[1, :, 0] = 1.0
        features[:, :, 1] = np.arange(length) / (length - 1)
        features[:, :, 2] = (length - 5) / 2
        features[1, mask_start:mask_start + 2, 3] = 1.0
        return features

    def _validate_ids(self, ids):
        ids = np.asarray(ids)
        if ids.ndim != 4 or ids.shape[0] < 1 or ids.shape[1] < 1 or ids.shape[2] != 2:
            raise ValueError("ids must have shape (batch, candidates, 2, length)")
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= self.vocab_size):
            raise ValueError("character ids must be integers in the declared vocabulary")
        return ids

    def _forward(self, ids, mask_start=0, *, keep_cache=False):
        self._check_parameters()
        ids = self._validate_ids(ids)
        batch, choices, _, length = ids.shape
        positions = self._position_features(length, mask_start)
        flat_ids = ids.reshape(-1, 2, length)
        x = np.empty((*flat_ids.shape, self.feature_dim), dtype=np.float64)
        x[..., :self.embedding_dim] = self.params["embedding"][flat_ids]
        x[..., self.embedding_dim:] = positions
        radius = self.kernel_width // 2
        # Padding is all-zero feature vectors, including position channels.
        padded = np.pad(x, ((0, 0), (0, 0), (radius, radius), (0, 0)))
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                pre_h = np.zeros((*flat_ids.shape, self.channels), dtype=np.float64)
                for offset in range(self.kernel_width):
                    pre_h += padded[:, :, offset:offset + length, :] @ self.params["horizontal_kernel"][offset]
                h = np.tanh(pre_h)
                paired = np.tanh(x[:, 0] @ self.params["pair_kernel"][0]
                                 + x[:, 1] @ self.params["pair_kernel"][1])
                h_mean = h.mean(axis=(1, 2))
                p_mean = paired.mean(axis=1)
                scores = h_mean @ self.params["horizontal_readout"] + p_mean @ self.params["pair_readout"]
        except FloatingPointError as exc:
            raise ValueError("non-finite arithmetic in local scorer forward pass") from exc
        if not np.isfinite(scores).all():
            raise ValueError("local scorer produced non-finite scores")
        cache = (flat_ids, x, padded, h, paired, h_mean, p_mean) if keep_cache else None
        return scores.reshape(batch, choices), cache

    def score_batch(self, ids, mask_start=0):
        """Return a (batch, candidates) score array without changing model state."""
        return self._forward(ids, mask_start)[0]

    def loss_and_gradients(self, ids, gold_indices, mask_start=0):
        """Mean listwise source-restoration NLL and exact manual derivatives.

        All supplied candidates participate; a hidden target absent from a pool
        has no legal gold index and must not be silently injected by this API.
        The loss excludes decoupled weight decay, which belongs to train_step.
        """
        scores, cache = self._forward(ids, mask_start, keep_cache=True)
        batch, choices = scores.shape
        gold = np.asarray(gold_indices)
        if gold.shape != (batch,) or gold.dtype.kind not in "iu" or np.any(gold < 0) or np.any(gold >= choices):
            raise ValueError("gold_indices must identify one supplied candidate per batch item")
        flat_ids, x, padded, h, paired, h_mean, p_mean = cache
        length = flat_ids.shape[-1]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                centered = scores - scores.max(axis=1, keepdims=True)
                exponential = np.exp(centered)
                denominator = exponential.sum(axis=1, keepdims=True)
                loss = float(np.mean(np.log(denominator[:, 0]) - centered[np.arange(batch), gold]))
                ds = exponential / denominator
                ds[np.arange(batch), gold] -= 1.0
                ds = ds.reshape(-1) / batch
                gradients = {name: np.zeros_like(array) for name, array in self.params.items()}
                gradients["horizontal_readout"] = h_mean.T @ ds
                gradients["pair_readout"] = p_mean.T @ ds
                dh = ds[:, None, None, None] * self.params["horizontal_readout"] * (1.0 - h * h) / (2 * length)
                dp = ds[:, None, None] * self.params["pair_readout"] * (1.0 - paired * paired) / length
                dx_padded = np.zeros_like(padded)
                dh_flat = dh.reshape(-1, self.channels)
                for offset in range(self.kernel_width):
                    window = padded[:, :, offset:offset + length, :]
                    gradients["horizontal_kernel"][offset] = window.reshape(-1, self.feature_dim).T @ dh_flat
                    dx_padded[:, :, offset:offset + length, :] += dh @ self.params["horizontal_kernel"][offset].T
                radius = self.kernel_width // 2
                dx = dx_padded[:, :, radius:radius + length, :]
                dp_flat = dp.reshape(-1, self.channels)
                for row in (0, 1):
                    gradients["pair_kernel"][row] = x[:, row].reshape(-1, self.feature_dim).T @ dp_flat
                    dx[:, row] += dp @ self.params["pair_kernel"][row].T
                np.add.at(gradients["embedding"], flat_ids.reshape(-1), dx[..., :self.embedding_dim].reshape(-1, self.embedding_dim))
                gradients["embedding"][0] = 0.0
        except FloatingPointError as exc:
            raise ValueError("non-finite arithmetic in ranking loss or gradient") from exc
        if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
            raise ValueError("non-finite ranking loss or gradient")
        return loss, gradients

    def train_step(self, ids, gold_indices, mask_start=0, *, learning_rate=3e-3,
                   weight_decay=1e-4, max_grad_norm=5.0):
        """One Adam update with decoupled decay and optional global-norm clipping.

        beta1=.9, beta2=.999, epsilon=1e-8. New parameters/moments are validated
        before any update commits, so a numeric failure cannot half-update the
        model. Unknown id 0 remains fixed in parameters and both moments.
        """
        for value, label in ((learning_rate, "learning_rate"), (weight_decay, "weight_decay")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if max_grad_norm is not None and (isinstance(max_grad_norm, bool)
                or not isinstance(max_grad_norm, (int, float))
                or not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive finite or None")
        self._check_parameters(optimizer=True)
        loss, gradients = self.loss_and_gradients(ids, gold_indices, mask_start)
        step = self.step_count + 1
        new_params, new_m, new_v = {}, {}, {}
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                scale = min(1.0, max_grad_norm / norm) if max_grad_norm is not None and norm > 0 else 1.0
                for name, old in self.params.items():
                    g = gradients[name] * scale
                    m = 0.9 * self.adam_m[name] + 0.1 * g
                    v = 0.999 * self.adam_v[name] + 0.001 * g * g
                    corrected_m = m / (1.0 - 0.9 ** step)
                    corrected_v = v / (1.0 - 0.999 ** step)
                    updated = old - learning_rate * (corrected_m / (np.sqrt(corrected_v) + 1e-8) + weight_decay * old)
                    if name == "embedding":
                        updated[0] = 0.0
                        m[0] = 0.0
                        v[0] = 0.0
                    if not all(np.isfinite(array).all() for array in (updated, m, v)):
                        raise ValueError("non-finite Adam parameter or state")
                    new_params[name], new_m[name], new_v[name] = updated, m, v
        except FloatingPointError as exc:
            raise ValueError("non-finite arithmetic in Adam update") from exc
        if not math.isfinite(norm):
            raise ValueError("non-finite gradient norm")
        self.params, self.adam_m, self.adam_v = new_params, new_m, new_v
        self.step_count = step
        self.last_optimizer = {"name": "Adam_with_decoupled_weight_decay", "learning_rate": learning_rate,
                               "weight_decay": weight_decay, "max_grad_norm": max_grad_norm,
                               "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
        return {"loss": loss, "gradient_norm": norm, "clip_scale": scale, "step": step}

    def parameter_count(self):
        """Stored scalar parameters, including the eight fixed unknown zeros."""
        return sum(value.size for value in self.params.values())

    def trainable_parameter_count(self):
        return self.parameter_count() - self.embedding_dim

    def core_parameter_count(self):
        return sum(value.size for name, value in self.params.items() if name != "embedding")

    def as_local_scorer(self):
        """Snapshot trained core weights in the original scalar scoring API."""
        self._check_parameters()
        model = LocalConvScorer(self.feature_dim, self.channels, self.kernel_width, self.seed)
        for name, values in self.params.items():
            if name != "embedding":
                setattr(model, name, values.copy())
        model.trained = self.step_count > 0
        model.status = "trained_source_restoration_not_literary_quality" if model.trained else "untrained"
        return model

    def feature_factory(self, vocabulary: Mapping[str, int], length: int, mask_start=0):
        """Snapshot fixed feature(row, column, character) for exact DP adapters.

        `row` is local to this couplet (0 or 1). Unknown characters use id 0.
        The frozen vocabulary is supplied by the caller and must match training.
        """
        self._check_parameters()
        positions = self._position_features(length, mask_start)
        vocab = dict(vocabulary)
        if any(not isinstance(token, str) or isinstance(index, bool)
               or not isinstance(index, Integral) or not 0 <= index < self.vocab_size
               for token, index in vocab.items()):
            raise ValueError("vocabulary mapping contains invalid character ids")
        embedding = self.params["embedding"].copy()

        def feature(row, column, token):
            if row not in (0, 1) or not 0 <= column < length:
                raise ValueError("feature position is outside the fixed couplet")
            return np.concatenate((embedding[vocab.get(token, 0)], positions[row, column]))
        return feature

    def save(self, path):
        """Persist weights and optimizer state without Python pickle objects."""
        self._check_parameters(optimizer=True)
        metadata = {"format": self.format_version, "vocab_size": self.vocab_size,
                    "kernel_width": self.kernel_width, "channels": self.channels,
                    "seed": self.seed, "embedding_dim": self.embedding_dim,
                    "feature_dim": self.feature_dim, "step_count": self.step_count,
                    "last_optimizer": self.last_optimizer,
                    "feature_contract": ["embedding8", "couplet_row", "normalized_column",
                                         "(length-5)/2", "lower_two_character_mask_indicator"],
                    "interpretation": "source_restoration_only_no_literary_quality_evaluation"}
        arrays = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
        for prefix, collection in (("parameter", self.params), ("adam_m", self.adam_m), ("adam_v", self.adam_v)):
            arrays.update({f"{prefix}_{name}": value for name, value in collection.items()})
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format") != cls.format_version or metadata.get("embedding_dim") != cls.embedding_dim or metadata.get("feature_dim") != cls.feature_dim:
                raise ValueError("unsupported learned model format or feature contract")
            model = cls(metadata["vocab_size"], metadata["kernel_width"], metadata["channels"], metadata["seed"])
            model.params = {name: np.array(archive[f"parameter_{name}"], dtype=np.float64, copy=True) for name in model._shapes()}
            model.adam_m = {name: np.array(archive[f"adam_m_{name}"], dtype=np.float64, copy=True) for name in model._shapes()}
            model.adam_v = {name: np.array(archive[f"adam_v_{name}"], dtype=np.float64, copy=True) for name in model._shapes()}
            model.step_count = metadata["step_count"]
            model.last_optimizer = metadata.get("last_optimizer")
            model._check_parameters(optimizer=True)
            return model
