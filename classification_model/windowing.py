"""Lightweight waveform windowing for inference (numpy-only).

Extracted from cnn1d_runtime_plot.py so the real-time inference path
(syllable_classifier.py) does not import matplotlib. Behavior is unchanged.

Note: normalization here is legacy/optional. The trained-model pipeline now
normalizes inside Conv1DClassifier using the persisted normalization_config,
so callers on the live path pass normalize=False and let the classifier
normalize once (see model_logics/CNN1D_INTEGRATION_NOTES.md).
"""

from __future__ import annotations

import numpy as np


def normalize_segment(seg: np.ndarray) -> np.ndarray:
    """Per-segment peak normalization (subtract mean, divide by peak abs)."""
    seg = seg.astype(np.float32, copy=False)
    seg = seg - np.mean(seg)
    peak = np.max(np.abs(seg))
    if peak > 0:
        seg = seg / peak
    return seg


def extract_window(wav: np.ndarray, center_idx: int, num_samples: int, normalize: bool = True) -> np.ndarray:
    """Extract a fixed-length window centred on ``center_idx`` (in samples).

    Pads with zeros past either edge so the result is always ``num_samples`` long.
    """
    start_idx = center_idx - num_samples // 2
    end_idx = start_idx + num_samples
    left_pad = max(0, -start_idx)
    right_pad = max(0, end_idx - len(wav))
    clipped_start = max(0, start_idx)
    clipped_end = min(len(wav), end_idx)
    seg = wav[clipped_start:clipped_end]
    if left_pad or right_pad:
        seg = np.pad(seg, (left_pad, right_pad), mode="constant")
    if len(seg) != num_samples:
        seg = np.resize(seg, num_samples)
    if normalize:
        seg = normalize_segment(seg)
    return seg.astype(np.float32, copy=False)
