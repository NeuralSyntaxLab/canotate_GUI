"""Apply a trained 1D-CNN to classify segmented windows of a wav file.

This is the inference counterpart to ``cnn1d_torch_train.py``. Where training
samples fixed-length windows from inside each labelled phrase and lets
``Conv1DClassifier`` normalize them once (via the persisted
``normalization_config``), this module does the same thing at inference time
for *unlabelled* segments.

Intended pipeline
-----------------
A (future or current) segmentation model produces a list of segments
``(onset_s, offset_s, label)``. Segments it cannot label are marked with the
``999`` placeholder used throughout this project. ``SegmentClassifier`` walks
those ``999`` segments, runs the trained classifier over each one, and writes
the predicted label back, yielding a fully annotated wav:

    segmenter -> [(t0, t1, 999), ...] -> SegmentClassifier -> [(t0, t1, <label>), ...]

Normalization note (matches the training/runtime contract)
----------------------------------------------------------
Windows are extracted RAW here (``normalize=False``). ``Conv1DClassifier``
applies the exact normalization the model was trained with (``segment_peak``,
``train_zscore``, or ``none``) using the config persisted in ``metadata.json``.
Normalizing here too would double-normalize and break ``train_zscore`` models.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf

from classification_model.cnn1d_torch_train import Conv1DClassifier
from classification_model.windowing import extract_window


# Project-wide placeholder for an unlabelled / unknown segment.
UNLABELED = 999

# A segment is (onset_s, offset_s, label). Label may be int (e.g. 999) or str.
SegmentTriplet = Tuple[float, float, Union[int, str]]


@dataclass
class SegmentPrediction:
    """Result of classifying one segment."""

    onset_s: float
    offset_s: float
    label: Union[int, str]            # predicted label (or original if not classified)
    confidence: float                 # mean-probability of the winning class (NaN if untouched)
    original_label: Union[int, str]   # label before classification
    classified: bool                  # True if this module assigned the label
    n_windows: int                    # number of windows aggregated for the prediction

    def as_triplet(self) -> SegmentTriplet:
        return (float(self.onset_s), float(self.offset_s), self.label)


def _load_mono_wav(wav_path: Path) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(wav_path), always_2d=False)
    if wav.ndim == 2:
        wav = wav[:, np.argmax(np.std(wav, axis=0))]
    return wav.astype(np.float32), int(sr)


def _coerce_label(value) -> Union[int, str]:
    """Render a model class string back to int when it is integer-like.

    Training stores labels as strings ("1".."29"); the segment pipeline uses
    ints (with 999 for unlabelled). Keep ints as ints so round-tripping the
    triplet format is lossless, but fall back to the raw string otherwise.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


class SegmentClassifier:
    """Classify segmented windows of a wav using a trained ``Conv1DClassifier``."""

    def __init__(
        self,
        classifier: Conv1DClassifier,
        *,
        num_samples: int,
        train_sample_rate: Optional[int] = None,
        sample_secs: Optional[float] = None,
        verbose: bool = True,
    ):
        self.classifier = classifier
        self.num_samples = int(num_samples)
        self.train_sample_rate = None if train_sample_rate is None else int(train_sample_rate)
        self.sample_secs = None if sample_secs is None else float(sample_secs)
        self.verbose = verbose

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls,
        model_dir: Union[str, Path],
        *,
        num_samples: Optional[int] = None,
        device=None,
        verbose: bool = True,
    ) -> "SegmentClassifier":
        """Load a trained model directory (the output of ``fit_from_dataset_pickle``).

        ``num_samples`` (the model's input width) and the training sample rate
        are recovered from ``metadata.json`` when not provided explicitly.
        """
        model_dir = Path(model_dir)
        meta_path = model_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        inner = meta.get("metadata", {}) if isinstance(meta, dict) else {}

        if num_samples is None:
            num_samples = inner.get("num_samples") or inner.get("input_num_samples")
        if num_samples is None:
            raise ValueError(
                "num_samples could not be recovered from metadata.json; pass it explicitly "
                "(the input window width in samples used during training)."
            )

        classifier = Conv1DClassifier.load(
            model_dir=model_dir, input_num_samples=int(num_samples), device=device
        )
        return cls(
            classifier=classifier,
            num_samples=int(num_samples),
            train_sample_rate=inner.get("sample_rate"),
            sample_secs=inner.get("sample_secs"),
            verbose=verbose,
        )

    # -------------------------------------------------------------- extraction
    def _segment_centers(self, onset_s: float, offset_s: float, sr: int, n_windows: int) -> np.ndarray:
        """Window-center sample indices spanning a segment.

        For ``n_windows == 1`` this is the segment midpoint (mirroring how the
        builder centers a window). For more, centers are spread evenly across
        the interior of the segment so a long segment is covered, not just its
        middle, then averaged for a robust per-segment label.
        """
        if n_windows <= 1:
            centers_s = np.asarray([0.5 * (onset_s + offset_s)], dtype=np.float64)
        else:
            # Evenly spaced, inset from the edges by half a step to avoid
            # windows that fall mostly outside the segment.
            edges = np.linspace(onset_s, offset_s, n_windows + 1)
            centers_s = 0.5 * (edges[:-1] + edges[1:])
        return np.clip(np.round(centers_s * sr).astype(int), 0, None)

    def _windows_for_segment(
        self, wav: np.ndarray, onset_s: float, offset_s: float, sr: int, n_windows: int
    ) -> np.ndarray:
        centers = self._segment_centers(onset_s, offset_s, sr, n_windows)
        return np.stack(
            [
                extract_window(wav, int(c), num_samples=self.num_samples, normalize=False)
                for c in centers
            ]
        )

    # --------------------------------------------------------------- inference
    def classify_segments(
        self,
        wav: np.ndarray,
        sr: int,
        segments: Sequence[SegmentTriplet],
        *,
        windows_per_segment: int = 3,
        only_unlabeled: bool = True,
        unlabeled_value: Union[int, str] = UNLABELED,
        min_confidence: float = 0.0,
    ) -> List[SegmentPrediction]:
        """Classify a list of ``(onset_s, offset_s, label)`` segments.

        Parameters
        ----------
        wav, sr
            Mono waveform and its sample rate.
        segments
            Segment triplets. Labels equal to ``unlabeled_value`` (default 999)
            are the ones classified when ``only_unlabeled`` is True.
        windows_per_segment
            Number of fixed-length windows sampled across each segment and
            averaged (mean class probability) for a single robust label.
        only_unlabeled
            If True, only segments whose label == ``unlabeled_value`` are
            classified; others are passed through unchanged. If False, every
            segment is (re)classified.
        min_confidence
            If the winning mean-probability is below this threshold, the
            segment keeps ``unlabeled_value`` instead of the prediction.

        Returns
        -------
        List[SegmentPrediction] in the same order as ``segments``.
        """
        if sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}")
        if self.train_sample_rate is not None and int(sr) != self.train_sample_rate:
            warnings.warn(
                f"wav sample rate ({sr} Hz) differs from the training sample rate "
                f"({self.train_sample_rate} Hz). The model input width is fixed at "
                f"{self.num_samples} samples, so each window now spans "
                f"{self.num_samples / sr:.4f}s instead of the trained "
                f"{self.num_samples / self.train_sample_rate:.4f}s. Resample the wav to "
                f"{self.train_sample_rate} Hz for faithful predictions."
            )

        # Collect the segments that actually need a forward pass, batch them
        # through the model once, then scatter the results back.
        to_classify: List[int] = []
        per_seg_windows: List[np.ndarray] = []
        for i, seg in enumerate(segments):
            onset_s, offset_s, label = float(seg[0]), float(seg[1]), seg[2]
            needs = (not only_unlabeled) or _labels_equal(label, unlabeled_value)
            if needs and offset_s > onset_s:
                windows = self._windows_for_segment(wav, onset_s, offset_s, sr, windows_per_segment)
                per_seg_windows.append(windows)
                to_classify.append(i)

        probs_by_seg: Dict[int, np.ndarray] = {}
        if per_seg_windows:
            counts = [w.shape[0] for w in per_seg_windows]
            stacked = np.concatenate(per_seg_windows, axis=0)
            all_probs = self.classifier.predict_proba(stacked)  # classifier normalizes internally
            offset = 0
            for seg_i, n in zip(to_classify, counts):
                probs_by_seg[seg_i] = all_probs[offset:offset + n].mean(axis=0)
                offset += n

        classes = self.classifier.label_encoder.classes_
        results: List[SegmentPrediction] = []
        for i, seg in enumerate(segments):
            onset_s, offset_s, label = float(seg[0]), float(seg[1]), seg[2]
            if i in probs_by_seg:
                mean_probs = probs_by_seg[i]
                win_idx = int(np.argmax(mean_probs))
                conf = float(mean_probs[win_idx])
                if conf >= min_confidence:
                    pred_label = _coerce_label(classes[win_idx])
                else:
                    pred_label = unlabeled_value
                results.append(
                    SegmentPrediction(
                        onset_s=onset_s,
                        offset_s=offset_s,
                        label=pred_label,
                        confidence=conf,
                        original_label=label,
                        classified=conf >= min_confidence,
                        n_windows=int(windows_per_segment),
                    )
                )
            else:
                results.append(
                    SegmentPrediction(
                        onset_s=onset_s,
                        offset_s=offset_s,
                        label=label,
                        confidence=float("nan"),
                        original_label=label,
                        classified=False,
                        n_windows=0,
                    )
                )

        if self.verbose:
            n_done = sum(1 for r in results if r.classified)
            print(f"[segment-classifier] classified {n_done}/{len(results)} segment(s)")
        return results

    def classify_segments_dense(
        self,
        wav: np.ndarray,
        sr: int,
        segments: Sequence[SegmentTriplet],
        *,
        only_unlabeled: bool = True,
        unlabeled_value: Union[int, str] = UNLABELED,
        min_confidence: float = 0.0,
    ) -> List[SegmentPrediction]:
        """Dense per-window classification with automatic sub-segment splitting.

        Slides the model's native window (num_samples wide, stride = one window
        duration) from onset to offset for each segment to classify.  Each
        window receives its own argmax label.  Consecutive windows with the
        same label are merged into a single SegmentPrediction, so one input
        segment can yield several output predictions with different labels.

        Non-classified segments (wrong label or zero duration) pass through
        unchanged as a single SegmentPrediction.
        """
        step_secs = (
            float(self.sample_secs)
            if self.sample_secs is not None
            else self.num_samples / float(self.train_sample_rate or sr)
        )

        # Build a flat list of windows for one batched forward pass.
        # Parallel list: (seg_idx, w_onset, w_offset) for each window.
        all_windows: List[np.ndarray] = []
        win_meta: List[Tuple[int, float, float]] = []  # (seg_idx, w_onset, w_offset)
        seg_window_counts: List[int] = []

        for seg_idx, seg in enumerate(segments):
            onset_s  = float(seg[0])
            offset_s = float(seg[1])
            label    = seg[2]
            needs    = (not only_unlabeled) or _labels_equal(label, unlabeled_value)

            if not needs or offset_s <= onset_s:
                seg_window_counts.append(0)
                continue

            n_steps = max(1, int(np.ceil((offset_s - onset_s) / step_secs)))
            count = 0
            for i in range(n_steps):
                w_onset  = onset_s + i * step_secs
                if w_onset >= offset_s:
                    break
                w_offset = min(onset_s + (i + 1) * step_secs, offset_s)
                center_idx = int(round(0.5 * (w_onset + w_offset) * sr))
                win = extract_window(wav, center_idx, num_samples=self.num_samples, normalize=False)
                all_windows.append(win)
                win_meta.append((seg_idx, w_onset, w_offset))
                count += 1
            seg_window_counts.append(count)

        # Single batched forward pass
        classes = self.classifier.label_encoder.classes_
        per_window_label: List[Union[int, str]] = []
        per_window_conf: List[float] = []
        if all_windows:
            all_probs = self.classifier.predict_proba(np.stack(all_windows))
            for probs in all_probs:
                win_idx = int(np.argmax(probs))
                conf    = float(probs[win_idx])
                if conf >= min_confidence:
                    per_window_label.append(_coerce_label(classes[win_idx]))
                else:
                    per_window_label.append(unlabeled_value)
                per_window_conf.append(conf)

        # Scatter back, run-length encode per segment
        win_cursor = 0
        results: List[SegmentPrediction] = []

        for seg_idx, seg in enumerate(segments):
            onset_s  = float(seg[0])
            offset_s = float(seg[1])
            label    = seg[2]
            n_wins   = seg_window_counts[seg_idx]

            if n_wins == 0:
                results.append(SegmentPrediction(
                    onset_s=onset_s, offset_s=offset_s, label=label,
                    confidence=float("nan"), original_label=label,
                    classified=False, n_windows=0,
                ))
                continue

            w_labels = per_window_label[win_cursor: win_cursor + n_wins]
            w_confs  = per_window_conf[win_cursor: win_cursor + n_wins]
            w_meta   = win_meta[win_cursor: win_cursor + n_wins]
            win_cursor += n_wins

            # Run-length encode label sequence → sub-segments
            groups: List[Tuple[float, float, Union[int, str], List[float]]] = []
            g_on, g_off, g_lbl, g_confs = w_meta[0][1], w_meta[0][2], w_labels[0], [w_confs[0]]
            for k in range(1, len(w_labels)):
                _, wk_on, wk_off = w_meta[k]
                if _labels_equal(w_labels[k], g_lbl):
                    g_off = wk_off
                    g_confs.append(w_confs[k])
                else:
                    groups.append((g_on, g_off, g_lbl, g_confs))
                    g_on, g_off, g_lbl, g_confs = wk_on, wk_off, w_labels[k], [w_confs[k]]
            groups.append((g_on, g_off, g_lbl, g_confs))

            for g_on, g_off, g_lbl, g_confs in groups:
                results.append(SegmentPrediction(
                    onset_s=g_on,
                    offset_s=g_off,
                    label=g_lbl,
                    confidence=float(np.mean(g_confs)),
                    original_label=label,
                    classified=not _labels_equal(g_lbl, unlabeled_value),
                    n_windows=len(g_confs),
                ))

        if self.verbose:
            n_done   = sum(1 for r in results if r.classified)
            n_split  = len(results) - len(segments)
            print(f"[segment-classifier] dense: {n_done}/{len(results)} classified"
                  f", {n_split:+d} sub-segments from splitting")
        return results

    def annotate_wav(
        self,
        wav_path: Union[str, Path],
        segments: Sequence[SegmentTriplet],
        *,
        windows_per_segment: int = 3,
        only_unlabeled: bool = True,
        unlabeled_value: Union[int, str] = UNLABELED,
        min_confidence: float = 0.0,
        dense: bool = False,
        fill_gaps: bool = False,
    ) -> List[SegmentPrediction]:
        """Load a wav file and classify its segments (convenience wrapper).

        Parameters
        ----------
        dense
            When True, uses dense per-window classification with automatic
            sub-segment splitting (classify_segments_dense) instead of the
            default averaged-window approach.  Recommended for the full
            annotation pipeline.
        fill_gaps
            When True, applies gap-filling after classification: any 999
            segment whose nearest labeled neighbors on both sides share the
            same label is relabeled to that label.
        """
        wav, sr = _load_mono_wav(Path(wav_path))
        if self.verbose:
            print(f"[segment-classifier] {Path(wav_path).name} | sr={sr} | n={len(wav)} | segments={len(segments)}")
        if dense:
            results = self.classify_segments_dense(
                wav, sr, segments,
                only_unlabeled=only_unlabeled,
                unlabeled_value=unlabeled_value,
                min_confidence=min_confidence,
            )
        else:
            results = self.classify_segments(
                wav, sr, segments,
                windows_per_segment=windows_per_segment,
                only_unlabeled=only_unlabeled,
                unlabeled_value=unlabeled_value,
                min_confidence=min_confidence,
            )
        if fill_gaps:
            results = fill_label_gaps(results, unlabeled_value)
        return results


def _labels_equal(a, b) -> bool:
    """Compare labels tolerant of int/str representation (999 vs '999')."""
    if a == b:
        return True
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def predictions_to_triplets(predictions: Sequence[SegmentPrediction]) -> List[SegmentTriplet]:
    """Flatten predictions back to the ``(onset_s, offset_s, label)`` triplet format."""
    return [p.as_triplet() for p in predictions]


def fill_label_gaps(
    predictions: List[SegmentPrediction],
    unlabeled_value: Union[int, str] = UNLABELED,
) -> List[SegmentPrediction]:
    """Relabel a 999 segment to label X when X flanks it on both sides.

    Scans the prediction list in order.  For each unlabeled (999) segment,
    finds the nearest non-999 neighbor to the left and to the right.  If both
    share the same label, the gap segment is relabeled to that label.

    A run of consecutive 999 segments bounded on both ends by the same label is
    filled entirely (e.g. ``x 999 999 x`` -> ``x x x x``), because each 999's
    nearest non-999 neighbours are still that same label.  Gaps whose left and
    right labels differ (e.g. ``x 999 y``) and 999 segments at the start/end of
    the list with no labelled neighbour on one side are left untouched.

    The prediction list must be in temporal (onset) order for the neighbour
    lookup to be meaningful.
    """
    result = list(predictions)
    for i, pred in enumerate(result):
        if not _labels_equal(pred.label, unlabeled_value):
            continue
        left_label = None
        for j in range(i - 1, -1, -1):
            if not _labels_equal(result[j].label, unlabeled_value):
                left_label = result[j].label
                break
        right_label = None
        for j in range(i + 1, len(result)):
            if not _labels_equal(result[j].label, unlabeled_value):
                right_label = result[j].label
                break
        if (left_label is not None
                and right_label is not None
                and _labels_equal(left_label, right_label)):
            result[i] = SegmentPrediction(
                onset_s=pred.onset_s,
                offset_s=pred.offset_s,
                label=left_label,
                confidence=pred.confidence,
                original_label=pred.original_label,
                classified=True,
                n_windows=pred.n_windows,
            )
    return result


if __name__ == "__main__":
    # Minimal usage example; edit paths as needed.
    script_dir = Path(__file__).parent.resolve()
    model_dir = script_dir / "conv1d_phrase_model"

    # Example: a segmenter handed us these spans, all unlabelled (999).
    wav_path = script_dir / "lb4444" / "example.wav"
    segments: List[SegmentTriplet] = [
        (0.50, 0.62, UNLABELED),
        (0.81, 0.95, UNLABELED),
        (1.20, 1.47, UNLABELED),
    ]

    clf = SegmentClassifier.load(model_dir, verbose=True)
    preds = clf.annotate_wav(wav_path, segments, windows_per_segment=3)
    for p in preds:
        print(f"  [{p.onset_s:.3f}, {p.offset_s:.3f}] -> {p.label} (conf={p.confidence:.3f})")
    print(predictions_to_triplets(preds))
