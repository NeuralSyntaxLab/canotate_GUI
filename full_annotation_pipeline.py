"""End-to-end annotation pipeline: segmentation model → classification model.

Usage (single file):
    pipeline = FullAnnotationPipeline.load(
        ConvMLP, {"input_size": 256}, "weights.pth", "clf_model_dir/"
    )
    segments = pipeline.annotate_wav("song.wav")

Usage (batch — merges into existing JSON):
    pipeline.annotate_directory(
        wav_dir="2024-06-15/chop_data/wav/",
        annotation_json="bird_annotation.json",
    )

Output segments are 4-tuples: (onset_s, offset_s, label, True).
The 4th element marks every segment as AI-generated so the GUI can
display and toggle them independently of manual annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from bird_song_detector.nn_segmentation import InferenceToAnnotationData
from bird_song_detector.nn_segmentation.segmentation_utils import _merge_close_segments
from classification_model.segment_classifier import (
    SegmentClassifier,
    UNLABELED,
)


# Single source of truth for where weights live — see cannotate_utils, which
# honours the CANNOTATE_MODELS_DIR override.
from cannotate.cannotate_utils import ONNX_DEFAULT_WEIGHTS  # noqa: E402,F401
from cannotate.onnx_segmentation import OnnxSegmentationModel  # noqa: E402,F401
from utils import normalize_annotation_dates, song_wavs_under


def _best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_ai_segment(seg) -> bool:
    """Return True when the segment carries the AI-generated flag (4th element)."""
    return len(seg) > 3 and bool(seg[3])


class FullAnnotationPipeline:
    """Chain segmentation → classification into a single annotate_wav() call.

    Models are loaded once and reused across files — no GPU re-init per wav.

    Output segments carry a 4th boolean element (True) so the GUI can show,
    hide, or override AI annotations independently of manual ones.
    """

    def __init__(
        self,
        segmenter: InferenceToAnnotationData,
        classifier: SegmentClassifier,
        *,
        seg_confidence: float = 0.5,
        clf_confidence: float = 0.0,
        clf_windows_per_segment: int = 3,
        dense: bool = True,
        fill_gaps: bool = True,
        merge_gap_threshold: float = 2 / 100,
    ):
        self._segmenter = segmenter
        self._classifier = classifier
        self._seg_confidence = float(seg_confidence)
        self._clf_confidence = float(clf_confidence)
        self._clf_windows = int(clf_windows_per_segment)
        self._dense = bool(dense)
        self._fill_gaps = bool(fill_gaps)
        self._merge_gap = float(merge_gap_threshold)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(
        cls,
        seg_model_cls,
        seg_model_kwargs: dict,
        seg_weights_path: Union[str, Path],
        clf_model_dir: Union[str, Path],
        *,
        seg_confidence: float = 0.5,
        clf_confidence: float = 0.0,
        clf_windows_per_segment: int = 3,
        dense: bool = True,
        fill_gaps: bool = True,
        device=None,
        verbose: bool = True,
    ) -> "FullAnnotationPipeline":
        """Load both models and return a ready pipeline.

        Parameters
        ----------
        seg_model_cls     : segmentation model class (e.g. ConvMLP)
        seg_model_kwargs  : constructor kwargs (e.g. {"input_size": 256})
        seg_weights_path  : .pth weights file for the segmentation model
        clf_model_dir     : directory containing metadata.json for SegmentClassifier
        seg_confidence    : binary classification threshold for the segmenter (0–1)
        clf_confidence    : min classifier confidence; segments below keep label 999
        clf_windows_per_segment : windows averaged per segment in the classifier (non-dense mode)
        dense             : use per-window dense classification with sub-segment splitting
        fill_gaps         : fill 999 gaps flanked on both sides by the same label
        device            : torch device; auto-selects cuda/mps/cpu when None
        verbose           : print progress info
        """
        if device is None:
            device = _best_device()
        if verbose:
            print(f"[pipeline] device: {device}")

        seg_model = seg_model_cls(**seg_model_kwargs)
        state = torch.load(str(seg_weights_path), map_location=device)
        seg_model.load_state_dict(state)
        seg_model.to(device)
        seg_model.eval()
        if verbose:
            print(f"[pipeline] segmentation model loaded from {seg_weights_path}")

        segmenter = InferenceToAnnotationData(seg_model, device)
        classifier = SegmentClassifier.load(clf_model_dir, device=device, verbose=verbose)

        return cls(
            segmenter,
            classifier,
            seg_confidence=seg_confidence,
            clf_confidence=clf_confidence,
            clf_windows_per_segment=clf_windows_per_segment,
            dense=dense,
            fill_gaps=fill_gaps,
        )

    @classmethod
    def load_onnx(
        cls,
        clf_model_dir: Union[str, Path],
        *,
        onnx_path: Union[str, Path] = ONNX_DEFAULT_WEIGHTS,
        seg_confidence: float = 0.5,
        clf_confidence: float = 0.0,
        clf_windows_per_segment: int = 3,
        dense: bool = True,
        fill_gaps: bool = True,
        device=None,
        verbose: bool = True,
    ) -> "FullAnnotationPipeline":
        """Load the pipeline using an ONNX segmentation model.

        The ONNX graph encodes the ConvMLP architecture from bird_song_detector.
        onnxruntime is used for inference; no PyTorch model constructor needed.

        Parameters
        ----------
        clf_model_dir : directory containing metadata.json for SegmentClassifier
        onnx_path     : path to the .onnx file (defaults to ONNX_DEFAULT_WEIGHTS)
        dense         : use per-window dense classification with sub-segment splitting
        fill_gaps     : fill 999 gaps flanked on both sides by the same label
        """
        if device is None:
            device = _best_device()
        if verbose:
            print(f"[pipeline] device: {device}")
            print(f"[pipeline] ONNX segmentation model: {onnx_path}")

        seg_model = OnnxSegmentationModel(onnx_path)
        segmenter = InferenceToAnnotationData(seg_model, device)
        classifier = SegmentClassifier.load(clf_model_dir, device=device, verbose=verbose)

        return cls(
            segmenter,
            classifier,
            seg_confidence=seg_confidence,
            clf_confidence=clf_confidence,
            clf_windows_per_segment=clf_windows_per_segment,
            dense=dense,
            fill_gaps=fill_gaps,
        )

    # ---------------------------------------------------------------- single file
    def annotate_wav(self, wav_path: Union[str, Path]) -> List[Tuple]:
        """Fully annotate one WAV file.

        Returns a list of ``(onset_s, offset_s, label, True)`` 4-tuples.
        The 4th element (True) marks every segment as AI-generated.
        Segments whose classifier confidence is below ``clf_confidence`` keep
        label 999 (unlabeled placeholder).
        """
        wav_path = str(wav_path)

        # Step 1 — segmentation (normalize_per_segment applied inside run_inference)
        raw_preds = self._segmenter.run_inference(
            wav_path, confidence_threshold=self._seg_confidence
        )
        fs = self._segmenter.recording_fs
        segments = self._segmenter.translate_inference_to_annotation_format(raw_preds, fs)

        if not segments:
            return []

        # Step 2 — classify every 999-labeled segment
        predictions = self._classifier.annotate_wav(
            wav_path,
            segments,
            windows_per_segment=self._clf_windows,
            min_confidence=self._clf_confidence,
            dense=self._dense,
            fill_gaps=self._fill_gaps,
        )

        # Step 3 — merge segments that are now known to share a real label and
        # sit close together (e.g. a one-frame detector dropout inside a
        # single syllable). Done here, after classification, because only now
        # do segments carry their real label — merging on time alone at the
        # segmentation stage can't tell that dropout apart from a short but
        # genuine pause between two different syllables.
        merged = _merge_close_segments(
            [[p.onset_s, p.offset_s, p.label] for p in predictions],
            gap_threshold=self._merge_gap,
        )

        # Step 4 — tag every output as AI-generated
        return [(float(onset), float(offset), label, True)
                for onset, offset, label in merged]

    # ---------------------------------------------------------------- batch
    def annotate_directory(
        self,
        wav_dir: Union[str, Path],
        annotation_json: Union[str, Path],
        *,
        skip_fully_annotated: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict:
        """Annotate all WAVs under wav_dir and merge results into annotation_json.

        Manual annotations (segments without the AI flag) are always preserved.
        AI segments from a previous run are replaced on re-runs.

        Parameters
        ----------
        wav_dir             : directory searched recursively for song wavs --
                              only files inside a folder named ``wav`` count,
                              so syllable clips and exports are left alone
        annotation_json     : path to the project annotation JSON (created if absent)
        skip_fully_annotated: when True, skip files that already have at least one
                              manually labeled (non-999, non-AI) segment
        progress_callback   : called as fn(done, total, wav_path_str) after each file
        """
        wav_dir = Path(wav_dir)
        annotation_json = Path(annotation_json)

        data: Dict = (
            json.loads(annotation_json.read_text(encoding="utf-8"))
            if annotation_json.exists()
            else {}
        )
        # Days written under a folder name ("YL10RB_2026-10-01") join the
        # plain date this run files them under, instead of being duplicated.
        normalize_annotation_dates(data)

        wav_files = [Path(p) for p in song_wavs_under(str(wav_dir))]
        total = len(wav_files)
        if total == 0:
            print(f"[pipeline] no WAV files found under {wav_dir}")
            return data

        print(f"[pipeline] processing {total} WAV file(s)…")

        for done, wav_path in enumerate(wav_files, 1):
            date_key = self._segmenter._extract_date_from_filename(wav_path.name)
            file_name = wav_path.name

            day_entries: List[Dict] = data.setdefault(date_key, [])
            existing = next(
                (e for e in day_entries if e["file_name"] == file_name), None
            )

            if skip_fully_annotated and existing is not None:
                segs = existing.get("segments", [])
                has_manual = any(
                    len(s) >= 3
                    and not is_ai_segment(s)
                    and s[2] not in (999, "999")
                    for s in segs
                )
                if has_manual:
                    if progress_callback:
                        progress_callback(done, total, str(wav_path))
                    continue

            try:
                new_segs = self.annotate_wav(wav_path)
            except Exception as exc:
                print(f"[pipeline] skipping {wav_path.name}: {exc}")
                if progress_callback:
                    progress_callback(done, total, str(wav_path))
                continue

            if existing is not None:
                # keep manual segments, replace AI segments
                manual = [s for s in existing.get("segments", []) if not is_ai_segment(s)]
                merged_segs = manual + [list(s) for s in new_segs]
                merged_segs.sort(key=lambda s: s[0])
                existing["segments"] = merged_segs
            else:
                day_entries.append(
                    {
                        "file_name": file_name,
                        # Machine-generated: only a human can flip this.
                        "human_verified": False,
                        "segments": sorted(
                            (list(s) for s in new_segs), key=lambda s: s[0]),
                    }
                )

            if progress_callback:
                progress_callback(done, total, str(wav_path))

        annotation_json.parent.mkdir(parents=True, exist_ok=True)
        annotation_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[pipeline] saved → {annotation_json}")
        return data
