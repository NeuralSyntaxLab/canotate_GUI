# Vendored `bird_song_detector`

This directory is a **partial copy** of the `bird_song_detector` package. It is
checked in so the annotation tool is self-contained — no separate install, no
`pip install -e`, no `PYTHONPATH` juggling on lab machines.

## Upstream

| | |
|---|---|
| Repo | `bird_song_detector_tool` |
| Commit | `f172284dd7795ea34b707490615ffe764b2dcfde` (2026-07-29) |
| Copied from | the installed package in the `moove_env` conda environment |
| Copied on | 2026-08-16 |

## What is here, and why

The tool imports exactly three names from this package:

- `bird_song_detector.model_zoo.ConvMLP` and `.ClassificationCnn` — used by
  [`cannotate/cannotate_utils.py`](../cannotate/cannotate_utils.py)
- `bird_song_detector.nn_segmentation.InferenceToAnnotationData` — used by
  [`cannotate/spectrogram_viewer.py`](../cannotate/spectrogram_viewer.py) and
  [`full_annotation_pipeline.py`](../full_annotation_pipeline.py)

| File | Status |
|---|---|
| `model_zoo.py` | verbatim copy |
| `nn_segmentation/__init__.py` | verbatim copy |
| `nn_segmentation/segmentation_utils.py` | verbatim copy |
| `nn_segmentation/inference_to_annotation_data.py` | verbatim **except** two import lines (see below) |
| `_helpers.py` | **new** — holds two functions lifted verbatim from upstream |
| `__init__.py` | **new** — minimal; upstream's version imported far more |

## The two deviations

**1. `__init__.py` was rewritten.** Upstream re-exports `load_from_wav`,
`load_data`, `AudioAugmentation` and `compute_spectrogram`, which import
`data_loader.py` and `data_visualizer.py` at package-import time. Nothing in
this project uses them, and they pull in h5py and matplotlib. Dropped.

**2. `inference_to_annotation_data.py` has two imports redirected.** It
originally reached into two large training-oriented modules for one function
each:

```python
from bird_song_detector.data_loader import load_from_wav        # 2848-line file
from bird_song_detector.training_loop import normalize_per_segment  # 1094-line file
```

Both functions are ~15 lines and depend only on torch/numpy/scipy. They are
reproduced verbatim in `_helpers.py`, and the import now reads:

```python
from bird_song_detector._helpers import load_from_wav, normalize_per_segment
```

This keeps h5py, pandas, scikit-learn, seaborn and tensorboard out of the
dependency tree entirely.

## Re-syncing

If upstream changes, re-copy the four verbatim files, re-apply the two-line
import edit to `inference_to_annotation_data.py`, and diff `_helpers.py`
against upstream's `load_from_wav` / `normalize_per_segment` to confirm they
still match. Then run `python smoke_test.py`.
