# Standalone Cannotate — extraction design

*2026-08-16*

## Goal

Take the annotation tool out of `birdsong-preprocessing-gui` and make it a
project of its own, launchable by double-click with no terminal.

`birdsong-preprocessing-gui` had accumulated training scripts, UMAP workflows,
ViT experiments, format converters and the annotation GUI in one flat
directory. Only the last of those is wanted here.

## What had to come along

Tracing imports from `Cannotate_GUI.py`:

**Hard** — module-level, crashes without them:
`cannotate/` (6 modules), `utils.py`, `colors.py`, plus `torch` and
`bird_song_detector` imported unguarded in `cannotate_utils.py` and
`spectrogram_viewer.py`.

**Lazy** — imported only when a button is pressed:
`classification_model/segment_classifier.py` (→ `cnn1d_torch_train`,
`windowing`, `model_zoo/`) and `full_annotation_pipeline.py`.

## Decisions

### NN features stay, but become optional

Rather than strip the NN buttons (risky — it means editing a 2800-line viewer)
or keep them mandatory (drags torch onto every lab machine), the two
module-level imports are wrapped in `try`/`except ImportError` and record
`NN_AVAILABLE` / `TORCH_AVAILABLE`.

The buttons are still created when torch is absent, just `state="disabled"`,
with a red note giving the import error. A missing feature that is visible and
explained beats a window that silently lacks a third of its controls.

One subtlety: `OnnxSegmentationModel.__call__` was annotated `x: torch.Tensor`.
Annotations are evaluated at class-definition time, so with `torch = None` that
raised `AttributeError` on import. The annotations are now quoted strings.

### `bird_song_detector` is vendored, not depended on

It is not on PyPI — it is a locally-installed package — so requiring it would
have made "standalone" a fiction. Only three names are needed: `ConvMLP`,
`ClassificationCnn`, `InferenceToAnnotationData`.

Copying the package wholesale would have brought h5py, pandas, scikit-learn,
seaborn and tensorboard. `InferenceToAnnotationData` reaches into
`data_loader.py` (2848 lines) and `training_loop.py` (1094 lines) for exactly
one function each — `load_from_wav` and `normalize_per_segment`, both ~15 lines
and depending only on torch/numpy/scipy. Those two are reproduced in
`bird_song_detector/_helpers.py`; everything else is a verbatim copy.

Full detail in [`bird_song_detector/PROVENANCE.md`](../../bird_song_detector/PROVENANCE.md).

### Model weights move into the project

`NN_WEIGHTS_DIR` pointed at a sibling repo (`../bird_song_detector_tool`) and
`ONNX_DEFAULT_WEIGHTS` two levels up into
`behavioral_experiment_syntax_manipulation/`. Both break the moment the project
is copied anywhere.

Weights now live in `models/` inside the project (~5 MB, small enough to
commit), overridable with `CANNOTATE_MODELS_DIR`.
`full_annotation_pipeline.py` had its own duplicate of the constant; it now
imports the one in `cannotate_utils` so there is a single source of truth.

### `utils.py` trimmed

The original held 18 functions — MAT→JSON converters, dataset pruning, JSON
merging. The GUI imports 5 names. Only those and their private helpers were
carried over, which also drops a stray unused `from sqlalchemy import exists`
that would otherwise have made sqlalchemy a dependency of an annotation tool.

### The launcher

`launch.py` holds all the logic and uses only the standard library, because it
must run under whatever Python is around — including one missing every package
the GUI needs. Its job is to find a better interpreter than the one running it.

Search order: `launcher.cfg` → `CANNOTATE_PYTHON` → current interpreter →
conda envs under the usual roots → `python3`/`python` on PATH. Each candidate
is probed in a subprocess for the hard dependencies; the first that passes is
cached back to `launcher.cfg`, so only the first launch pays for the search.

`torch` is deliberately *not* a probe requirement — the tool works without it,
so requiring it would turn an optional feature into a hard blocker.
`colors.py` accepts either OpenCV or scikit-image, so the probe treats that
pair as an alternatives group.

Four wrappers, all thin: `Cannotate.command` / `Cannotate.app` on macOS,
`Cannotate.bat` / `Cannotate.vbs` on Windows. The `.app` and `.vbs` variants
show no console window. The `.app` resolves its own location at runtime rather
than baking in an absolute path, so moving the folder does not break it.

## Verification

`smoke_test.py` covers the properties that make this a separate project rather
than a folder of copies:

1. `Cannotate_GUI` imports
2. no first-party module resolves outside the project directory — catches an
   incomplete extraction that only *looks* standalone on this machine
3. the vendored subset has not regrown its training dependencies
4. the default model files are present
5. the NN availability flags exist
6. the viewer builds its widget tree, and the NN buttons match torch
   availability

All six pass under `moove_env`, and — more tellingly — under
`birdsong_preprocessing`, which does **not** have `bird_song_detector`
installed. That is the proof the vendoring works.

The button-state check was also run with torch blocked by an import hook,
confirming the disabled path builds correctly.

## Known limitations

- `Cannotate.app` is unsigned, so macOS Gatekeeper requires right-click → Open
  on first launch.
- The Windows wrappers are written but untested — there is no Windows machine
  in reach.
- `classification_model/cnn1d_torch_train.py` imports pandas and scikit-learn
  at module level even though only inference is used here. It stays lazy, so it
  costs nothing unless the Classify button is pressed; splitting inference out
  of the training module was left alone as out of scope.
