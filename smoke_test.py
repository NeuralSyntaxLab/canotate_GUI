#!/usr/bin/env python3
"""Checks that this project really is standalone.

Run it after moving the project, changing the vendored code, or upgrading an
environment:

    python smoke_test.py

It does not test annotation behaviour — it tests the properties that make this
a separate project rather than a folder of copied files.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Modules that legitimately live outside the project: the standard library and
# third-party packages installed into the environment.
STDLIB_PREFIXES = {Path(p).resolve() for p in sys.path if "site-packages" in p}


def _fail(msg):
    print(f"FAIL: {msg}")
    return False


def test_gui_imports_headlessly():
    """The entry point and every module it needs import cleanly."""
    # Importing Cannotate_GUI only defines Cannotate(); it does not open a
    # window until called, so this is safe without a display.
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import Cannotate_GUI  # noqa: F401
    except Exception as e:
        return _fail(f"could not import Cannotate_GUI: {e!r}")
    print("ok: Cannotate_GUI imports")
    return True


def test_no_imports_escape_the_project():
    """No first-party module resolves to a path outside this project.

    This is the check that catches an incomplete extraction: if some module is
    still being picked up from the old birdsong-preprocessing-gui folder, the
    project only *looks* standalone on this machine.
    """
    first_party = ("cannotate", "bird_song_detector", "classification_model",
                   "utils", "colors", "full_annotation_pipeline")
    leaked = []
    for name, module in sorted(sys.modules.items()):
        if not name.split(".")[0] in first_party:
            continue
        path = getattr(module, "__file__", None)
        if not path:
            continue
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            leaked.append(f"{name} -> {resolved}")
    if leaked:
        return _fail("these modules load from outside the project:\n    "
                     + "\n    ".join(leaked))
    print(f"ok: all first-party modules resolve inside {PROJECT_ROOT}")
    return True


def test_vendored_subset_is_lean():
    """The vendored bird_song_detector does not drag in the training stack.

    The whole point of trimming it was to keep h5py, pandas, scikit-learn,
    seaborn and tensorboard off the import path. If someone re-copies the
    upstream files wholesale, this catches it.
    """
    banned = {"h5py", "seaborn", "torch.utils.tensorboard"}
    before = set(sys.modules)
    import bird_song_detector  # noqa: F401
    import bird_song_detector.nn_segmentation  # noqa: F401
    pulled_in = banned & (set(sys.modules) - before)
    if pulled_in:
        return _fail(f"vendored bird_song_detector pulled in {sorted(pulled_in)}")
    print("ok: vendored bird_song_detector stays lean")
    return True


def test_model_files_are_present():
    """The weights the GUI defaults to actually ship with the project."""
    from cannotate.cannotate_utils import NN_WEIGHTS_DIR, ONNX_DEFAULT_WEIGHTS

    problems = []
    if not os.path.isdir(NN_WEIGHTS_DIR):
        problems.append(f"weights dir missing: {NN_WEIGHTS_DIR}")
    if not os.path.isfile(ONNX_DEFAULT_WEIGHTS):
        problems.append(f"default ONNX weights missing: {ONNX_DEFAULT_WEIGHTS}")

    clf_dir = PROJECT_ROOT / "classification_model" / "conv1d_phrase_model"
    if not (clf_dir / "metadata.json").is_file():
        problems.append(f"classifier model missing: {clf_dir}")

    if problems:
        return _fail("\n    ".join(problems))
    print("ok: model files present")
    return True


def test_nn_optionality_is_wired():
    """The availability flags exist, so the UI can grey buttons out."""
    from cannotate import NN_AVAILABLE
    from cannotate.spectrogram_viewer import TORCH_AVAILABLE

    print(f"ok: NN flags present (NN_AVAILABLE={NN_AVAILABLE}, "
          f"TORCH_AVAILABLE={TORCH_AVAILABLE})")
    if not TORCH_AVAILABLE:
        print("   note: torch is absent here, so NN buttons will be disabled.")
    return True


def test_viewer_builds_its_widgets():
    """The annotator window constructs, and NN buttons match torch availability.

    Building the widget tree is the cheapest way to catch a broken row: an
    empty architecture dropdown, a bad ``state=`` value, a widget referring to
    something the import guard set to None.
    """
    import tkinter as tk
    from cannotate.spectrogram_viewer import (
        IntegratedSpectrogramViewer, TORCH_AVAILABLE)

    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"skip: no display available ({e})")
        return True

    root.withdraw()  # build the widgets without showing a window
    try:
        IntegratedSpectrogramViewer(root)
    except Exception as e:
        root.destroy()
        return _fail(f"viewer failed to build: {e!r}")

    wanted = {"▶ Run NN Segmentation", "Classify 999 Segments",
              "⚡ Full Auto Annotate"}
    states = {}

    def walk(widget):
        for child in widget.winfo_children():
            try:
                text = child.cget("text")
            except (tk.TclError, AttributeError):
                text = None
            if text in wanted:
                states[text] = str(child.cget("state"))
            walk(child)

    walk(root)
    root.destroy()

    expected = "normal" if TORCH_AVAILABLE else "disabled"
    wrong = {label: states.get(label) for label in wanted
             if states.get(label) != expected}
    if wrong:
        return _fail(f"NN buttons should be '{expected}' but got {wrong}")
    print(f"ok: viewer builds, NN buttons are '{expected}'")
    return True


def _synthetic_song(path, fs=44100, seconds=2.0):
    """Four tone bursts on a quiet floor — enough for a segmenter to find edges."""
    import numpy as np
    from scipy.io import wavfile

    t = np.arange(0, seconds, 1.0 / fs)
    y = np.zeros_like(t)
    for start, stop, f0 in [(0.2, 0.5, 3000), (0.7, 1.0, 5000),
                            (1.2, 1.5, 4000), (1.7, 1.9, 6000)]:
        m = (t >= start) & (t < stop)
        y[m] += np.sin(2 * np.pi * f0 * t[m]) * np.hanning(int(m.sum()))
    rng = np.random.default_rng(13)
    y = y + rng.normal(0, 0.01, y.shape)
    wavfile.write(path, fs, (y * 8000).astype(np.int16))
    return fs


def test_nn_inference_actually_runs():
    """Segmentation and classification produce output, not just enabled buttons.

    ``test_viewer_builds_its_widgets`` only proves the NN buttons are not
    greyed out. That passes on a machine where torch imports but the models
    are unreadable or the wrong architecture, which is exactly the state a
    half-finished setup leaves behind. This runs the models.

    An absent optional backend is reported and skipped; one that is present
    but broken fails. Checking that every package is *installed* is the setup
    script's job, not this file's.

    The song is synthetic tone bursts, not birdsong, and both segmentation
    models score most of it above threshold. So the segment counts printed
    here say nothing about accuracy -- they only show the models ran. Judging
    a model's output needs real recordings.
    """
    import tempfile

    try:
        import torch
    except ImportError as e:
        print(f"skip: no torch, NN features are unavailable ({e})")
        return True

    sys.path.insert(0, str(PROJECT_ROOT))
    import numpy as np
    from scipy.io import wavfile

    from bird_song_detector.model_zoo import ConvMLP
    from bird_song_detector.nn_segmentation import InferenceToAnnotationData
    from cannotate.cannotate_utils import NN_WEIGHTS_DIR, ONNX_DEFAULT_WEIGHTS
    from cannotate.onnx_segmentation import OnnxSegmentationModel

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "bird_0001_2024_07_05_06_58_44.wav")
        fs = _synthetic_song(wav_path)
        device = torch.device("cpu")

        # ── PyTorch segmentation weights ──────────────────────────
        pth = os.path.join(NN_WEIGHTS_DIR, "weights_best_epoch4_vacc95.pth")
        if not os.path.isfile(pth):
            return _fail(f"segmentation weights missing: {pth}")
        try:
            model = ConvMLP(input_size=256)
            model.load_state_dict(torch.load(pth, map_location=device))
            segments = InferenceToAnnotationData(model, device).annotatate_data(wav_path)
        except Exception as e:
            return _fail(f"PyTorch segmentation failed: {e!r}")
        if not isinstance(segments, list):
            return _fail(f"segmentation returned {type(segments).__name__}, expected list")
        print(f"ok: PyTorch segmentation ran ({len(segments)} segments)")

        # ── ONNX segmentation weights ─────────────────────────────
        try:
            import onnxruntime  # noqa: F401
        except ImportError as e:
            print(f"skip: no onnxruntime, .onnx weights unavailable ({e})")
        else:
            if not os.path.isfile(ONNX_DEFAULT_WEIGHTS):
                return _fail(f"ONNX weights missing: {ONNX_DEFAULT_WEIGHTS}")
            try:
                onnx_model = OnnxSegmentationModel(ONNX_DEFAULT_WEIGHTS)
                onnx_segments = InferenceToAnnotationData(
                    onnx_model, device).annotatate_data(wav_path)
            except Exception as e:
                return _fail(f"ONNX segmentation failed: {e!r}")
            print(f"ok: ONNX segmentation ran ({len(onnx_segments)} segments)")

        # ── Phrase classifier ─────────────────────────────────────
        from classification_model.segment_classifier import SegmentClassifier
        model_dir = os.path.join(PROJECT_ROOT, "classification_model",
                                 "conv1d_phrase_model")
        if not os.path.isdir(model_dir):
            return _fail(f"classification model missing: {model_dir}")
        try:
            clf = SegmentClassifier.load(model_dir, device=device, verbose=False)
            _, audio = wavfile.read(wav_path)
            preds = clf.classify_segments(
                np.asarray(audio, dtype=np.float32), fs,
                [(0.2, 0.5, 999), (0.7, 1.0, 999)])
        except Exception as e:
            return _fail(f"phrase classification failed: {e!r}")
        labels = [p.label for p in preds]
        if len(labels) != 2:
            return _fail(f"classifier returned {len(labels)} predictions, expected 2")
        print(f"ok: phrase classifier ran (labels {labels})")

    return True


def main():
    tests = [
        test_gui_imports_headlessly,
        test_no_imports_escape_the_project,
        test_vendored_subset_is_lean,
        test_model_files_are_present,
        test_nn_optionality_is_wired,
        test_viewer_builds_its_widgets,
        test_nn_inference_actually_runs,
    ]
    results = [t() for t in tests]
    passed, total = sum(1 for r in results if r), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
