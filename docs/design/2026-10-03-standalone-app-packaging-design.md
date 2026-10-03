# Standalone app packaging — design

*2026-10-03*

> **Superseded on 2026-10-03, before any of it was built.** Kept for the
> measurements and for the bug it turned up.
>
> This design assumed the app would be sent to someone unreachable, which is
> what makes a self-contained bundle worth its cost. It is not: there is one
> annotator, on one Mac, who can be set up in person. Two facts then undid it.
>
> Dropping torch was the whole point of the ONNX-only plan, and the plan was
> to save download size. But conda-forge's CPU `pytorch` build is 26 MB, not
> the 328 MB the pip wheel unpacks to, because it carries no CUDA payload. A
> complete environment — torch, onnxruntime, scipy, the lot — is a 249 MB
> download, which is what the ONNX-only *bundle* was estimated at. The size
> argument for giving up PyTorch evaporated once the right number was
> measured.
>
> And new `.pth` weights get trained regularly here, so ONNX-only would have
> taxed every retrain with an export step, permanently, to buy nothing.
>
> What was built instead: `setup/install_mac.command`, which bootstraps
> micromamba and builds that environment on the annotator's Mac. See
> [`INSTALL.md`](../INSTALL.md).
>
> The double-normalization bug this document describes in "Fixing a
> double-normalization bug on the way" was real and is fixed — it had nothing
> to do with packaging. The size table below is accurate but measures pip
> wheels, which is the wrong package source for this project.

## Goal

Ship Cannotate as a self-contained application for macOS and Windows that a
non-programmer installs and double-clicks, with the NN features working out of
the box and **no Python on their machine**.

Today's `Cannotate.app` / `Cannotate.bat` are not applications. They are thin
wrappers around [`launch.py`](../../launch.py), which hunts the machine for a
Python that can import numpy, scipy, matplotlib, soundfile, sounddevice,
tkinter and scikit-image, and — when it finds none — puts up a dialog with a
`pip install` line. That dialog is the end of the road for the intended user.
The NN buttons add a second cliff: torch is optional, so on a machine that has
Python but not torch the window opens with a third of its controls greyed out.

The target user has neither Python nor torch and will not be installing either.

## Constraints that shaped this

- **Apple Silicon only.** arm64. Intel Macs are out of scope.
- **Windows is built in CI.** No tool cross-compiles a Windows `.exe` from
  macOS, and there is no Windows machine in the loop, so GitHub Actions builds
  it on `windows-latest`.
- **Unsigned, delivered by download link.** No Apple Developer ID. This has a
  consequence severe enough to get its own section below.
- **torch is not shipped.** See the next section.

## Decision: ONNX only, no torch in the bundle

torch is 328 MB installed. A PyInstaller bundle carrying it lands somewhere
between 500 MB and 1 GB per platform; without it the bundle is roughly
180–250 MB. For an app delivered over a Drive link to a non-programmer, that
difference is the difference between "download it" and "let me come to your
office with a USB stick".

The weighed alternative was to bundle torch and change almost no code. It was
rejected on download size. The costs of not bundling it are real and are
addressed explicitly in "What this costs" below.

The project is already most of the way there. `models/` ships
`weights_best_epoch4_vacc91.onnx`, and both
[`spectrogram_viewer.py`](../../cannotate/spectrogram_viewer.py) and
[`full_annotation_pipeline.py`](../../full_annotation_pipeline.py) already carry
an `OnnxSegmentationModel` wrapper. What remains is the phrase classifier, which
is still a `.pt`, and the handful of torch calls around both models.

### The three seams

torch survives in exactly three places on the inference path, and each is
already a numpy boundary.

**1. `bird_song_detector/_helpers.py`.** `load_from_wav` returns torch tensors;
`normalize_per_segment` is five lines of torch reductions. Both become numpy.

One trap: `load_from_wav` builds its segments as `torch.float16` and
`run_inference` casts back with `.float()`. That round-trip quantises the
waveform, and the trained model's behaviour is calibrated against it. The numpy
port must do the same `astype(np.float16).astype(np.float32)` or predictions
shift. This is the single most likely way to get a silently-wrong port.

**2. `InferenceToAnnotationData.run_inference`.** One `torch.no_grad()` block
with a `torch.sigmoid` and a threshold. `torch.sigmoid` becomes
`1.0 / (1.0 + np.exp(-x))`; `no_grad` disappears because onnxruntime has no
autograd. The whole-file single forward pass is preserved as-is.

**3. The phrase classifier.** `SegmentClassifier` touches `Conv1DClassifier`
through a surface of exactly three things —
`predict_proba(np.ndarray) -> np.ndarray`, `.label_encoder`, and
`.normalization_config`. `NumpyLabelEncoder` is already pure numpy, and
`_preprocess_segments` already delegates to `normalize_segments`, which is also
pure numpy. So a new `OnnxPhraseClassifier` honouring that same surface is
about sixty lines, and `SegmentClassifier` itself does not change.

`cnn1d_torch_train.py`, `bird_song_detector/model_zoo.py` and
`classification_model/model_zoo/` stay in the repo untouched. They are training
code; they are simply excluded from the bundle. Training on a developer machine
with torch installed works exactly as it does today.

### Fixing a double-normalization bug on the way

There are two different `OnnxSegmentationModel` classes in the project and they
are not equivalent:

- [`spectrogram_viewer.py`](../../cannotate/spectrogram_viewer.py) passes its
  input straight to the ONNX session.
- [`full_annotation_pipeline.py`](../../full_annotation_pipeline.py) has a
  `preprocess_input` that squares the input and z-scores it before the session.

Both are wrapped by `InferenceToAnnotationData.run_inference`, which *already*
applies `normalize_per_segment` — square, then z-score. So **Run NN Segmentation
normalizes once and Full Auto Annotate normalizes twice**, and Full Auto
Annotate has been feeding the segmenter wrongly-scaled input.

`normalize_per_segment` is the correct behaviour: it is what the upstream
`bird_song_detector` training loop applied, so it is what the weights were
trained against. `preprocess_input` is therefore removed, and the two classes
are consolidated into a single `OnnxSegmentationModel` that passes input
through. This changes Full Auto Annotate's output — for the better, but
visibly — and should be called out in the release notes.

### Dropping scikit-image and OpenCV too

[`colors.py`](../../colors.py) needs exactly one function, `rgb2lab`, and gets it
from OpenCV when present and scikit-image otherwise. opencv-python is ~90 MB and
scikit-image is ~60 MB plus networkx, imageio and tifffile.

sRGB→Lab is about fifteen lines of numpy: linearise the gamma curve, multiply by
the sRGB→XYZ matrix, divide by the D65 white point, apply the cube-root/linear
piecewise `f`, and combine. It gets a test asserting agreement with
`skimage.color.rgb2lab` to 1e-6 on a dense grid, skipped when scikit-image is
absent. Both libraries then leave the dependency set entirely, taking the
`skimage or cv2` fork in `launch.py`'s `REQUIRED` with them.

## ONNX artifacts are committed, not built

A new `tools/export_onnx.py` is run once, by hand, on a machine that has torch.
It writes `cnn1d_model.onnx` next to each existing `cnn1d_model.pt` —
`classification_model/conv1d_phrase_model/` and the two
`models/conv1d_phrase_model_*/` directories — and those files are committed.
Each is a few MB.

Exporting in CI would mean installing torch on every build: slower, and it would
make the bundle's correctness depend on a toolchain the bundle does not contain.
Committing the artifacts keeps CI torch-free and makes the exact bytes that ship
reviewable in git.

The segmentation `.onnx` already exists and is reused unchanged.

## Equivalence testing

This is what makes dropping torch safe rather than hopeful. `tests/test_onnx_equivalence.py`
is skipped wholesale when torch is unavailable, so it runs on a developer
machine and never blocks a CI build that has no torch by design.

For each of the segmentation model and the phrase classifier, over both
synthetic arrays and real WAV segments:

- raw logits from the torch path and the ONNX path agree to 1e-4;
- the *final decisions* are identical — the binary segmentation mask for the
  segmenter, the `inverse_transform`ed labels for the classifier. Logit
  tolerance is informative, but label identity is the property that matters.

Plus one end-to-end test: `FullAnnotationPipeline` over a real WAV via the torch
path and via the ONNX path must produce identical segment lists. This test is
written *before* `preprocess_input` is removed, against the single-normalization
behaviour, so it pins the bug fix as well as the port.

## Packaging

PyInstaller, `--onedir`, one shared `cannotate.spec` driven by the platform.
`--onefile` would re-extract ~200 MB to a temp directory on every launch, a
30-second stall each time the user opens the app; `--onedir` starts in seconds
and the clutter is hidden inside the `.app` on macOS and behind the installer on
Windows.

| | macOS | Windows |
|---|---|---|
| Artifact | `Cannotate-<version>-arm64.dmg` | `Cannotate-<version>-Setup.exe` |
| Contents | `Cannotate.app`, an `/Applications` symlink, `READ ME FIRST.rtf` | Inno Setup installer → `%LOCALAPPDATA%\Programs\Cannotate`, Start Menu and desktop shortcuts |

Size budget, from measured on-disk sizes: numpy 36 MB, scipy 99 MB, matplotlib
30 MB, onnxruntime ~61 MB of binaries, soundfile 3 MB, plus CPython and Tcl/Tk.
PyInstaller drops tests and headers, so the expectation is 180–250 MB. scipy is
the largest single item and only `scipy.io`, `scipy.signal` and `scipy.fft` are
used, but `scipy.signal` pulls `linalg`, `special` and `sparse` behind it, so
trimming it is a measured optimisation to attempt after the build works, not a
promise made up front.

PyInstaller needs explicit help with: `sounddevice` (the PortAudio binary),
`soundfile` (`_soundfile_data`/libsndfile), `matplotlib`'s `backend_tkagg`,
`scipy.io.wavfile`, onnxruntime's execution providers, and Tcl/Tk data files.

### Resource paths

`_PROJECT_ROOT` in [`cannotate_utils.py`](../../cannotate/cannotate_utils.py) is
`__file__`-relative, which inside a frozen app resolves into the bundle. A
`resource_root()` helper in `cannotate_utils` returns `sys._MEIPASS` when
`sys.frozen` is set and the project root otherwise; `MODELS_DIR` and `CLASSIFICATION_MODEL_DIR` route
through it. `CANNOTATE_MODELS_DIR` keeps overriding, so a lab can still point
the app at a shared weights folder.

### Writable state leaves the bundle

An installed app directory is not writable, and on macOS writing into a signed
bundle breaks it. `launcher.cfg` becomes obsolete — there is no interpreter to
search for — and `cannotate.log` moves to `~/Library/Logs/Cannotate/` on macOS
and `%LOCALAPPDATA%\Cannotate\logs\` on Windows.

### Entry point

A new `app_main.py` is the frozen entry point. It configures file logging to the
per-user log directory, installs a `sys.excepthook` that shows the traceback in
a Tkinter dialog, and then starts the GUI. The excepthook is not optional: a
frozen GUI application has no console, so without it an uncaught exception makes
the window vanish with no explanation anywhere the user can see.

It also accepts `--self-test`, which imports every bundled module, loads both
ONNX models, runs inference over a synthetic WAV, and exits 0 or 1 without
opening a window.

[`launch.py`](../../launch.py) stays, and keeps its interpreter search, for
developers running from a checkout. It is not part of the bundle. Its `REQUIRED`
list loses the `("skimage", "cv2")` entry, and `torch` moves from `OPTIONAL` to
"needed only for training and ONNX export".

## CI

`.github/workflows/build.yml`, matrix over `macos-14` and `windows-latest`,
triggered by a `v*` tag and by manual dispatch. Each job: checkout, Python 3.11,
install `requirements-app.txt` (the runtime set — no torch, no scikit-image, no
OpenCV), run `pytest` (the equivalence and rgb2lab tests self-skip, since
neither torch nor scikit-image is installed there), run PyInstaller, run the
frozen binary's `--self-test`, package to `.dmg` or run Inno Setup, upload to
the Release.

The `--self-test` step is the real gate. A Tkinter GUI cannot be smoke-tested on
a headless runner, but almost every way PyInstaller breaks — a missing hidden
import, an absent data file, an unbundled native library — shows up as an
import or model-load failure, which `--self-test` catches on the actual target
platform.

**Prerequisite:** the repository has no git remote. It must be created on GitHub
and pushed before any of this runs. A public repository gets unlimited Actions
minutes; a private one gets 2000 per month with macOS billed at 10×, which is
roughly thirteen macOS builds.

## Order of work

macOS is built and verified locally first — PyInstaller run by hand, the `.app`
launched, a real bird folder annotated, NN segmentation and Full Auto Annotate
exercised against real WAVs — before any CI is written. CI is then the
automation of a process already known to work, instead of a debugging session
conducted through a twenty-minute feedback loop on a remote runner.

So: de-torch and prove equivalence → build and verify the macOS app by hand →
wire CI for both platforms → installers and guide.

## macOS Gatekeeper, and why the guide matters

On Apple Silicon an unsigned application that arrives over the internet does not
produce a warning the user can click past. It produces **"Cannotate is damaged
and can't be opened. You should move it to the Trash."** Right-click → Open does
not help. The quarantine attribute must be removed:

```
xattr -dr com.apple.quarantine /Applications/Cannotate.app
```

There is no free way around this. A $99/year Apple Developer ID plus
notarisation in CI would remove it entirely, and the workflow is structured so
that signing can be switched on later by adding secrets rather than rewritten.
Until then, that command is the first thing in `READ ME FIRST.rtf` inside the
DMG and in `docs/INSTALL.md`, written for someone who has never opened Terminal:
what to click, what to paste, what it will look like when it works.

Windows is milder — SmartScreen shows "Windows protected your PC" with a "More
info → Run anyway" link — but gets the same screenshot-level treatment.

## What this costs

[`spectrogram_viewer.py`](../../cannotate/spectrogram_viewer.py)'s weights
picker currently lets the user browse to any `.pth`, `.pt` or `.onnx` file. The
frozen app cannot read PyTorch weights at all.

So in a frozen build the file dialog filters to `.onnx`, and selecting a
PyTorch file reports: *"This build cannot read PyTorch weights. Export the model
to ONNX with tools/export_onnx.py and select the .onnx file instead."* The
capability is explained rather than silently absent — the same principle the
greyed-out NN buttons followed in the original extraction.

The practical consequence for the lab: **new segmentation or classification
weights now need an ONNX export step before the packaged app can use them.**
`tools/export_onnx.py` exists for precisely this, and running it is a developer
task on a machine that has torch.

## Out of scope

- Intel macOS and Linux builds.
- Code signing and notarisation (designed for, not implemented).
- Auto-update.
- Any change to annotation behaviour beyond the double-normalization fix.
