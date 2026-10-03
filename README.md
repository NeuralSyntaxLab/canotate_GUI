# Cannotate

A GUI for annotating bird-song recordings: browse a bird's WAV files by date,
segment songs into syllables, label them, and save everything to a single
annotation JSON.

This is the annotation tool on its own. It was extracted from the
`birdsong-preprocessing-gui` project, which also carried training scripts, UMAP
workflows, ViT experiments and format converters — none of which are here.

## Setting it up on a machine that has no Python

Double-click `setup/install_mac.command` (macOS) or
`setup/install_windows.bat` (Windows).

It builds Cannotate a private Python under `~/.cannotate` —
`%LOCALAPPDATA%\Cannotate` on Windows — with everything the tool needs,
including torch and onnxruntime, so the NN buttons work. It bootstraps
micromamba, so no Python has to exist beforehand, nothing goes on PATH, and no
administrator rights are needed. About 250 MB to download; a few minutes.

Running it again rebuilds the environment from scratch, which is the fix when
something breaks. It ends by running [`smoke_test.py`](smoke_test.py), so a
successful run means the models actually load and run, not just that the files
copied.

[`docs/INSTALL.md`](docs/INSTALL.md) is the same thing written for someone who
does not program — that is the one to send to an annotator.

If you already have a conda environment with the right packages, you do not
need any of this; `launch.py` will find it.

## Starting it

**Do not open a terminal.** Double-click one of these:

| | macOS | Windows |
|---|---|---|
| **No console window** | `Cannotate.app` | `Cannotate.vbs` |
| Console window alongside | `Cannotate.command` | `Cannotate.bat` |

The console-window versions do the same thing but leave a window open showing
progress and errors — useful when something is not working.

All four first look in `launcher.cfg` for an interpreter — the one the setup
script recorded — and run it directly. They read that file with shell
built-ins rather than Python, because on a machine with no Python there is
nothing to read it with. On macOS this matters more than it looks:
`/usr/bin/python3` exists on every Mac but is a stub that prompts to install
the Command Line Tools.

Failing that, they fall back to a Python on PATH and let
[`launch.py`](launch.py) search the machine, remembering what it finds in
`launcher.cfg`.

If nothing suitable is found, you are pointed at the setup script above.

> **macOS, first run:** right-click `Cannotate.app` → **Open** → **Open**.
> Gatekeeper blocks double-clicking an unsigned app the first time only.

### Pointing it at a specific Python

Copy `launcher.cfg.example` to `launcher.cfg` and set the path:

```ini
[cannotate]
python = /opt/anaconda3/envs/moove_env/bin/python
```

`CANNOTATE_PYTHON=/path/to/python` works too, and takes precedence over
everything except `launcher.cfg`.

## Requirements

The setup script above installs all of these, so this section is only for
setting an environment up by hand.

Required — the GUI will not start without them:

```
numpy  scipy  matplotlib  soundfile  sounddevice  scikit-image  tkinter
```

`tkinter` is not a pip package; it ships with Python. On macOS use the
python.org installer or `brew install python-tk`; on Windows tick
"tcl/tk and IDLE" in the installer; on Linux `apt install python3-tk`.

Optional — only the NN-assisted buttons need these:

```
torch  onnxruntime  scikit-learn  pandas
```

Without them the tool opens normally and every manual feature works. The three
NN buttons (**Run NN Segmentation**, **Classify 999 Segments**, **Full Auto
Annotate**) are greyed out, with a note saying why.

Install torch from conda-forge rather than pip where you can: the CPU build is
26 MB against the 328 MB the pip wheel unpacks to, because it carries no CUDA
payload. On a Mac none of that payload could ever be used anyway.

```
pip install -r requirements.txt
```

## What is in here

| Path | |
|---|---|
| `Cannotate_GUI.py` | entry point — the date/file browser window |
| `cannotate/` | the tool: spectrogram viewer, navigator, DTW suggestions, template browser |
| `utils.py`, `colors.py` | annotation-JSON helpers and the palette generator |
| `models/` | segmentation weights (`.onnx`, `.pth`) |
| `classification_model/` | the 1D-CNN syllable classifier and its trained model |
| `full_annotation_pipeline.py` | segmentation → classification in one pass |
| `bird_song_detector/` | **vendored** — see [PROVENANCE.md](bird_song_detector/PROVENANCE.md) |
| `launch.py` + wrappers | the no-terminal launcher |
| `smoke_test.py` | checks the project is still self-contained |

### Vendored code

`bird_song_detector/` is a trimmed copy of a separately-developed package,
checked in so this project needs no extra install. It carries only `ConvMLP`,
`ClassificationCnn` and `InferenceToAnnotationData`.
[PROVENANCE.md](bird_song_detector/PROVENANCE.md) records the upstream commit,
what was changed, and how to re-sync it.

### Model weights

`models/` holds the segmentation weights, and the GUI's **Weights** dropdown
lists whatever it finds there. Drop new `.pth` / `.onnx` files in to make them
selectable. To use a shared weights folder instead:

```
CANNOTATE_MODELS_DIR=/path/to/weights
```

The syllable classifier is separate — it lives in
`classification_model/conv1d_phrase_model/` and is chosen with the
**Classifier model** box in the GUI.

## Pick one environment and stay in it

The tool caches syllable templates next to your annotation JSON
(`*.templates_cache.pkl`). Those caches are pickles, and pickles are not
portable across NumPy major versions: a cache written under NumPy 2.x cannot be
read under NumPy 1.x. An unreadable cache is handled gracefully — it is ignored
and rebuilt — but rebuilding a large dataset takes about a minute, and the
rebuild overwrites the cache, so alternating between environments means paying
that cost every time.

If you have several usable conda environments, pin one in `launcher.cfg` so
this never comes up.

## Checking the install

```
python smoke_test.py                  # is this project still self-contained?
python tests/test_templates_cache.py  # can a bad cache still break startup?
```

`smoke_test.py` verifies the GUI imports, that no module is secretly loading
from the old project, that the vendored subset has not regrown its training
dependencies, that the model files are present, and that the viewer builds its
widgets. Run it after moving the project or changing anything under
`bird_song_detector/`.
