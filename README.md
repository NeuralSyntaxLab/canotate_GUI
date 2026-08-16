# Cannotate

A GUI for annotating bird-song recordings: browse a bird's WAV files by date,
segment songs into syllables, label them, and save everything to a single
annotation JSON.

This is the annotation tool on its own. It was extracted from the
`birdsong-preprocessing-gui` project, which also carried training scripts, UMAP
workflows, ViT experiments and format converters — none of which are here.

## Starting it

**Do not open a terminal.** Double-click one of these:

| | macOS | Windows |
|---|---|---|
| **No console window** | `Cannotate.app` | `Cannotate.vbs` |
| Console window alongside | `Cannotate.command` | `Cannotate.bat` |

The console-window versions do the same thing but leave a window open showing
progress and errors — useful when something is not working.

All four run [`launch.py`](launch.py), which finds a Python with the right
packages, remembers it in `launcher.cfg`, and starts the GUI. The first launch
takes a few seconds while it searches; later ones are immediate.

If nothing suitable is found, a dialog tells you exactly what is missing and
gives you the `pip install` line to fix it.

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
