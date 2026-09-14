"""Paths, labels and parameters for the cross-correlation sandbox.

Everything tunable lives here so the experiment scripts stay declarative.
See docs/design/2026-09-14-xcorr-validation-design.md for the rationale
behind each choice.
"""
import os

# ── Data ─────────────────────────────────────────────────────────────

ANNOTATION_FILE = ("/Users/nitzankarby/Desktop/masters/yarden cohen/dev/"
                   "behavioral_experiment_syntax_manipulation/"
                   "LBRK219_auto_phase_level_only_verified.json")

AUDIO_ROOT = "/Volumes/Extreme Pro/LBRK219RT"

# Wavs live at <AUDIO_ROOT>/<date>/chop_data/wav/<file_name>.
WAV_SUBDIR = os.path.join("chop_data", "wav")

# Written by cache.py / templates.py, read by everything else.  Kept out of the
# repo (large, and reproducible) but in a *stable* place: a session scratchpad
# or /tmp is wiped on reboot, and nobody can retype the path.
#
# The label set is part of the directory name, automatically.  Sharing one
# cache between label sets would silently score phrases against templates for
# labels the run does not contain, so the separation must not depend on the
# caller remembering to pass a different path.
_CACHE_ROOT = os.environ.get(
    "XCORR_CACHE_DIR",
    os.path.expanduser("~/.cache/cannotate-xcorr"))

# ── Labels under test ────────────────────────────────────────────────
#
# 4/5/6 are the originally requested set; 9/10/15/16 were added because
# duration cannot separate them (9 vs 10 differ by 50 ms with ~100 ms
# spread; 6/15/16 all sit near 1.8 s).  With all seven, a duration-only
# classifier scores 52.6% — so the spectral evidence has real work to do.
LABELS = [4, 5, 6, 9, 10, 15, 16]

# Override with e.g. XCORR_LABELS=all, or a comma list.  "all" widens the test
# to every label with enough instances, which matters because a closed set of
# seven is an easy question: real error correction must choose among all of the
# bird's syllable types, not a hand-picked few.
if os.environ.get("XCORR_LABELS"):
    _spec = os.environ["XCORR_LABELS"].strip()
    LABELS = None if _spec == "all" else [int(x) for x in _spec.split(",")]

# When LABELS is None, keep every label with at least this many instances.
MIN_INSTANCES_PER_LABEL = 30

if os.environ.get("XCORR_CACHE_DIR"):
    CACHE_DIR = _CACHE_ROOT                     # explicit path wins, as given
elif LABELS is None:
    CACHE_DIR = os.path.join(_CACHE_ROOT, "all")
else:
    CACHE_DIR = os.path.join(_CACHE_ROOT,
                             "-".join(str(l) for l in sorted(LABELS)))

PATCHES_NPZ = os.path.join(CACHE_DIR, "patches.npz")
TEMPLATES_NPZ = os.path.join(CACHE_DIR, "templates.npz")
FIGURE_DIR = os.path.join(CACHE_DIR, "figures")

# ── Spectrogram ──────────────────────────────────────────────────────
#
# Same DPSS two-taper parameters as the viewer, but WITHOUT its 5th/96th
# percentile clip.  That clip is non-linear and computed over a whole
# recording, so it biases cross-file comparison in a way ZNCC's z-scoring
# cannot undo.  ZNCC supplies all the normalisation this path needs.
NFFT = 512
# 2 ms, not the viewer's 1 ms.  The analysis window is 512 samples = 11.6 ms at
# 44.1 kHz, so consecutive 1 ms frames overlap by 91% and carry almost no
# independent information; 2 ms halves the cache for no measurable loss.
HOP_SECONDS = 0.002
DPSS_NW = 4
DPSS_KMAX = 2

# Band applied once at cache time, purely to bound the cache size.  Kept wide:
# the modulation profile of a real recording still carries 95% of its mass
# below ~16 kHz but only 2% below 250 Hz, so this drops rumble and little else.
# The *discriminative* band is chosen per label at scoring time.
CACHE_BAND = (250.0, 16000.0)

# Patches kept per label.  200 x 7 labels is ~670 MB of float32 and gives
# confusion-matrix cells with ample precision; set to 0 to keep everything.
MAX_PER_LABEL = 200

# ── Templates ────────────────────────────────────────────────────────

TEMPLATE_MAX_SECONDS = 0.200        # the 200 ms cap
N_CANDIDATES = 30                   # phrases drawn per label
N_TEMPLATES_KEPT = 1                # raise to 3 to probe for bimodality

# ── Scoring ──────────────────────────────────────────────────────────

# Per-label frequency band: cumulative mass of the label's time-modulation
# profile between these percentiles.  A single global band handicaps
# whichever label sits furthest from the average.
BAND_PERCENTILES = (5.0, 95.0)
GLOBAL_BAND = None                  # (fmin, fmax) to override the per-label band

# A phrase shorter than this many template-lengths has too few lags for a
# stable sidelobe estimate; it is reported as "insufficient support"
# rather than silently skipped.
MIN_SUPPORT_RATIO = 3.0

# ── Experiment ───────────────────────────────────────────────────────

SPLIT_BY = "date"                   # "date" (primary) or "file" (robustness)
RANDOM_SEED = 0
