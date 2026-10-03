"""
Error correction — find suspicious *phrases* across the whole annotation
file and fix them (relabel / split) from an embedded spectrogram view.

A *phrase* is a run of consecutive segments that share the same label
inside one song.  Outlier detection is pluggable: every method is a
subclass of :class:`OutlierDetector`, so adding a new one means writing
a subclass and appending it to :data:`DETECTOR_CLASSES` — the browser
window builds its UI from that registry alone.
"""
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from colors import distinguishable_colors
from . import template_store
from .cannotate_utils import read_wav
from .template_browser import _compute_spectrogram_norm, _parse_label


# Labels the annotation format uses for "no label yet".
UNLABELED = (None, 999)

# Segment start/end times are matched back to the JSON with this tolerance
# (seconds) — the same idiom template_browser.py uses for relabeling.
TIME_TOL = 1e-3

# Seconds of context shown on each side of a phrase in the detail view.
CONTEXT_PAD = 0.3

# Matplotlib mouse buttons. A click on the spectrogram acts on which button
# was pressed rather than on a mode: modifier keys are not usable here because
# matplotlib only reports them while its canvas holds keyboard focus.
LEFT_BUTTON = 1
RIGHT_BUTTON = 3

# The label the "e" shortcut assigns.
ZERO_LABEL = 0


# ── Phrase model ─────────────────────────────────────────────────────

@dataclass
class Phrase:
    """A run of consecutive same-label segments within one song.

    ``segments`` holds ``(start, end, label)`` triples rather than list
    indices so that edits elsewhere in the file cannot invalidate it.
    Relabeling part of a phrase makes its labels non-uniform, which is why
    the per-segment label is carried here rather than once per phrase.
    """
    date: str
    file_name: str
    segments: list = field(default_factory=list)

    @property
    def start(self):
        return self.segments[0][0]

    @property
    def end(self):
        return self.segments[-1][1]

    @property
    def duration(self):
        return self.end - self.start

    @property
    def n_segments(self):
        return len(self.segments)

    @property
    def label(self):
        """The phrase's label, or ``None`` once its segments disagree."""
        labels = {seg[2] for seg in self.segments}
        return labels.pop() if len(labels) == 1 else None

    @property
    def label_text(self):
        label = self.label
        return "mixed" if label is None else str(label)


def build_phrases(date, file_name, segments):
    """Group *segments* of one song into phrases.

    Segments are sorted by onset; consecutive ones sharing the same label
    form a phrase.  Unlabeled segments break a run and never join one.
    """
    usable = []
    for seg in segments:
        if not seg or len(seg) < 3:
            continue
        try:
            t0, t1 = float(seg[0]), float(seg[1])
        except (TypeError, ValueError):
            continue
        usable.append((t0, t1, _parse_label(seg[2])))
    usable.sort(key=lambda s: s[0])

    phrases = []
    current = None
    current_label = None
    for t0, t1, label in usable:
        if label in UNLABELED:
            current = None
            continue
        if current is not None and current_label == label:
            current.segments.append((t0, t1, label))
        else:
            current = Phrase(date=date, file_name=file_name,
                             segments=[(t0, t1, label)])
            phrases.append(current)
        current_label = label
    return phrases


def build_all_phrases(db_data):
    """Flatten every song in *db_data* into one dataset-wide phrase list."""
    phrases = []
    for date, entries in db_data.items():
        for entry in entries:
            phrases.extend(build_phrases(date, entry.get("file_name"),
                                         entry.get("segments", [])))
    return phrases


# ── Detector interface ───────────────────────────────────────────────

@dataclass
class Candidate:
    """One label a phrase might be, with the evidence for it.

    ``template`` is the band-limited log-power patch the score came from, kept
    so the detail view can show *why* a label was suggested rather than only
    that it was.
    """
    label: object
    peak: float
    template: object = None
    band: tuple = None

    @property
    def summary(self):
        return f"{self.label} · {self.peak:.3f}"


@dataclass
class OutlierResult:
    phrase: Phrase
    score: float
    reason: str
    detector_key: str

    # Optional, and rendered by the browser only when present, so a detector
    # that has nothing to suggest — the duration z-score has no notion of
    # "which label instead" — needs to know nothing about any of this.
    suggestion: object = None
    candidates: list = field(default_factory=list)
    note: str = ""
    margin: float = float("nan")


@dataclass
class DetectionContext:
    """What a detector may need from the window beyond the phrases themselves.

    Audio arrives two ways because the two consumers differ: ``load_song``
    serves the main process (and reuses the window's one-song cache), while
    ``wav_paths`` is what can cross a process boundary to a scan worker.
    """
    load_song: object = None
    wav_paths: dict = field(default_factory=dict)
    progress: object = None


class OutlierDetector(ABC):
    """Base class for a phrase-outlier detection method.

    Subclasses own their parameter widgets (``build_controls``) and their
    scoring logic (``detect``).  The browser window only ever talks to
    this interface.
    """

    key = ""
    display_name = ""

    @abstractmethod
    def build_controls(self, parent, labels):
        """Create the method's parameter widgets inside *parent*.

        *labels* are the distinct labels present in the dataset.
        """

    def set_default_bank(self, path):
        """Offer the dataset's own template bank, before controls are built.

        Called for every detector; those that do not score against templates
        ignore it.  It is a *default*: a bank the user chose by hand always
        wins, so an experimental bank can still be compared against.
        """

    @abstractmethod
    def detect(self, phrases, context):
        """Score *phrases* and return the outliers, worst first.

        *context* is a :class:`DetectionContext`; a method that needs only
        the annotations ignores it.  Raise :class:`ValueError` with a
        user-facing message when the chosen parameters cannot be used.
        """


def _build_label_picker(parent, labels, row=0):
    """Shared 'apply to label:' dropdown used by the detectors."""
    tk.Label(parent, text="Label:").grid(row=row, column=0, sticky="w")
    combo = ttk.Combobox(parent, values=[str(l) for l in labels],
                         state="readonly", width=8)
    if labels:
        combo.current(0)
    combo.grid(row=row, column=1, sticky="w", padx=(2, 12))
    return combo


def _selected_label(combo, labels):
    choice = combo.get()
    for label in labels:
        if str(label) == choice:
            return label
    raise ValueError("Select a label to analyse first.")


class DurationOutlierDetector(OutlierDetector):
    """Flag phrases whose duration is far from the mean for their label."""

    key = "duration"
    display_name = "Duration (z-score)"

    def __init__(self):
        self._labels = []
        self._label_combo = None
        self._threshold = None

    def build_controls(self, parent, labels):
        self._labels = list(labels)
        self._label_combo = _build_label_picker(parent, labels)
        tk.Label(parent, text="|z| threshold:").grid(row=0, column=2, sticky="w")
        self._threshold = tk.Spinbox(parent, from_=0.5, to=10.0, increment=0.1,
                                     width=5)
        self._threshold.delete(0, tk.END)
        self._threshold.insert(0, "2.5")
        self._threshold.grid(row=0, column=3, sticky="w", padx=2)

    def detect(self, phrases, context=None):
        label = _selected_label(self._label_combo, self._labels)
        try:
            threshold = float(self._threshold.get())
        except ValueError:
            raise ValueError("The z-score threshold must be a number.")

        subset = [p for p in phrases if p.label == label]
        if len(subset) < 3:
            raise ValueError(
                f"Label {label} has only {len(subset)} phrase(s) — "
                "too few for a z-score.")

        durations = np.array([p.duration for p in subset], dtype=float)
        std = durations.std()
        if std == 0:
            raise ValueError(
                f"All {len(subset)} phrases of label {label} have the same "
                "duration — nothing to flag.")
        mean = durations.mean()

        results = []
        for phrase, duration in zip(subset, durations):
            z = (duration - mean) / std
            if abs(z) >= threshold:
                results.append(OutlierResult(
                    phrase=phrase, score=abs(z),
                    reason=(f"{duration:.3f}s, z={z:+.2f} "
                            f"(mean {mean:.3f}s, sd {std:.3f}s)"),
                    detector_key=self.key))
        results.sort(key=lambda r: r.score, reverse=True)
        return results


# How many alternative labels the detail view offers per phrase.  Three fits
# the strip without scrolling, and the sandbox's three-template probe found
# ranks beyond the third contributing almost nothing.
N_CANDIDATES_SHOWN = 3


def cross_correlation_outliers(phrases, bank_path, label, context,
                               margin_threshold=0.0, workers=None):
    """Flag phrases labelled *label* that the template bank disagrees with.

    Every phrase carrying *label* is scored against **every** label's
    templates, so each result arrives with a suggested correct label rather
    than only a complaint.  A phrase is flagged when the bank's winner is not
    *label*, or when the margin — the assigned label's peak ZNCC minus the best
    competitor's — is at or below *margin_threshold*.  Zero therefore means
    "plain disagreement"; the sandbox found the margin histogram bimodal with a
    valley near +0.15, which is why the knob exists.

    Phrases the method cannot advise on are returned too, carrying a ``note``
    and no ``suggestion``.  Dropping them would hide real mislabels behind a
    silent filter, and the user can still relabel them by hand.
    """
    from .xcorr import bank as bank_mod
    from .xcorr import scan as scan_mod

    subset = [p for p in phrases if p.label == label]
    if not subset:
        raise ValueError(f"There are no phrases labelled {label}.")
    if not bank_path or not os.path.exists(bank_path):
        raise ValueError(
            "Choose a template bank (.npz) first — build one with\n"
            "    python -m cannotate.xcorr fit --annotations ... --out t.npz")
    try:
        loaded = bank_mod.Bank.load(bank_path)
    except Exception as e:                                    # noqa: BLE001
        raise ValueError(f"{os.path.basename(bank_path)} is not a readable "
                         f"template bank: {e}")

    by_spec = {}
    specs = []
    for phrase in subset:
        spec = (phrase.date, phrase.file_name, phrase.start, phrase.end, label)
        by_spec.setdefault(spec, phrase)
        specs.append(spec)

    wav_paths = {key: path for key, path in context.wav_paths.items()}
    scored = scan_mod.scan(specs, wav_paths, bank_path, workers=workers,
                           progress=context.progress)

    results = []
    for entry in scored:
        phrase = by_spec.get(entry.spec)
        if phrase is None:
            continue
        candidates = [
            Candidate(label=candidate_label, peak=peak,
                      template=loaded.templates[candidate_label][
                          entry.scores[candidate_label]["template_rank"]],
                      band=loaded.band)
            for candidate_label, peak in entry.ranked_candidates(
                N_CANDIDATES_SHOWN)]

        if entry.outcome == scan_mod.NO_TEMPLATE:
            results.append(OutlierResult(
                phrase=phrase, score=float("inf"),
                reason=f"no template for label {label}",
                detector_key=CrossCorrelationOutlierDetector.key,
                candidates=candidates,
                note=(f"The bank has no template for label {label}, so there "
                      f"is nothing to agree or disagree with. The labels "
                      f"below are the closest matches it does hold — pick "
                      f"one, or use Relabel.")))
        elif entry.outcome == scan_mod.NO_SELF_SCORE:
            fitted = " or ".join(str(c.label) for c in candidates)
            results.append(OutlierResult(
                phrase=phrase, score=float("inf"),
                reason=f"too short to compare with label {label}",
                detector_key=CrossCorrelationOutlierDetector.key,
                candidates=candidates,
                note=(f"This phrase is shorter than label {label}'s own "
                      f"template, so it could never be compared with the "
                      f"label it already has and no suggestion is possible. "
                      + (f"It was short enough to score against {fitted}, "
                         f"shown below, but a short phrase matching a short "
                         f"template is weak evidence — check it by eye."
                         if candidates else
                         "No template was short enough to score it against."))))
        elif entry.outcome == scan_mod.TOO_SHORT:
            results.append(OutlierResult(
                phrase=phrase, score=float("inf"), reason="too short to score",
                detector_key=CrossCorrelationOutlierDetector.key,
                note=("This phrase is too short to correlate against a "
                      "template. Use Relabel, or Split the phrase it belongs "
                      "with.")))
        elif entry.outcome == scan_mod.FILE_ERROR:
            results.append(OutlierResult(
                phrase=phrase, score=float("inf"), reason="audio unavailable",
                detector_key=CrossCorrelationOutlierDetector.key,
                note=(f"The audio for {phrase.file_name} could not be read, "
                      f"so this phrase was not scored.")))
        elif entry.disagrees or entry.margin <= margin_threshold:
            results.append(OutlierResult(
                phrase=phrase, score=-entry.margin, margin=entry.margin,
                reason=(f"looks like {entry.predicted} "
                        f"(margin {entry.margin:+.3f})"),
                detector_key=CrossCorrelationOutlierDetector.key,
                suggestion=entry.predicted, candidates=candidates))
    return results


class CrossCorrelationOutlierDetector(OutlierDetector):
    """Flag phrases whose spectrogram matches another label's template better.

    Graduated from ``sandbox/xcorr`` after measuring 95.6% agreement with
    human labels across 17 labels on a date-level split, against 0.0% with the
    template-to-label mapping shuffled.  See ``cannotate/xcorr/__init__.py``.
    """

    key = "cross_correlation"
    display_name = "Cross correlation"

    def __init__(self):
        self._labels = []
        self._label_combo = None
        self._threshold = None
        self._bank_path = None
        self._bank_label = None

    def build_controls(self, parent, labels):
        self._labels = list(labels)
        self._label_combo = _build_label_picker(parent, labels)

        tk.Label(parent, text="margin ≤").grid(row=0, column=2, sticky="w")
        self._threshold = tk.Spinbox(parent, from_=-1.0, to=1.0, increment=0.01,
                                     width=6, format="%.2f")
        self._threshold.delete(0, tk.END)
        self._threshold.insert(0, "0.00")
        self._threshold.grid(row=0, column=3, sticky="w", padx=2)

        tk.Button(parent, text="Templates…", command=self._choose_bank).grid(
            row=0, column=4, sticky="w", padx=(12, 2))
        self._bank_label = tk.Label(parent, fg="gray40",
                                    text=self._bank_summary())
        self._bank_label.grid(row=0, column=5, sticky="w")

    def set_default_bank(self, path):
        if path and not self._bank_path:
            self._bank_path = path

    def _bank_summary(self):
        if not self._bank_path:
            return "no template bank chosen"
        return os.path.basename(self._bank_path)

    def _choose_bank(self):
        path = filedialog.askopenfilename(
            title="Choose a template bank",
            filetypes=[("Template bank", "*.npz"), ("All files", "*.*")])
        if path:
            self._bank_path = path
            if self._bank_label is not None:
                self._bank_label.config(text=self._bank_summary())

    def detect(self, phrases, context=None):
        label = _selected_label(self._label_combo, self._labels)
        try:
            threshold = float(self._threshold.get())
        except ValueError:
            raise ValueError("The margin threshold must be a number.")
        if context is None:
            raise ValueError("Cross correlation needs access to the audio.")
        return cross_correlation_outliers(phrases, self._bank_path, label,
                                          context, margin_threshold=threshold)


DETECTOR_CLASSES = [DurationOutlierDetector, CrossCorrelationOutlierDetector]


# ── Annotation edits ─────────────────────────────────────────────────

def _find_entry(db_data, date, file_name):
    for entry in db_data.get(date, []):
        if entry.get("file_name") == file_name:
            return entry
    return None


def _matches(seg, t0, t1):
    try:
        return (abs(float(seg[0]) - t0) < TIME_TOL
                and abs(float(seg[1]) - t1) < TIME_TOL)
    except (TypeError, ValueError, IndexError):
        return False


def relabel_phrase(db_data, phrase, new_label, targets=None):
    """Set *new_label* on some or all segments of *phrase*.

    *targets* is a list of segment indices into ``phrase.segments``; all of
    them are relabeled when it is ``None``.  Returns how many changed.  The
    AI-generated flag is dropped, matching what the viewer does when a human
    relabels an AI annotation.
    """
    entry = _find_entry(db_data, phrase.date, phrase.file_name)
    if entry is None:
        return 0
    if targets is None:
        targets = range(phrase.n_segments)

    segments = entry.get("segments", [])
    changed = 0
    for target in targets:
        t0, t1, _ = phrase.segments[target]
        index = next((i for i, seg in enumerate(segments)
                      if _matches(seg, t0, t1)), None)
        if index is None:
            continue
        segments[index] = [t0, t1, int(new_label)]
        phrase.segments[target] = (t0, t1, int(new_label))
        changed += 1
    return changed


def split_phrase(db_data, phrase, split_times):
    """Cut the phrase's segments at *split_times*; return the pieces added.

    Each split time cuts the one segment that contains it, so gaps between
    the phrase's segments are preserved and neighbouring phrases are
    untouched.  Every piece keeps the original label.
    """
    entry = _find_entry(db_data, phrase.date, phrase.file_name)
    if entry is None:
        return 0
    segments = entry.get("segments", [])

    new_phrase_segments = []
    added = 0
    for t0, t1, label in phrase.segments:
        cuts = sorted(t for t in split_times if t0 + TIME_TOL < t < t1 - TIME_TOL)
        bounds = [t0] + cuts + [t1]
        pieces = [(float(a), float(b), label) for a, b in zip(bounds[:-1], bounds[1:])]
        new_phrase_segments.extend(pieces)
        if len(pieces) == 1:
            continue
        index = next((i for i, seg in enumerate(segments)
                      if _matches(seg, t0, t1)), None)
        if index is None:
            continue
        template = segments[index]
        tail = list(template[3:])
        segments[index:index + 1] = [[a, b, template[2]] + tail
                                     for a, b, _ in pieces]
        added += len(pieces) - 1

    phrase.segments = new_phrase_segments
    return added


# ── Browser window ───────────────────────────────────────────────────

def open_error_correction_browser(root, db_data, date_to_wavs,
                                  annotation_file, refresh_counts,
                                  bird_folder=None):
    """Open the Error Correction window over *db_data*.

    ``date_to_wavs`` maps a date to its WAV paths (for the spectrogram),
    ``annotation_file`` is where edits are saved, and ``refresh_counts``
    lets the main window update after an edit.  ``bird_folder`` is where the
    dataset's own template bank lives, so the cross-correlation method starts
    pointed at it instead of at nothing.
    """
    phrases = build_all_phrases(db_data)
    if not phrases:
        messagebox.showinfo(
            "Error Correction",
            "No labeled phrases found in the annotation file.")
        return

    labels = sorted({p.label for p in phrases}, key=str)
    detectors = [cls() for cls in DETECTOR_CLASSES]
    default_bank = template_store.newest_bank_path(bird_folder)
    for detector in detectors:
        detector.set_default_bank(default_bank)

    try:
        palette = distinguishable_colors(1000)
    except Exception:
        palette = None

    top = tk.Toplevel(root)
    top.title("Error Correction")
    top.geometry("1100x700")

    results = []
    index = -1
    split_marks = []
    selected_segments = set()   # indices into the current phrase's segments
    audio_cache = {}

    # ── Controls ───────────────────────────────────────────
    controls = tk.Frame(top)
    controls.pack(side=tk.TOP, fill="x", padx=6, pady=6)

    tk.Label(controls, text="Method:").grid(row=0, column=0, sticky="w")
    method_combo = ttk.Combobox(controls,
                                values=[d.display_name for d in detectors],
                                state="readonly", width=22)
    method_combo.current(0)
    method_combo.grid(row=0, column=1, sticky="w", padx=(2, 12))

    params = tk.Frame(controls)
    params.grid(row=0, column=2, sticky="w")

    def current_detector():
        return detectors[method_combo.current()]

    def rebuild_params(event=None):
        for child in params.winfo_children():
            child.destroy()
        current_detector().build_controls(params, labels)

    method_combo.bind("<<ComboboxSelected>>", rebuild_params)
    rebuild_params()

    status = tk.Label(top, text="", fg="gray30", anchor="w")
    # Why a phrase carries no suggestion, when it carries none.  Kept
    # separate from the status line so the triage hints stay in one place.
    note = tk.Label(top, text="", fg="#8a5a00", anchor="w",
                    justify="left", wraplength=1000)

    # ── Result list + detail view ──────────────────────────
    body = tk.Frame(top)
    body.pack(side=tk.TOP, fill="both", expand=True, padx=6)

    list_frame = tk.Frame(body)
    list_frame.pack(side=tk.LEFT, fill="y")
    tk.Label(list_frame, text="Suspected outliers:").pack(anchor="w")
    listbox = tk.Listbox(list_frame, width=46, height=24, selectmode="browse",
                         exportselection=False)
    listbox.pack(fill="y", expand=True)

    canvas_frame = tk.Frame(body)
    canvas_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=(6, 0))
    fig = plt.Figure(figsize=(7, 4), dpi=100)
    # The phrase, and beside it the template that was matched against it.  The
    # reference axes is hidden unless the detector supplied a candidate, so
    # methods with nothing to show against get the full width.
    ax, ref_ax = fig.subplots(1, 2, width_ratios=[4, 1])
    fig.subplots_adjust(wspace=0.05)
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)

    # ── Candidate strip ────────────────────────────────────
    #
    # Clicking a candidate *previews* its template rather than applying it:
    # the whole point of the reference panel is to be looked at before a
    # label is committed.  Packed before the canvas so that shrinking the
    # window eats into the spectrogram rather than clipping the buttons.
    candidate_frame = tk.Frame(canvas_frame)
    candidate_frame.pack(side=tk.BOTTOM, fill="x", pady=(4, 0))
    canvas.get_tk_widget().pack(side=tk.TOP, fill="both", expand=True)
    candidate_widgets = []
    candidate_choice = tk.IntVar(master=top, value=0)

    def current_candidates():
        result = current_result()
        return list(result.candidates) if result is not None else []

    def selected_candidate_label():
        """The label Apply would assign, or ``None`` when there is no panel."""
        candidates = current_candidates()
        if not candidates:
            return None
        position = max(0, min(candidate_choice.get(), len(candidates) - 1))
        return candidates[position].label

    def on_candidate_click():
        draw_reference()
        canvas.draw()

    def rebuild_candidates():
        for widget in candidate_widgets:
            widget.destroy()
        candidate_widgets.clear()
        candidates = current_candidates()
        if not candidates:
            return
        caption = tk.Label(candidate_frame, text="Looks like:")
        caption.pack(side=tk.LEFT)
        candidate_widgets.append(caption)
        for position, candidate in enumerate(candidates):
            button = tk.Radiobutton(
                candidate_frame, text=candidate.summary, indicatoron=False,
                padx=8, pady=2, value=position, variable=candidate_choice,
                command=on_candidate_click)
            button.pack(side=tk.LEFT, padx=2)
            candidate_widgets.append(button)
        hint = tk.Label(candidate_frame, fg="gray40",
                        text="click to preview · Enter applies")
        hint.pack(side=tk.LEFT, padx=(10, 0))
        candidate_widgets.append(hint)

    def label_color(label):
        try:
            return palette[int(label) - 1] if palette is not None else "orange"
        except Exception:
            return "orange"

    def load_song(date, file_name):
        path = next((p for p in date_to_wavs.get(date, [])
                     if os.path.basename(p) == file_name), None)
        if path is None:
            return None, None
        if path not in audio_cache:
            audio_cache.clear()  # keep at most one song in memory
            fs, y = read_wav(path)
            if y.ndim == 2:
                y = y.mean(axis=1)
            audio_cache[path] = (fs, y)
        return audio_cache[path]

    def current_result():
        if 0 <= index < len(results):
            return results[index]
        return None

    def _list_row(result):
        phrase = result.phrase
        row = (f"{phrase.file_name} — label {phrase.label_text} — "
               f"{phrase.n_segments} seg — {phrase.duration:.3f}s")
        if result.suggestion is not None:
            row += f"  →  {result.suggestion}"
        elif result.note:
            row += "  →  ?"
        return row

    def show_message(text):
        ax.clear()
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
        ax.set_axis_off()
        ref_ax.clear()
        ref_ax.set_axis_off()
        canvas.draw()

    def draw_reference():
        """Show the template behind the selected candidate, if there is one."""
        ref_ax.clear()
        ref_ax.set_axis_off()
        candidates = current_candidates()
        if not candidates:
            return
        candidate = candidates[max(0, min(candidate_choice.get(),
                                          len(candidates) - 1))]
        if candidate.template is None:
            return
        band = candidate.band or (0.0, 1.0)
        ref_ax.set_axis_on()
        ref_ax.imshow(candidate.template, aspect="auto", origin="lower",
                      extent=[0, candidate.template.shape[1],
                              band[0], band[1]],
                      cmap="gray_r")
        ref_ax.set_title(f"template {candidate.label}", fontsize=8)
        ref_ax.set_xticks([])
        # Ticks on the outside edge: on the inside they are drawn over the
        # phrase's own spectrogram, which is the panel that matters.
        ref_ax.yaxis.tick_right()
        ref_ax.tick_params(axis="y", labelsize=6)

    def render():
        result = current_result()
        rebuild_candidates()
        if result is None:
            show_message("Run a detection method to list suspected outliers.")
            status.config(text="")
            note.config(text="")
            return

        note.config(text=result.note)
        phrase = result.phrase
        fs, y = load_song(phrase.date, phrase.file_name)
        if y is None:
            show_message(f"WAV file not found for {phrase.file_name}.")
        else:
            t_lo = max(0.0, phrase.start - CONTEXT_PAD)
            t_hi = min(len(y) / fs, phrase.end + CONTEXT_PAD)
            snippet = y[int(round(t_lo * fs)):int(round(t_hi * fs))]
            ax.clear()
            ax.set_axis_on()
            try:
                S_norm, F, T = _compute_spectrogram_norm(snippet, fs)
                ax.imshow(S_norm, aspect="auto", origin="lower",
                          extent=[t_lo, t_hi, F[0], F[-1]],
                          cmap="gray_r", vmin=0, vmax=1)
            except Exception as e:
                show_message(f"Could not generate spectrogram: {e}")
                return
            for i, (t0, t1, label) in enumerate(phrase.segments):
                color = label_color(label)
                selected = i in selected_segments
                ax.axvspan(t0, t1, facecolor=color, linewidth=0,
                           alpha=0.5 if selected else 0.3)
                if selected:
                    ax.axvline(t0, color="blue", lw=2)
                    ax.axvline(t1, color="blue", lw=2)
                ax.text((t0 + t1) / 2, F[-1] * 0.93, str(label),
                        ha="center", va="top", fontsize=8,
                        color="blue" if selected else "black")
            for t in split_marks:
                ax.axvline(t, color="red", lw=1.5, ls="--")
            ax.set_xlim(t_lo, t_hi)
            ax.set_xlabel("Time (s)")
            # Left-aligned and small: the file names are long, and a centred
            # title on this axes runs over the reference panel beside it.
            ax.set_title(f"{phrase.date} / {phrase.file_name} — "
                         f"label {phrase.label_text} — {result.reason}",
                         fontsize=9, loc="left")
            draw_reference()
            canvas.draw()

        n_selected = len(selected_segments)
        scope = (f"{n_selected} segment(s) selected — Relabel applies to them"
                 if n_selected else
                 "no selection — Relabel applies to the whole phrase")
        if result.suggestion is not None:
            # Accept and Apply deliberately ignore the segment selection: the
            # phrase as a whole is what was scored, so the suggestion is
            # evidence about all of it or none of it.
            hint = (f"a = accept {result.suggestion} · Enter = apply selected "
                    f"· d = dismiss · both relabel the whole phrase")
        elif result.candidates:
            hint = "Enter = apply the selected label to the whole phrase"
        else:
            hint = ("left-click a segment to select it, "
                    "right-click inside one to mark a split")
        status.config(text=(
            f"Outlier {index + 1} of {len(results)}   |   "
            f"{phrase.n_segments} segment(s), {phrase.duration:.3f}s   |   "
            f"{scope}   |   {hint}"))
        if listbox.curselection() != (index,):
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(index)
            listbox.see(index)

    def select(new_index):
        nonlocal index
        if not results:
            return
        index = max(0, min(new_index, len(results) - 1))
        split_marks.clear()
        selected_segments.clear()
        # Every phrase starts on its own best candidate; carrying the previous
        # phrase's choice over would make Enter mean something different for
        # each one.
        candidate_choice.set(0)
        render()

    def on_listbox_select(event=None):
        selection = listbox.curselection()
        if selection and selection[0] != index:
            select(selection[0])

    listbox.bind("<<ListboxSelect>>", on_listbox_select)

    def wav_paths_by_song():
        """``(date, file_name)`` -> path, for detectors that read audio.

        Resolved here because only the window knows where the wavs are, and a
        scan worker cannot be handed a closure.
        """
        return {(date, os.path.basename(path)): path
                for date, paths in date_to_wavs.items() for path in paths}

    def run_detection():
        nonlocal results, index
        detector = current_detector()

        progress_window = tk.Toplevel(top)
        progress_window.title(f"Running {detector.display_name}")
        progress_window.transient(top)
        progress_window.resizable(False, False)
        cancelled = [False]
        progress_text = tk.Label(progress_window, text="Preparing…", width=40)
        progress_text.pack(padx=12, pady=(12, 4))
        bar = ttk.Progressbar(progress_window, length=280, mode="determinate")
        bar.pack(padx=12, pady=4)
        tk.Button(progress_window, text="Cancel",
                  command=lambda: cancelled.__setitem__(0, True)).pack(
            pady=(4, 12))
        # Closing the window means the same as pressing Cancel; leaving it
        # closable but inert would look like a hang.
        progress_window.protocol("WM_DELETE_WINDOW",
                                 lambda: cancelled.__setitem__(0, True))
        # ``update()`` below pumps *every* pending event, so without a grab the
        # user could press Run Detection again mid-scan and start a second one
        # inside the first.
        try:
            progress_window.grab_set()
        except tk.TclError:
            pass

        def report(done, total):
            bar.config(maximum=max(1, total), value=done)
            progress_text.config(text=f"Scanning file {done} of {total}…")
            # The scan holds the Tk thread, so the window only repaints — and
            # only notices Cancel — because it is pumped from here.
            progress_window.update()
            return not cancelled[0]

        context = DetectionContext(load_song=load_song,
                                   wav_paths=wav_paths_by_song(),
                                   progress=report)
        try:
            found = detector.detect(build_all_phrases(db_data), context)
        except ValueError as e:
            progress_window.destroy()
            messagebox.showerror("Error Correction", str(e), parent=top)
            return
        except Exception as e:                                # noqa: BLE001
            progress_window.destroy()
            messagebox.showerror("Error Correction",
                                 f"{detector.display_name} failed:\n{e}",
                                 parent=top)
            return
        finally:
            if progress_window.winfo_exists():
                progress_window.destroy()

        results = found
        index = -1
        split_marks.clear()
        selected_segments.clear()
        listbox.delete(0, tk.END)
        for result in results:
            listbox.insert(tk.END, _list_row(result))
        if not results:
            show_message(
                f"No outliers found with {detector.display_name}."
                + (" (cancelled)" if cancelled[0] else ""))
            status.config(text="")
            note.config(text="")
            return
        select(0)

    tk.Button(controls, text="Run Detection", command=run_detection).grid(
        row=0, column=3, sticky="w", padx=12)

    # ── Segment selection and split marks ──────────────────
    def on_canvas_click(event):
        result = current_result()
        if result is None or event.inaxes is not ax or event.xdata is None:
            return
        phrase = result.phrase
        x = float(event.xdata)
        hit = next((i for i, (t0, t1, _) in enumerate(phrase.segments)
                    if t0 < x < t1), None)

        if event.button == RIGHT_BUTTON:
            tolerance = 0.01 * (phrase.duration + 2 * CONTEXT_PAD)
            near = next((t for t in split_marks if abs(t - x) <= tolerance),
                        None)
            if near is not None:
                split_marks.remove(near)
            elif hit is not None:
                split_marks.append(x)
            else:
                return
        elif event.button == LEFT_BUTTON:
            if hit is not None:
                selected_segments.symmetric_difference_update({hit})
            else:
                selected_segments.clear()
        else:   # middle click, scroll-wheel buttons — not ours
            return
        render()

    canvas.mpl_connect("button_press_event", on_canvas_click)

    # ── Actions ────────────────────────────────────────────
    def save_annotations():
        try:
            with open(annotation_file, "w") as f:
                json.dump(db_data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save annotations:\n{e}",
                                 parent=top)
            return False
        try:
            refresh_counts()
        except Exception:
            pass
        return True

    def refresh_list_entry():
        result = current_result()
        if result is None:
            return
        listbox.delete(index)
        listbox.insert(index, _list_row(result))
        listbox.selection_set(index)

    # ── Triage: accept a suggestion, or pick another, then move on ──
    #
    # These three are the loop the window exists for.  All of them advance,
    # because deciding about a phrase and then having to reach for an arrow
    # key is what makes a list of 200 outliers feel long.

    def apply_label(new_label):
        """Relabel the whole phrase and step to the next outlier.

        Whole-phrase on purpose: the phrase is the unit that was scored, so
        the evidence is about all of it.  Use Relabel (r) to act on a
        selection instead.
        """
        result = current_result()
        if result is None or new_label is None:
            return
        if not relabel_phrase(db_data, result.phrase, new_label, None):
            messagebox.showerror(
                "Relabel",
                "Could not find these segments in the annotation file — "
                "they may have been edited elsewhere.", parent=top)
            return
        if not save_annotations():
            return
        # The suggestion has been acted on; keeping it would leave the list
        # row claiming a correction that has already happened.
        result.suggestion = None
        result.note = f"relabelled {new_label}"
        refresh_list_entry()
        advance()

    def advance():
        """Step to the next outlier, or stay put on the last one."""
        if index >= len(results) - 1:
            render()
            return
        select(index + 1)

    def do_accept():
        result = current_result()
        if result is None or result.suggestion is None:
            return
        apply_label(result.suggestion)

    def do_apply_selected():
        apply_label(selected_candidate_label())

    def do_dismiss():
        """Mark the phrase correct as it stands and move on."""
        result = current_result()
        if result is None:
            return
        result.suggestion = None
        result.note = "kept as is"
        refresh_list_entry()
        advance()

    def do_relabel():
        result = current_result()
        if result is None:
            return
        phrase = result.phrase
        targets = sorted(selected_segments) if selected_segments else None
        if targets is None:
            scope = f"all {phrase.n_segments} segment(s) of this phrase"
            current = phrase.label_text
        else:
            scope = f"the {len(targets)} selected segment(s)"
            current = ", ".join(
                sorted({str(phrase.segments[i][2]) for i in targets}))
        new_label = simpledialog.askinteger(
            "Relabel", f"New label for {scope} (currently {current}):",
            parent=top)
        if new_label is None:
            return
        changed = relabel_phrase(db_data, phrase, new_label, targets)
        if not changed:
            messagebox.showerror(
                "Relabel",
                "Could not find these segments in the annotation file — "
                "they may have been edited elsewhere.", parent=top)
            return
        if not save_annotations():
            return
        selected_segments.clear()
        refresh_list_entry()
        render()

    def do_relabel_zero():
        """Label the selection — or the whole phrase — 0, with no prompt.

        Scope matches Relabel: the selected segments if there are any,
        otherwise every segment of the phrase.
        """
        result = current_result()
        if result is None:
            return
        targets = sorted(selected_segments) if selected_segments else None
        changed = relabel_phrase(db_data, result.phrase, ZERO_LABEL, targets)
        if not changed:
            messagebox.showerror(
                "Set 0",
                "Could not find these segments in the annotation file — "
                "they may have been edited elsewhere.", parent=top)
            return
        if not save_annotations():
            return
        selected_segments.clear()
        refresh_list_entry()
        render()

    def do_split():
        result = current_result()
        if result is None:
            return
        if not split_marks:
            messagebox.showinfo(
                "Split Phrase",
                "Right-click inside a segment on the spectrogram to mark "
                "where to split it.", parent=top)
            return
        added = split_phrase(db_data, result.phrase, list(split_marks))
        if not added:
            messagebox.showerror(
                "Split Phrase",
                "Could not split — no marked point falls inside a segment of "
                "this phrase.", parent=top)
            return
        if not save_annotations():
            return
        split_marks.clear()
        selected_segments.clear()
        refresh_list_entry()
        render()

    def clear_marks():
        split_marks.clear()
        selected_segments.clear()
        render()

    actions = tk.Frame(top)
    actions.pack(side=tk.TOP, fill="x", padx=6, pady=6)
    tk.Button(actions, text="◀ Prev",
              command=lambda: select(index - 1)).pack(side=tk.LEFT, padx=3)
    tk.Button(actions, text="Next ▶",
              command=lambda: select(index + 1)).pack(side=tk.LEFT, padx=3)
    tk.Button(actions, text="Accept  (a)",
              command=do_accept).pack(side=tk.LEFT, padx=(12, 3))
    tk.Button(actions, text="Apply Selected  (Enter)",
              command=do_apply_selected).pack(side=tk.LEFT, padx=3)
    tk.Button(actions, text="Keep As Is  (d)",
              command=do_dismiss).pack(side=tk.LEFT, padx=3)
    tk.Button(actions, text="Relabel Selected/Phrase  (r)",
              command=do_relabel).pack(side=tk.LEFT, padx=12)
    tk.Button(actions, text="Set 0  (e)",
              command=do_relabel_zero).pack(side=tk.LEFT, padx=3)
    tk.Button(actions, text="Split at Marks",
              command=do_split).pack(side=tk.LEFT, padx=3)
    tk.Button(actions, text="Clear Marks & Selection",
              command=clear_marks).pack(side=tk.LEFT, padx=3)

    note.pack(side=tk.TOP, fill="x", padx=6, pady=(0, 2))
    status.pack(side=tk.TOP, fill="x", padx=6, pady=(0, 4))
    tk.Label(top, fg="gray40", anchor="w",
             text="Edits are saved to the annotation file immediately; a "
                  "spectrogram viewer open on the same song will not refresh "
                  "on its own.").pack(side=tk.TOP, fill="x", padx=6,
                                      pady=(0, 6))

    def shortcut(action):
        """Run *action* unless the keystroke was meant for a text field."""
        def handler(event):
            if isinstance(event.widget, (tk.Entry, tk.Spinbox, ttk.Entry)):
                return
            action()
        return handler

    top.bind("<Left>", shortcut(lambda: select(index - 1)))
    top.bind("<Right>", shortcut(lambda: select(index + 1)))
    top.bind("<r>", shortcut(do_relabel))
    top.bind("<R>", shortcut(do_relabel))
    top.bind("<e>", shortcut(do_relabel_zero))
    top.bind("<E>", shortcut(do_relabel_zero))
    top.bind("<a>", shortcut(do_accept))
    top.bind("<A>", shortcut(do_accept))
    top.bind("<d>", shortcut(do_dismiss))
    top.bind("<D>", shortcut(do_dismiss))
    top.bind("<Return>", shortcut(do_apply_selected))
    top.bind("<KP_Enter>", shortcut(do_apply_selected))
    # 1/2/3 jump straight to a candidate without reaching for the mouse; they
    # preview it, exactly as clicking the button does.
    for position in range(N_CANDIDATES_SHOWN):
        top.bind(str(position + 1),
                 shortcut(lambda p=position: (candidate_choice.set(p),
                                              on_candidate_click())))

    show_message("Choose a method and label, then press Run Detection.")
