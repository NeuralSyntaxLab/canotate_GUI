"""The per-bird template bank: where it lives, when it grows, what feeds it.

One bank per bird, at the dataset root::

    <bird_folder>/<bird_name>_templates_YYYYMMDD.npz

The newest date wins.  A bank is **never refit automatically**: startup fits
only the labels the bank does not already hold, so the first launch after a
new syllable class appears pays for that one class and every later launch
reads the npz alone.  :func:`ensure` with ``force=True`` is the deliberate
rebuild, and the only thing that ever drops a template.

This module is the layer between the GUI's view of a dataset (``db_data``,
``date_to_wavs``, a bird folder) and :mod:`cannotate.xcorr.bank`, which knows
nothing about either and takes plain specs and a loader.  Keeping the two
apart is what lets a scan worker import the scoring path without dragging in
annotation conventions.
"""
import datetime as _dt
import os
import re

from .cannotate_utils import read_wav
from .xcorr import bank as xbank


# A label needs this many *segments* in the fitting pool before it earns a
# template.  Segments, not phrases: a segment is one rendition of the syllable,
# which is both what the user selects when asking for a suggestion and the
# granularity a 200 ms template is cut at.
MIN_SEGMENTS = 10

# Never templated.  999 is the unlabeled placeholder; 0 is the reject marker
# the error-correction "e" shortcut writes.  Both are heterogeneous by
# construction, and a template cut from either would match anything unusual.
EXCLUDED_LABELS = (0, 999)

FILENAME_RE = re.compile(r"^(?P<bird>.+)_templates_(?P<date>\d{8})\.npz$")


# ── Where the bank lives ─────────────────────────────────────────────

def bank_filename(bird_name, date=None):
    day = (date or _dt.date.today()).strftime("%Y%m%d")
    return f"{bird_name}_templates_{day}.npz"


def bank_path(bird_folder, bird_name=None, date=None):
    """The path a bank written *today* (or on *date*) would take."""
    name = bird_name or os.path.basename(os.path.normpath(bird_folder))
    return os.path.join(bird_folder, bank_filename(name, date))


def newest_bank_path(bird_folder):
    """The most recent ``*_templates_YYYYMMDD.npz`` in *bird_folder*.

    Ties on the date fall back to mtime, so two birds' banks sharing a folder
    — or a file restored from a backup — still resolve to one answer rather
    than to whichever `os.listdir` happened to return first.
    """
    if not bird_folder or not os.path.isdir(bird_folder):
        return None
    found = []
    for name in os.listdir(bird_folder):
        m = FILENAME_RE.match(name)
        if not m:
            continue
        full = os.path.join(bird_folder, name)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            mtime = 0.0
        found.append((m.group("date"), mtime, full))
    if not found:
        return None
    return max(found)[2]


def load_bank(bird_folder):
    """The newest bank in *bird_folder*, or ``None``.

    An unreadable bank returns ``None`` rather than raising: a corrupt file
    must not stop the program from opening, and the rebuild that follows fixes
    it.
    """
    path = newest_bank_path(bird_folder)
    if not path:
        return None
    try:
        return xbank.Bank.load(path)
    except Exception as e:                                    # noqa: BLE001
        print(f"Ignoring unreadable template bank {path}: "
              f"{type(e).__name__}: {e}")
        return None


# ── What feeds it ────────────────────────────────────────────────────

def song_loader(date_to_wavs):
    """A ``load_song(date, file_name) -> (fs, y)`` over a date->paths map."""
    lookup = {(date, os.path.basename(p)): p
              for date, paths in (date_to_wavs or {}).items() for p in paths}

    def load_song(date, file_name):
        path = lookup.get((date, file_name))
        if path is None:
            raise FileNotFoundError(f"No wav for {date}/{file_name}")
        fs, y = read_wav(path)
        if getattr(y, "ndim", 1) == 2:
            y = y.mean(axis=1)
        return fs, y

    return load_song


def _segments_of(entry):
    """The ``(t0, t1, label)`` triples of *entry* a template may be cut from.

    A 4-element segment carries the AI-generated flag and does not unpack as a
    triple, so machine guesses are excluded here exactly as they were from the
    DTW templates — a template cut from a model's mistake teaches the model's
    mistake to the user.
    """
    out = []
    for seg in entry.get("segments") or []:
        try:
            t0, t1, label = seg
        except Exception:                                     # noqa: BLE001
            continue
        if label is None:
            continue
        try:
            t0, t1, label = float(t0), float(t1), int(label)
        except (TypeError, ValueError):
            continue
        if label in EXCLUDED_LABELS or t1 <= t0:
            continue
        out.append((t0, t1, label))
    return out


def fit_pool(db_data, date_to_wavs, min_segments=MIN_SEGMENTS):
    """The specs a fit may draw from, and how many each label offers.

    Returns ``(specs, counts)`` where *specs* are
    ``(date, file_name, t_start, t_end, label)`` and *counts* maps a label to
    its number of segments **in the pool actually chosen for it**.

    Verified annotations are preferred per label rather than globally: the
    cross-correlation numbers were measured on human-verified data, but a
    corpus nobody has verified yet must still get templates.  So a label with
    at least *min_segments* verified segments is fitted from those alone, and
    one without falls back to everything it has.
    """
    known = {(date, os.path.basename(p))
             for date, paths in (date_to_wavs or {}).items() for p in paths}

    verified, unverified = {}, {}
    for date, entries in (db_data or {}).items():
        for entry in entries or []:
            file_name = entry.get("file_name")
            if (date, file_name) not in known:
                continue          # no audio to cut a template from
            bucket = verified if entry.get("human_verified") else unverified
            for t0, t1, label in _segments_of(entry):
                bucket.setdefault(label, []).append(
                    (date, file_name, t0, t1, label))

    specs, counts = [], {}
    for label in sorted(set(verified) | set(unverified)):
        good = verified.get(label, [])
        chosen = good if len(good) >= min_segments else (
            good + unverified.get(label, []))
        specs.extend(chosen)
        counts[label] = len(chosen)
    return specs, counts


def eligible_labels(counts, min_segments=MIN_SEGMENTS):
    return {label for label, n in counts.items() if n >= min_segments}


def as_display_templates(bank, date_to_wavs):
    """The bank as the dict shape the viewer's template screens already read.

    ``{label: {"spec", "avg_duration", "source_path", "source_start",
    "source_end"}}`` — the same keys the DTW medoid templates had, so the
    class/audio previews, the on-disk PNG/WAV regeneration and the suggestion
    popup keep working against the bank without knowing what a bank is.

    ``spec`` is the band-limited template patch itself, so what is displayed is
    exactly what is scored.
    """
    lookup = {(date, os.path.basename(p)): p
              for date, paths in (date_to_wavs or {}).items() for p in paths}
    out = {}
    for label in (bank.labels if bank else []):
        stack = bank.templates[label]
        if not stack:
            continue
        src = bank.source_for(label)
        entry = {"spec": stack[0], "avg_duration": 0.0, "source_path": None,
                 "source_start": None, "source_end": None}
        if src is not None and src.file_name:
            entry["source_path"] = lookup.get((src.date, src.file_name))
            entry["source_start"] = src.t_start
            entry["source_end"] = src.t_end
            entry["avg_duration"] = float(src.t_end - src.t_start)
        out[label] = entry
    return out


# ── Keeping it up to date ────────────────────────────────────────────

class EnsureResult:
    """What :func:`ensure` did, so a caller can say so without guessing."""

    def __init__(self, bank, path, added=(), removed=(), pending=(),
                 written=False):
        self.bank = bank
        self.path = path
        self.added = sorted(added)
        self.removed = sorted(removed)
        #: Labels that exist but have not reached ``min_segments`` yet.
        self.pending = sorted(pending)
        self.written = written

    def summary(self):
        if self.bank is None:
            return "No templates yet — no label has enough annotated segments."
        parts = [f"{len(self.bank.labels)} template(s): {self.bank.labels}"]
        if self.added:
            parts.append(f"added {self.added}")
        if self.removed:
            parts.append(f"dropped {self.removed}")
        if self.pending:
            parts.append(f"not enough segments yet for {self.pending}")
        return "\n".join(parts)


def ensure(bird_folder, db_data, date_to_wavs, force=False,
           min_segments=MIN_SEGMENTS, load_song=None, progress=None,
           date=None, verbose=True):
    """Make sure every eligible label has a template, and return the bank.

    Ordinarily this fits **only labels the existing bank lacks** and leaves
    the rest exactly as they were.  When nothing is missing, no audio is read
    and no file is written.

    With *force* it refits every eligible label from current data and drops
    templates whose label is no longer eligible — a merged-away class stops
    being suggested, and a bad early template can be replaced.

    *progress* is passed through to the fit as ``progress(done, total)`` over
    wav files and may return ``False`` to stop; a cancelled fit writes nothing.
    """
    existing = load_bank(bird_folder)
    specs, counts = fit_pool(db_data, date_to_wavs, min_segments)
    eligible = eligible_labels(counts, min_segments)
    pending = set(counts) - eligible
    have = set(existing.templates) if existing else set()

    wanted = eligible if force else (eligible - have)
    removed = (have - eligible) if force else set()

    if not wanted:
        bank = existing.without_labels(removed) if removed else existing
        if removed and bank is not None and bank.templates:
            path = _write(bank, bird_folder, date, verbose)
            return EnsureResult(bank, path, removed=removed, pending=pending,
                                written=True)
        return EnsureResult(bank, newest_bank_path(bird_folder),
                            pending=pending)

    if load_song is None:
        load_song = song_loader(date_to_wavs)

    cancelled = []

    def watch(done, total):
        if progress is None:
            return None
        if progress(done, total) is False:
            cancelled.append(True)
            return False
        return None

    try:
        fitted = xbank.fit([s for s in specs if s[4] in wanted], load_song,
                           labels=wanted, progress=watch, verbose=verbose)
    except ValueError as e:
        # No label produced a template — every candidate was too short or
        # unreadable.  That is not a failure the user can act on, and the
        # existing bank is still good.
        if verbose:
            print(f"[templates] Nothing fitted: {e}")
        return EnsureResult(existing, newest_bank_path(bird_folder),
                            pending=pending)

    if cancelled:
        if verbose:
            print("[templates] Fit cancelled; the bank on disk is unchanged.")
        return EnsureResult(existing, newest_bank_path(bird_folder),
                            pending=pending)

    if force or existing is None:
        bank = fitted
    else:
        bank = existing.merged_with(fitted)
    path = _write(bank, bird_folder, date, verbose)
    return EnsureResult(bank, path, added=set(fitted.templates),
                        removed=removed, pending=pending, written=True)


def _write(bank, bird_folder, date, verbose):
    """Save *bank* under today's name.

    A second write on the same day replaces that day's file: history is kept
    per day, not per write, and a folder that accumulated a file per template
    addition would be worse than no history at all.
    """
    path = bank_path(bird_folder, date=date)
    bank.save(path)
    if verbose:
        print(f"[templates] Wrote {len(bank.labels)} template(s) to {path}")
    return path
