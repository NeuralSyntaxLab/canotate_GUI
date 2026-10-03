"""The template bank: fitting it, saving it, loading it.

One template per label — a short window cut from the most *typical* rendition
of that label — is the only thing this path persists.  It is about 1 MB, and
everything else (spectrogram patches, score vectors) is recomputed on demand.

The sandbox keeps a patch cache because it re-scores the same phrases dozens
of times while parameters change.  Fitting touches each candidate exactly
once, so there is nothing to reuse and no cache here: audio is read, windowed,
ranked, and dropped.

Audio arrives through an injected ``load_song(date, file_name) -> (fs, y)``
callable rather than a path convention, because the two callers already know
where their wavs are and disagree about it: the GUI has ``date_to_wavs``, the
CLI has an ``--audio-root``.
"""
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import scoring, spectro


# One band for every label.  The sandbox measured per-label bands at 95.6% and
# a single shared band at 95.7% — no loss — and sharing one band is what lets a
# phrase's Fourier transform be computed once and reused across all labels, an
# 11x speedup.  See sandbox/xcorr/README.md.
DEFAULT_BAND = (1000.0, 13000.0)

# The template is the centred window of a phrase, capped here.  Longer windows
# reject fewer phrases but reach past the stereotyped part of a syllable.
TEMPLATE_MAX_SECONDS = 0.200
MIN_TEMPLATE_SECONDS = 0.030

# Phrases drawn per label when ranking candidates.  Fitting reads only these,
# so it touches a small fraction of a corpus.
N_CANDIDATES = 30

# A label needs at least this many usable phrases to get a template: ranking a
# candidate means scoring it against the others, which needs at least one other.
MIN_PHRASES_PER_LABEL = 2

RANDOM_SEED = 0


# ── The bank ─────────────────────────────────────────────────────────

@dataclass
class Source:
    """Where a template was cut from — the window, not the whole phrase.

    Keeping it lets the exact patch be re-read from the wav: the class and
    audio previews the GUI writes to ``syllable_templates/`` are produced from
    this, and a bank whose templates cannot be traced back to audio is a bank
    nobody can check.
    """
    date: str
    file_name: str
    t_start: float
    t_end: float


@dataclass
class Bank:
    """Band-limited templates, plus the parameters they were cut with.

    ``templates`` maps a label to a list of ``(n_freq, m)`` arrays, best
    first.  The spectrogram parameters travel with the templates because a
    bank fitted at one hop is meaningless against a query computed at another.

    ``sources`` maps a label to a list of :class:`Source` positionally aligned
    with ``templates[label]``.  It is allowed to be empty for a label: banks
    written before provenance existed load without it, and everything that
    scores works the same without it.
    """
    templates: dict
    band: tuple
    hop_seconds: float = spectro.HOP_SECONDS
    nfft: int = spectro.NFFT
    sources: dict = field(default_factory=dict)

    @property
    def labels(self):
        return sorted(self.templates)

    def source_for(self, label, index=0):
        """The :class:`Source` of one template, or ``None`` if untracked."""
        stack = self.sources.get(label) or []
        return stack[index] if index < len(stack) else None

    def merged_with(self, other):
        """A new bank holding this bank's labels plus *other*'s.

        *other* wins where both hold a label.  Raises if the two were cut with
        different spectrogram parameters or bands: their templates live on
        different frequency rows, so mixing them would score nonsense.
        """
        mine = (self.band, self.hop_seconds, self.nfft)
        theirs = (other.band, other.hop_seconds, other.nfft)
        if mine != theirs:
            raise ValueError(
                f"Cannot merge banks cut with different parameters: "
                f"{mine} vs {theirs}.")
        templates = dict(self.templates)
        templates.update(other.templates)
        sources = dict(self.sources)
        sources.update(other.sources)
        return Bank(templates=templates, band=self.band,
                    hop_seconds=self.hop_seconds, nfft=self.nfft,
                    sources=sources)

    def without_labels(self, labels):
        """A new bank with *labels* dropped."""
        drop = set(labels)
        return Bank(
            templates={l: t for l, t in self.templates.items() if l not in drop},
            band=self.band, hop_seconds=self.hop_seconds, nfft=self.nfft,
            sources={l: s for l, s in self.sources.items() if l not in drop})

    def template_bank(self):
        """A :class:`scoring.TemplateBank` ready to score phrases."""
        return scoring.TemplateBank(self.templates)

    def save(self, path):
        """Write ``templates.npz``.

        Templates differ in width, so they are stored concatenated along time
        with a column count per template — a ragged array would force
        ``allow_pickle``.
        """
        index, arrays, sources = [], [], []
        for label in self.labels:
            stack = self.templates[label]
            found = self.sources.get(label) or []
            for i, template in enumerate(stack):
                index.append((label, template.shape[1]))
                arrays.append(np.asarray(template, dtype=np.float32))
                src = found[i] if i < len(found) else None
                sources.append(src or Source("", "", float("nan"),
                                             float("nan")))
        if not arrays:
            raise ValueError("Refusing to save a bank with no templates.")
        np.savez(path,
                 data=np.concatenate(arrays, axis=1),
                 cols=np.array([c for _, c in index], dtype=np.int64),
                 label=np.array([l for l, _ in index], dtype=np.int64),
                 band=np.array(self.band, dtype=np.float64),
                 hop_seconds=np.array(self.hop_seconds, dtype=np.float64),
                 nfft=np.array(self.nfft, dtype=np.int64),
                 src_date=np.array([s.date for s in sources]),
                 src_file=np.array([s.file_name for s in sources]),
                 src_t0=np.array([s.t_start for s in sources], dtype=np.float64),
                 src_t1=np.array([s.t_end for s in sources], dtype=np.float64))
        return path

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            data, cols, label = z["data"], z["cols"], z["label"]
            band = tuple(float(x) for x in z["band"])
            hop_seconds = float(z["hop_seconds"])
            nfft = int(z["nfft"])
            # Banks written before provenance existed have none of these.
            # They are still perfectly good banks, so this is a default, not
            # an error.
            has_src = "src_date" in z.files
            if has_src:
                src = (z["src_date"], z["src_file"], z["src_t0"], z["src_t1"])
        templates = defaultdict(list)
        sources = defaultdict(list)
        start = 0
        for i, (l, c) in enumerate(zip(label, cols)):
            templates[int(l)].append(data[:, start:start + c].astype(np.float64))
            if has_src:
                sources[int(l)].append(Source(str(src[0][i]), str(src[1][i]),
                                              float(src[2][i]),
                                              float(src[3][i])))
            start += c
        return cls(templates=dict(templates), band=band,
                   hop_seconds=hop_seconds, nfft=nfft,
                   sources=dict(sources))


# ── Reading phrases as spectrogram patches ───────────────────────────

def patches_for(specs, load_song, band, hop_seconds=spectro.HOP_SECONDS,
                nfft=spectro.NFFT, progress=None):
    """Yield ``(spec, patch, times)`` for each phrase spec, one wav at a time.

    *times* holds the frame-centre time of every column of *patch*, so a
    caller that cuts a window out of the patch can say where in the recording
    that window is — which is what a template's provenance records.

    *specs* are ``(date, file_name, t_start, t_end, label)`` tuples.  They are
    grouped by file so each wav is read once however many phrases it holds,
    and each file's spectrogram is dropped before the next is opened — peak
    memory is one file, whatever the corpus size.

    A phrase covering fewer than two spectrogram frames, or living in a file
    that cannot be read, is skipped silently; callers that need to report
    those use :mod:`cannotate.xcorr.scan`.

    *progress*, if given, is called as ``progress(files_done, files_total)``
    and may return ``False`` to stop early.
    """
    by_file = defaultdict(list)
    for spec in specs:
        by_file[(spec[0], spec[1])].append(spec)

    total = len(by_file)
    for done, ((date, file_name), group) in enumerate(sorted(by_file.items()), 1):
        try:
            fs, y = load_song(date, file_name)
        except Exception:                                     # noqa: BLE001
            fs = None
        if fs is not None and y is not None:
            S, F, T = spectro.compute(np.asarray(y, dtype=np.float64), fs,
                                      nfft=nfft, hop_seconds=hop_seconds)
            del y
            S, _ = scoring.band_limit(S, F, band[0], band[1])
            for spec in group:
                cols = np.where((T >= spec[2]) & (T <= spec[3]))[0]
                if cols.size >= 2:
                    yield spec, S[:, cols], T[cols]
            del S
        if progress is not None and progress(done, total) is False:
            return


# ── Fitting ──────────────────────────────────────────────────────────

def _template_cols(durations, hop_seconds):
    """Columns for ``min(TEMPLATE_MAX_SECONDS, mean - std)`` of a label.

    Subtracting a standard deviation keeps the window inside all but the
    shortest renditions, so few phrases are rejected for being too short.
    """
    durations = np.asarray(durations, dtype=float)
    seconds = min(TEMPLATE_MAX_SECONDS,
                  float(durations.mean() - durations.std()))
    seconds = max(seconds, MIN_TEMPLATE_SECONDS)
    return max(1, int(round(seconds / hop_seconds))), seconds


def _centred_window(patch, m):
    """Cut *m* columns from the centre of *patch*.

    Returns ``(window, start_column)``, or ``(None, 0)`` when the patch is too
    short.  The offset is returned because a template that cannot say where in
    its phrase it came from cannot be turned back into audio.
    """
    n = patch.shape[1]
    if n < m:
        return None, 0
    start = (n - m) // 2
    return patch[:, start:start + m], start


def _source_of(found, start, m, hop_seconds):
    """The :class:`Source` of the window at column *start* of a fitted patch.

    Times come from the patch's own column times rather than from the phrase's
    annotated bounds plus an offset: the first column of a patch sits at the
    first frame centre at or after ``t_start``, which is up to one hop later.
    """
    spec, _patch, times = found
    t0 = float(times[start])
    t1 = float(times[min(start + m - 1, len(times) - 1)])
    return Source(date=str(spec[0]), file_name=str(spec[1]),
                  t_start=t0, t_end=t1 + hop_seconds)


def _rank_candidates(windows, patches):
    """Order candidate windows by mean peak ZNCC against the other phrases.

    This picks the most *typical* rendition — a medoid under the same
    statistic used at scoring time, rather than an average, which would blur
    the time-frequency structure that ZNCC keys on.
    """
    ranked = []
    for i, window in windows.items():
        peaks = []
        for j, phrase in patches.items():
            if j == i or phrase.shape[1] < window.shape[1]:
                continue
            peaks.append(float(scoring.zncc_surface(phrase, window).max()))
        ranked.append((float(np.mean(peaks)) if peaks else -1.0, i))
    ranked.sort(reverse=True)
    return ranked


def fit(specs, load_song, labels=None, band=DEFAULT_BAND,
        n_candidates=N_CANDIDATES, n_keep=1, hop_seconds=spectro.HOP_SECONDS,
        nfft=spectro.NFFT, seed=RANDOM_SEED, progress=None, verbose=False):
    """Build a :class:`Bank` from verified phrase annotations.

    *specs* are ``(date, file_name, t_start, t_end, label)`` tuples and
    *load_song* resolves audio; see :func:`patches_for`.  At most
    *n_candidates* phrases per label are read, so fitting is cheap relative to
    a scan even on a large corpus.

    Labels with fewer than :data:`MIN_PHRASES_PER_LABEL` usable phrases are
    skipped — a template ranked against nothing is not a template.  Raises
    :class:`ValueError` if that leaves no labels at all.
    """
    rng = np.random.default_rng(seed)

    wanted = None if labels is None else set(labels)
    pools = defaultdict(list)
    for spec in specs:
        if wanted is None or spec[4] in wanted:
            pools[spec[4]].append(spec)

    chosen = []
    for label in sorted(pools):
        group = pools[label]
        if len(group) < MIN_PHRASES_PER_LABEL:
            continue
        picks = rng.permutation(len(group))[:n_candidates]
        chosen.extend(group[i] for i in picks)

    by_label = defaultdict(list)
    for spec, patch, times in patches_for(chosen, load_song, band,
                                          hop_seconds=hop_seconds, nfft=nfft,
                                          progress=progress):
        by_label[spec[4]].append((spec, patch, times))

    templates, sources = {}, {}
    for label in sorted(by_label):
        found = by_label[label]
        patches = [p for _spec, p, _t in found]
        if len(patches) < MIN_PHRASES_PER_LABEL:
            continue
        m, seconds = _template_cols([p.shape[1] * hop_seconds for p in patches],
                                    hop_seconds)
        usable = {i: p for i, p in enumerate(patches) if p.shape[1] >= m}
        if len(usable) < MIN_PHRASES_PER_LABEL:
            continue
        cut = {i: _centred_window(p, m) for i, p in usable.items()}
        windows = {i: w for i, (w, _start) in cut.items()}
        ranked = _rank_candidates(windows, usable)
        templates[label] = [windows[i] for _, i in ranked[:n_keep]]
        sources[label] = [_source_of(found[i], cut[i][1], m, hop_seconds)
                          for _, i in ranked[:n_keep]]
        if verbose:
            top = ", ".join(f"{s:.3f}" for s, _ in ranked[:5])
            print(f"[fit] label {label:>3}: {m} cols ({seconds * 1000:.0f} ms), "
                  f"{len(usable)} candidates, top consistency {top}")

    if not templates:
        raise ValueError(
            f"Fitting produced no templates: no label has "
            f"{MIN_PHRASES_PER_LABEL} phrases long enough to cut a "
            f"{MIN_TEMPLATE_SECONDS * 1000:.0f} ms window from.")

    return Bank(templates=templates, band=tuple(float(b) for b in band),
                hop_seconds=hop_seconds, nfft=nfft, sources=sources)


def describe(path):
    """One line per label, for the CLI and for the GUI's bank summary."""
    loaded = Bank.load(path)
    size = os.path.getsize(path) / 1e6
    lines = [f"{os.path.basename(path)} — {len(loaded.labels)} label(s), "
             f"band {loaded.band[0]:.0f}-{loaded.band[1]:.0f} Hz, "
             f"hop {loaded.hop_seconds * 1000:.0f} ms, {size:.2f} MB"]
    for label in loaded.labels:
        cols = ", ".join(str(t.shape[1]) for t in loaded.templates[label])
        lines.append(f"  label {label:>3}: {len(loaded.templates[label])} "
                     f"template(s), {cols} columns")
    return "\n".join(lines)
