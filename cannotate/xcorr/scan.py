"""Score a corpus of phrases against a saved template bank.

Nothing is cached.  A scan touches each phrase once, so caching its
spectrogram is pure cost — 3000 songs would be roughly 13 GB written and read
back for no reuse.  Each file is read, turned into a spectrogram, scored, and
dropped before the next is opened; peak memory is one file per worker
(~30 MB) whatever the corpus size.

The unit of work is a **file**, not a phrase, so a wav holding twenty phrases
is read once.  Workers are initialised with the bank's *path* rather than a
pickled bank, and jobs carry absolute wav paths resolved by the caller, so the
only thing crossing the process boundary is a handful of numbers per phrase.

This module deliberately imports no GUI: a spawned worker pays for numpy and
scipy, not tkinter and matplotlib.
"""
import os
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from multiprocessing import TimeoutError as mp_TimeoutError

import numpy as np

from . import scoring, spectro
from .bank import Bank


SCORED = "scored"
NO_TEMPLATE = "no_template"
# The bank has a template for this phrase's label, but no usable comparison
# could be made with it: either the phrase is shorter than that template, or it
# is shorter than every *other* label's, leaving nothing to compete.  Either
# way the margin is undefined, and a margin is the whole statistic — so this
# cannot be reported as a disagreement.
NO_SELF_SCORE = "no_self_score"
TOO_SHORT = "too_short"
FILE_ERROR = "file_error"

# Four workers, not eight.  This machine has 4 performance cores and 8 workers
# measured *slower* (7.1 vs 8.3 files/s) through contention.  See
# sandbox/xcorr/README.md.
MAX_WORKERS = 4

# One file per task, deliberately.  ``imap_unordered`` only returns an object
# with a timeout-capable ``next()`` when the chunksize is 1 — above that it
# wraps the iterator in a plain generator to unchunk, and a plain generator can
# only be waited on forever.  Chunking would buy nothing here in any case: a
# task is a whole file, ~150 ms of work, so the queue round trip is noise.
CHUNKSIZE = 1

# How often the scan looks up from the workers to answer the Cancel button.
# Short enough to feel responsive, long enough not to spin.
POLL_SECONDS = 0.5

_WORKER = {}


def default_workers():
    return max(1, min(MAX_WORKERS, (os.cpu_count() or 2) // 2))


@dataclass
class PhraseScore:
    """What a scan learned about one phrase.

    ``scores`` maps every label the phrase could be scored against to that
    label's best figures, so the caller can rank candidates without rescoring.
    ``predicted`` is ``None`` exactly when there is no suggestion to make.
    """
    spec: tuple
    outcome: str
    predicted: object = None
    margin: float = float("nan")
    scores: dict = field(default_factory=dict)

    @property
    def label(self):
        return self.spec[4]

    @property
    def disagrees(self):
        return self.predicted is not None and self.predicted != self.label

    def ranked_candidates(self, limit=None):
        """Labels best-peak first, as ``(label, peak)`` pairs."""
        ordered = sorted(self.scores.items(), key=lambda kv: -kv[1]["peak"])
        pairs = [(label, figures["peak"]) for label, figures in ordered]
        return pairs if limit is None else pairs[:limit]


# ── One file's worth of work ─────────────────────────────────────────

def _score_file(path, specs, bank, band, hop_seconds, nfft):
    """Score every phrase of one file.  Returns a list of ``PhraseScore``."""
    from cannotate.cannotate_utils import read_wav

    try:
        fs, y = read_wav(path)
        S, F, T = spectro.compute(np.asarray(y, dtype=np.float64), fs,
                                  nfft=nfft, hop_seconds=hop_seconds)
    except Exception:                                         # noqa: BLE001
        return [PhraseScore(spec=spec, outcome=FILE_ERROR) for spec in specs]
    del y
    S, _ = scoring.band_limit(S, F, band[0], band[1])

    known = set(bank.labels)
    out = []
    for spec in specs:
        cols = np.where((T >= spec[2]) & (T <= spec[3]))[0]
        if cols.size < 2:
            out.append(PhraseScore(spec=spec, outcome=TOO_SHORT))
            continue
        scores = scoring.score_with_bank(bank, S[:, cols])
        if not scores:
            out.append(PhraseScore(spec=spec, outcome=TOO_SHORT))
            continue
        if spec[4] not in known:
            # No template for this label, so there is nothing to agree or
            # disagree with.  The candidates are still worth showing — the
            # user picks from them — but naming one of them "the suggestion"
            # would be asserting more than was measured.
            out.append(PhraseScore(spec=spec, outcome=NO_TEMPLATE,
                                   scores=scores))
            continue
        predicted, margin = scoring.decide(scores, spec[4])
        if predicted is None or not np.isfinite(margin):
            # decide() returns a nan margin when the assigned label was not
            # among the labels that could be scored, or when it was the only
            # one.  Calling the surviving label the winner would assert a
            # comparison that never happened — which for a 110 ms phrase whose
            # own template is 200 ms means proposing a relabel on no evidence.
            out.append(PhraseScore(spec=spec, outcome=NO_SELF_SCORE,
                                   scores=scores))
            continue
        out.append(PhraseScore(spec=spec, outcome=SCORED,
                               predicted=int(predicted), margin=float(margin),
                               scores=scores))
    del S
    return out


def _init_worker(bank_path):
    loaded = Bank.load(bank_path)
    _WORKER.update(bank=loaded.template_bank(), band=loaded.band,
                   hop=loaded.hop_seconds, nfft=loaded.nfft)


def _scan_one(job):
    path, specs = job
    return _score_file(path, specs, _WORKER["bank"], _WORKER["band"],
                       _WORKER["hop"], _WORKER["nfft"])


# ── The scan ─────────────────────────────────────────────────────────

def _build_jobs(specs, wav_paths):
    """Group specs by file.  Specs with no known wav become FILE_ERROR."""
    by_file = defaultdict(list)
    for spec in specs:
        by_file[(spec[0], spec[1])].append(spec)

    jobs, orphans = [], []
    for key in sorted(by_file):
        path = wav_paths.get(key)
        if path is None:
            orphans.extend(PhraseScore(spec=spec, outcome=FILE_ERROR)
                           for spec in by_file[key])
        else:
            jobs.append((path, by_file[key]))
    return jobs, orphans


def scan(specs, wav_paths, bank_path, workers=None, progress=None):
    """Score *specs* against the bank saved at *bank_path*.

    *specs* are ``(date, file_name, t_start, t_end, label)`` tuples;
    *wav_paths* maps ``(date, file_name)`` to an absolute wav path.

    *progress* is called as ``progress(files_done, files_total)`` after each
    file and may return ``False`` to cancel, in which case the results
    gathered so far are returned rather than raising — an interrupted scan is
    still worth browsing.

    Returns a list of :class:`PhraseScore`, worst margin first, with
    unscoreable phrases last.
    """
    workers = default_workers() if workers is None else max(1, int(workers))
    jobs, results = _build_jobs(specs, wav_paths)
    total = len(jobs)

    if workers == 1 or total <= 1:
        # No pool: for a single file the fork/spawn cost exceeds the work, and
        # this path is the reference the pooled one is tested against.
        loaded = Bank.load(bank_path)
        scorer = loaded.template_bank()
        for done, (path, group) in enumerate(jobs, 1):
            results.extend(_score_file(path, group, scorer, loaded.band,
                                       loaded.hop_seconds, loaded.nfft))
            if progress is not None and progress(done, total) is False:
                break
    else:
        with Pool(workers, initializer=_init_worker,
                  initargs=(bank_path,)) as pool:
            iterator = pool.imap_unordered(_scan_one, jobs,
                                           chunksize=CHUNKSIZE)
            got, cancelled = _drain(iterator, total, progress)
            results.extend(got)
            if cancelled:
                pool.terminate()

    return sort_results(results)


def _drain(iterator, total, progress):
    """Collect from a pool iterator, polling so a stall stays interruptible.

    Iterating the pool directly would block in ``next()`` until a file
    finished, and *progress* is the only thing that reads the Cancel button.
    A worker that dies or gets stuck on one long file would therefore leave
    the window frozen with no way out.  Polling with a timeout means Cancel is
    answered within :data:`POLL_SECONDS` whatever the workers are doing.

    Returns ``(results, cancelled)``.
    """
    results = []
    done = 0
    while done < total:
        try:
            got = iterator.next(timeout=POLL_SECONDS)
        except mp_TimeoutError:
            if progress is not None and progress(done, total) is False:
                return results, True
            continue
        except StopIteration:
            break
        results.extend(got)
        done += 1
        if progress is not None and progress(done, total) is False:
            return results, True
    return results, False


def sort_results(results):
    """Worst first: negative margins, then near-ties, then everything else.

    ``nan`` margins — no template, too short — sort last rather than
    unpredictably, which is what a bare sort on ``nan`` would do.
    """
    def key(result):
        if result.outcome != SCORED or not np.isfinite(result.margin):
            return (1, 0.0)
        return (0, result.margin)

    return sorted(results, key=key)
