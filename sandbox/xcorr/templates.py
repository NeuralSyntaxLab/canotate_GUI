"""Stage 1 — choose one (or a few) templates per label.

Template *duration* is ``min(200 ms, mean_duration - std)`` per label.  On this
corpus the cap binds for every label in scope: even the shortest of them has
``mean - std`` above 0.36 s, so the rule is effectively "200 ms with a floor
for pathologically short labels".

Template *position* follows the consistency rule: draw 30 phrases, cut a
centred candidate window from each, and keep the candidate that correlates
best with the *other* phrases.  Candidates are scored against whole phrases
rather than against each other, because that is exactly the quantity used at
test time; and since the score is a maximum over lags, it does not matter
where in the rendition cycle each candidate window happened to land.

Run:  python templates.py [--n-templates 3]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import scoring
from cache import Patches


def template_cols(patches, label, indices):
    """Columns for ``min(TEMPLATE_MAX_SECONDS, mean - std)`` of this label."""
    durations = np.array([patches.duration(i) for i in indices])
    seconds = min(config.TEMPLATE_MAX_SECONDS,
                  float(durations.mean() - durations.std()))
    seconds = max(seconds, 0.030)          # never degenerate to nothing
    return patches.cols_for_seconds(seconds), seconds


def centred_window(patch, m):
    """Cut ``m`` columns from the centre of *patch*, or ``None`` if too short."""
    n = patch.shape[1]
    if n < m:
        return None
    start = (n - m) // 2
    return patch[:, start:start + m]


def select(patches, label, indices, n_candidates=None, n_keep=None,
           rng=None, verbose=True):
    """Return ``{"templates": [...], "band": (fmin, fmax), ...}`` for *label*.

    The frequency band is derived from the *pooled* candidate windows rather
    than from the eventual winner, which keeps band choice independent of
    template choice instead of circular.
    """
    n_candidates = n_candidates or config.N_CANDIDATES
    n_keep = n_keep or config.N_TEMPLATES_KEPT
    rng = rng or np.random.default_rng(config.RANDOM_SEED)

    m, seconds = template_cols(patches, label, indices)
    usable = [i for i in indices if patches[i].shape[1] >= m]
    if len(usable) < 2:
        raise ValueError(f"Label {label}: fewer than 2 phrases reach "
                         f"{seconds * 1000:.0f} ms.")

    chosen = rng.permutation(usable)[:n_candidates]
    candidates = [(i, centred_window(patches[i], m)) for i in chosen]
    candidates = [(i, w) for i, w in candidates if w is not None]

    if config.GLOBAL_BAND is not None:
        band = config.GLOBAL_BAND
    else:
        band = scoring.modulation_band([w for _, w in candidates],
                                       patches.F, config.BAND_PERCENTILES)

    banded = {i: scoring.band_limit(patches[i], patches.F, band[0], band[1])[0]
              for i, _ in candidates}
    banded_windows = {i: scoring.band_limit(w, patches.F, band[0], band[1])[0]
                      for i, w in candidates}

    # Score each candidate by its mean peak-ZNCC against every *other* phrase.
    consistency = []
    for i, window in banded_windows.items():
        peaks = []
        for j, phrase in banded.items():
            if j == i or phrase.shape[1] < window.shape[1]:
                continue
            surface = scoring.zncc_surface(phrase, window)
            peaks.append(surface.max())
        consistency.append((float(np.mean(peaks)) if peaks else -1.0, i))
    consistency.sort(reverse=True)

    keep = consistency[:n_keep]
    if verbose:
        scores = ", ".join(f"{s:.3f}" for s, _ in consistency[:5])
        print(f"[templates] label {label:>3}: {m} cols "
              f"({seconds * 1000:.0f} ms), band "
              f"{band[0]:.0f}-{band[1]:.0f} Hz, {len(candidates)} candidates, "
              f"top consistency {scores}")

    return {
        "templates": [banded_windows[i] for _, i in keep],
        "band": band,
        "cols": m,
        "seconds": seconds,
        "source_index": [int(i) for _, i in keep],
        "consistency": [float(s) for s, _ in keep],
        "candidate_indices": [int(i) for i, _ in candidates],
    }


def build(template_indices_by_label, patches=None, n_keep=None, verbose=True):
    """Build a template spec for every label from the given index pools."""
    patches = patches or Patches()
    rng = np.random.default_rng(config.RANDOM_SEED)
    out = {}
    for label in config.LABELS:
        indices = template_indices_by_label.get(label, [])
        if len(indices) < 2:
            if verbose:
                print(f"[templates] label {label}: skipped "
                      f"({len(indices)} phrase(s) in the template split)")
            continue
        out[label] = select(patches, label, indices, n_keep=n_keep, rng=rng,
                            verbose=verbose)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-templates", type=int, default=None)
    args = ap.parse_args()
    p = Patches()
    pools = {l: list(p.indices_for(l)) for l in config.LABELS}
    specs = build(pools, patches=p, n_keep=args.n_templates)
    print(f"\nbuilt templates for {len(specs)} label(s) "
          f"(all data, no split — use experiment.py for the real thing)")
