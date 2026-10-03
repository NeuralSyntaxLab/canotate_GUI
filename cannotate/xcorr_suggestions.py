"""Suggesting a syllable label by cross-correlating against the template bank.

Replaces the DTW path.  A selection is scored against every label's template
with the same statistic the outlier detector uses — peak ZNCC over time lags —
and the three best labels are offered.

Two details are worth knowing before reading the code.

**The viewer's spectrogram cannot be used.**  ``S_norm`` is percentile-clipped
to [0, 1] at a 1 ms hop over the full band; the bank lives on raw ``log1p``
power at the bank's own hop, nfft and band.  So a song gets a second
spectrogram, computed at the *bank's* parameters — which also means a bank
fitted months ago still scores correctly against today's audio.

**A selection can be shorter than a template.**  ZNCC needs the query to be at
least as long as what slides across it, and a 200 ms template does not fit
inside a 60 ms selection.  Those templates are scored with the roles reversed:
the query slides across the template, and the best lag picks the trim of the
template that the query actually matches.  It is the only way such a selection
is scorable at all, and it is permissive — a long generic template can host a
short query almost anywhere inside itself — which is why every suggestion
carries the fraction of the template the query covered.
"""
from dataclasses import dataclass

import numpy as np

from .xcorr import scoring, spectro


TOP_K = 3


@dataclass
class Suggestion:
    """One label the selection might be, with the evidence for it."""

    label: int
    #: Peak ZNCC, in [-1, 1].  Higher is better; 1.0 is an exact match.
    score: float
    #: Fraction of the template the query covered, in (0, 1].  Below 1 means
    #: the query was shorter than the template and only part of it was scored.
    coverage: float
    #: Column of the best alignment, in whichever array did the sliding.
    lag: int
    template: np.ndarray

    @property
    def trimmed(self):
        return self.coverage < 1.0


class SuggestionEngine:
    """Ranks selections against a fitted :class:`cannotate.xcorr.bank.Bank`."""

    def __init__(self, bank):
        self.bank = bank
        self._scorer = bank.template_bank()

    @property
    def labels(self):
        return self.bank.labels

    # ── the query's spectrogram ──────────────────────────────────────

    def song_spectrogram(self, y, fs):
        """``(S, T)`` for a whole song, band-limited to the bank's band.

        Computed once per song by the viewer and reused for every selection:
        one of these costs about 150 ms, and re-deriving it per suggestion
        would make the popup feel slow for no reason.
        """
        S, F, T = spectro.compute(np.asarray(y, dtype=np.float64), fs,
                                  nfft=self.bank.nfft,
                                  hop_seconds=self.bank.hop_seconds)
        S, _F = scoring.band_limit(S, F, self.bank.band[0], self.bank.band[1])
        return S, T

    @staticmethod
    def query_patch(S, T, t_start, t_end):
        """The columns of *S* inside ``[t_start, t_end]``, or ``None``.

        Fewer than two columns is not a query: ZNCC over a single frame
        measures nothing.
        """
        if S is None or T is None:
            return None
        cols = np.where((T >= t_start) & (T <= t_end))[0]
        if cols.size < 2:
            return None
        return S[:, cols]

    # ── ranking ──────────────────────────────────────────────────────

    def rank(self, query):
        """Every label scored against *query*, best first."""
        if query is None or query.shape[1] < 2:
            return []

        results = []
        too_long = {}
        for label, surfaces in self._scorer.surfaces(query).items():
            best = None
            for k, surface in enumerate(surfaces):
                template = self.bank.templates[label][k]
                if surface is None:              # template longer than query
                    too_long.setdefault(label, []).append((k, template))
                    continue
                lag = int(np.argmax(surface))
                cand = Suggestion(label=label, score=float(surface[lag]),
                                  coverage=1.0, lag=lag, template=template)
                if best is None or cand.score > best.score:
                    best = cand
            if best is not None:
                results.append(best)

        if too_long:
            results.extend(self._rank_reversed(query, too_long, results))

        results.sort(key=lambda s: s.score, reverse=True)
        return results

    def _rank_reversed(self, query, too_long, already):
        """Score templates longer than the query by sliding the query in them.

        The query becomes the thing that slides, so it is transformed once and
        reused across every oversized template — the same trick
        :class:`~cannotate.xcorr.scoring.TemplateBank` plays in the ordinary
        direction.
        """
        scored = {s.label for s in already}
        slider = scoring.TemplateBank({0: [query]})
        n = query.shape[1]
        out = []
        for label, templates in too_long.items():
            best = None
            for _k, template in templates:
                surface = slider.surfaces(template)[0][0]
                if surface is None:
                    continue
                lag = int(np.argmax(surface))
                cand = Suggestion(label=label, score=float(surface[lag]),
                                  coverage=n / template.shape[1], lag=lag,
                                  template=template)
                if best is None or cand.score > best.score:
                    best = cand
            # A label whose shorter template already scored keeps that score:
            # a full-coverage match is the more honest number of the two.
            if best is not None and label not in scored:
                out.append(best)
        return out

    def suggest(self, S, T, t_start, t_end, top_k=TOP_K):
        """The *top_k* labels for the selection ``[t_start, t_end]``."""
        return self.rank(self.query_patch(S, T, t_start, t_end))[:top_k]
