"""Cross-correlation scoring core — pure functions, no I/O, no plotting.

This is the module that graduates into ``cannotate/`` if the experiment
succeeds, so it deliberately depends on nothing but numpy and scipy.

The central object is the **ZNCC surface**: the zero-mean normalised
cross-correlation of a short template against a longer phrase, evaluated at
*every* time lag.  Two properties follow from computing it over lags rather
than on a pre-aligned pair:

* the score no longer depends on where the phrase onset happened to fall
  relative to the template (which a fixed tiling would make it depend on), and
* a peak-to-sidelobe ratio becomes definable at all, since PSR is a property
  of a surface and is meaningless for a single aligned score.
"""
import numpy as np
from scipy import fft
from scipy.signal import fftconvolve


# ── Frequency band ───────────────────────────────────────────────────

def band_limit(S, F, fmin, fmax):
    """Restrict a spectrogram to the ``[fmin, fmax]`` frequency band.

    Returns ``(S_band, F_band)``.  ``None`` bounds are left open.
    """
    mask = np.ones(F.shape, dtype=bool)
    if fmin is not None:
        mask &= F >= fmin
    if fmax is not None:
        mask &= F <= fmax
    if not mask.any():
        raise ValueError(f"Band [{fmin}, {fmax}] Hz selects no frequency bins.")
    return S[mask, :], F[mask]


def modulation_band(patches, F, percentiles=(5.0, 95.0)):
    """Derive a frequency band from where a label's energy actually *modulates*.

    For each patch the per-row standard deviation across time measures how much
    that frequency varies within the syllable.  Pooling those profiles and
    taking a cumulative-mass interval picks the band carrying the syllable's
    structure, rather than the band carrying the most absolute energy — which
    for a broadband noise floor would be the low frequencies regardless of the
    bird.

    Returns ``(fmin, fmax)`` in Hz.
    """
    profile = np.zeros(len(F), dtype=np.float64)
    used = 0
    for patch in patches:
        if patch.shape[1] < 2:
            continue
        profile += patch.std(axis=1)
        used += 1
    if used == 0 or profile.sum() <= 0:
        return float(F[0]), float(F[-1])
    cumulative = np.cumsum(profile) / profile.sum()
    lo_pct, hi_pct = percentiles[0] / 100.0, percentiles[1] / 100.0
    lo = int(np.searchsorted(cumulative, lo_pct))
    hi = int(np.searchsorted(cumulative, hi_pct))
    lo = min(lo, len(F) - 1)
    hi = min(max(hi, lo + 1), len(F) - 1)
    return float(F[lo]), float(F[hi])


# ── The ZNCC surface ─────────────────────────────────────────────────

def zncc_surface(phrase, template):
    """ZNCC of *template* against *phrase* at every time lag.

    ``phrase`` is ``(n_freq, n)`` and ``template`` is ``(n_freq, m)`` with
    ``n >= m``; both must already be band-limited to the same rows.  The result
    has length ``n - m + 1``, entry ``k`` being the correlation of the template
    with ``phrase[:, k:k+m]``.  Values lie in ``[-1, 1]``.

    Exact, and O(n log n) rather than the O(n*m*n_freq) of the direct form.
    Two algebraic facts make it cheap:

    * Centring the *template* is enough for the numerator.  The window-mean
      term of the query drops out because ``sum(template - mean) == 0``, so
      only the query's running sums are needed, not a running re-centring.
    * A 2-D 'valid' convolution against the doubly-flipped template is exactly
      the per-row correlation summed over frequency, because the kernel spans
      every frequency row — the frequency axis of the output collapses to one.
    """
    phrase = np.asarray(phrase, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    if phrase.shape[0] != template.shape[0]:
        raise ValueError("Phrase and template must share a frequency axis "
                         f"({phrase.shape[0]} vs {template.shape[0]} bins).")
    n_freq, m = template.shape
    n = phrase.shape[1]
    if n < m:
        raise ValueError(f"Phrase ({n} cols) is shorter than the template "
                         f"({m} cols).")

    centred = template - template.mean()
    template_norm = np.sqrt((centred ** 2).sum())
    if template_norm <= 0:
        return np.zeros(n - m + 1)

    # Numerator: sum over frequency of the per-row valid correlation.
    numerator = fftconvolve(phrase, centred[::-1, ::-1], mode="valid")[0]

    # Denominator: the query window's standard deviation, from running sums.
    col_sum = phrase.sum(axis=0)
    col_sq = (phrase ** 2).sum(axis=0)
    cs = np.concatenate(([0.0], np.cumsum(col_sum)))
    cq = np.concatenate(([0.0], np.cumsum(col_sq)))
    window_sum = cs[m:] - cs[:-m]
    window_sq = cq[m:] - cq[:-m]
    count = n_freq * m
    variance = window_sq - window_sum ** 2 / count
    np.maximum(variance, 0.0, out=variance)
    denominator = np.sqrt(variance) * template_norm

    surface = np.zeros_like(numerator)
    nonzero = denominator > 0
    surface[nonzero] = numerator[nonzero] / denominator[nonzero]
    return surface


# ── Many templates against one phrase (the production path) ──────────

class TemplateBank:
    """Templates pre-transformed for repeated scoring against many phrases.

    Scoring one phrase against every label recomputes the *phrase's* transform
    once instead of once per label, and never recomputes the templates' at all.
    That is the whole reason this class exists rather than a loop over
    :func:`zncc_surface`.

    It also drops the 2-D convolution used by :func:`zncc_surface`, which pads
    the frequency axis to ``2*n_freq - 1`` rows and then keeps a single one.
    Correlating along time only and summing over frequency in the frequency
    domain computes exactly the same numbers without that waste.

    All templates must share a frequency band (so they share the phrase's
    transform).  They may differ in length; each length gets its own denominator.

    Transform lengths are **bucketed to powers of two**.  Keying the cache on
    the exact length needed by each phrase looks tighter but is catastrophic:
    real phrases all differ in length, so the key never hits and every template
    is re-transformed for every phrase.  With a few hundred templates that
    allocates hundreds of megabytes per phrase and will take a machine down.
    Bucketing turns "once per phrase" into "once per bucket", and there are
    only a handful of buckets.
    """

    #: Refuse to build a bank whose cached spectra would exceed this, rather
    #: than exhausting memory.  Callers with many templates should chunk them.
    MAX_SPECTRA_BYTES = 256 * 1024 ** 2
    MAX_CACHED_BUCKETS = 3

    def __init__(self, templates_by_label):
        self.labels = list(templates_by_label)
        self.templates = templates_by_label
        self.n_freq = None
        for label, stack in templates_by_label.items():
            for template in stack:
                if self.n_freq is None:
                    self.n_freq = template.shape[0]
                elif template.shape[0] != self.n_freq:
                    raise ValueError(
                        "TemplateBank requires one shared frequency band "
                        f"({self.n_freq} vs {template.shape[0]} bins).")
        self._centred = {
            label: [t - t.mean() for t in stack]
            for label, stack in templates_by_label.items()}
        self._norm = {
            label: [float(np.sqrt((t ** 2).sum())) for t in stack]
            for label, stack in self._centred.items()}
        self.max_cols = max(t.shape[1] for stack in self._centred.values()
                            for t in stack)
        self.n_templates = sum(len(stack) for stack in self._centred.values())
        self._spectra = {}
        self._bucket_order = []

    @staticmethod
    def _bucket(needed):
        """Round an FFT length up to the next power of two.

        Powers of two are both fast to transform and few in number, which is
        the point: it is the *number of distinct lengths* that decides how
        often the template cache is rebuilt.
        """
        return 1 << max(4, int(needed - 1).bit_length())

    def estimated_bytes(self, length):
        """Bytes one bucket of cached template spectra would occupy."""
        return self.n_templates * self.n_freq * (length // 2 + 1) * 8

    def _prepare(self, length):
        """Transform every template at FFT length *length*, once per bucket."""
        if length in self._spectra:
            return
        budget = self.estimated_bytes(length)
        if budget > self.MAX_SPECTRA_BYTES:
            raise MemoryError(
                f"TemplateBank with {self.n_templates} templates would need "
                f"{budget / 1e6:.0f} MB of spectra at FFT length {length} "
                f"(limit {self.MAX_SPECTRA_BYTES / 1e6:.0f} MB). "
                "Score the templates in chunks.")
        # complex64 is ample: the inputs are float32 spectrogram values and the
        # result is divided by a norm of similar magnitude.
        self._spectra[length] = {
            label: [np.conj(fft.rfft(t, n=length, axis=1)).astype(np.complex64)
                    for t in stack]
            for label, stack in self._centred.items()}
        self._bucket_order.append(length)
        while len(self._bucket_order) > self.MAX_CACHED_BUCKETS:
            del self._spectra[self._bucket_order.pop(0)]

    def surfaces(self, phrase):
        """``{label: [surface per template]}`` for one phrase.

        The phrase is transformed once here and reused for every label, which
        is where the speed comes from.
        """
        phrase = np.asarray(phrase, dtype=np.float64)
        if phrase.shape[0] != self.n_freq:
            raise ValueError(f"Phrase has {phrase.shape[0]} frequency bins, "
                             f"templates have {self.n_freq}.")
        n = phrase.shape[1]
        length = self._bucket(n + self.max_cols - 1)
        self._prepare(length)
        spectra = self._spectra[length]
        phrase_spectrum = fft.rfft(phrase, n=length, axis=1).astype(np.complex64)

        # Running sums for the query-window standard deviation, shared across
        # every template of a given length.
        cs = np.concatenate(([0.0], np.cumsum(phrase.sum(axis=0))))
        cq = np.concatenate(([0.0], np.cumsum((phrase ** 2).sum(axis=0))))
        denominators = {}

        out = {}
        for label in self.labels:
            surfaces = []
            for k, spectrum in enumerate(spectra[label]):
                m = self._centred[label][k].shape[1]
                if n < m:
                    surfaces.append(None)
                    continue
                if m not in denominators:
                    window_sum = cs[m:] - cs[:-m]
                    window_sq = cq[m:] - cq[:-m]
                    variance = window_sq - window_sum ** 2 / (self.n_freq * m)
                    np.maximum(variance, 0.0, out=variance)
                    denominators[m] = np.sqrt(variance)
                numerator = fft.irfft(
                    (phrase_spectrum * spectrum).sum(axis=0).astype(np.complex128),
                    n=length)[:n - m + 1]
                den = denominators[m] * self._norm[label][k]
                surface = np.zeros_like(numerator)
                nonzero = den > 0
                surface[nonzero] = numerator[nonzero] / den[nonzero]
                surfaces.append(surface)
            out[label] = surfaces
        return out


def zncc_surface_reference(phrase, template):
    """Direct O(n*m*n_freq) ZNCC, for testing :func:`zncc_surface`.

    Never use this in anger — it exists so the fast path has something
    independent to be checked against.
    """
    phrase = np.asarray(phrase, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    m = template.shape[1]
    n = phrase.shape[1]
    t = template - template.mean()
    t_norm = np.sqrt((t ** 2).sum())
    out = np.zeros(n - m + 1)
    for k in range(n - m + 1):
        w = phrase[:, k:k + m]
        w = w - w.mean()
        w_norm = np.sqrt((w ** 2).sum())
        if w_norm > 0 and t_norm > 0:
            out[k] = float((w * t).sum() / (w_norm * t_norm))
    return out


# ── Peak and peak-to-sidelobe ratio ──────────────────────────────────

MIN_SIDELOBE_SAMPLES = 8


def peak_psr(surface, template_cols, exclusion=None):
    """Return ``(peak, peak_lag, psr)`` for a ZNCC surface.

    PSR measures *distinctiveness*, not similarity: a flat or noisy patch can
    reach a respectable peak ZNCC by accident, but only a genuine match
    produces a peak that stands clear of its surroundings.  It is the peak's
    height above the sidelobe mean, in sidelobe standard deviations, where the
    sidelobe is everything outside an exclusion zone around the peak.

    ``psr`` is ``nan`` when the surface is too short to estimate a sidelobe —
    a reported outcome, not a silent zero.
    """
    surface = np.asarray(surface, dtype=np.float64)
    if surface.size == 0:
        return float("nan"), -1, float("nan")
    peak_lag = int(np.argmax(surface))
    peak = float(surface[peak_lag])

    if exclusion is None:
        exclusion = int(template_cols)
    lo = max(0, peak_lag - exclusion)
    hi = min(surface.size, peak_lag + exclusion + 1)
    mask = np.ones(surface.size, dtype=bool)
    mask[lo:hi] = False
    sidelobe = surface[mask]

    if sidelobe.size < MIN_SIDELOBE_SAMPLES:
        return peak, peak_lag, float("nan")
    spread = sidelobe.std()
    if spread <= 0:
        return peak, peak_lag, float("nan")
    return peak, peak_lag, float((peak - sidelobe.mean()) / spread)


# ── Scoring one phrase against every label ───────────────────────────

INSUFFICIENT_SUPPORT = "insufficient_support"
TOO_SHORT = "too_short"
SCORED = "scored"


def score_phrase(patch, F, templates_by_label, min_support_ratio=3.0):
    """Score one phrase patch against every label's templates.

    ``templates_by_label`` maps a label to ``{"templates": [array, ...],
    "band": (fmin, fmax)}``.  Each label's band is applied to both that
    label's template and the query, so every comparison sees the frequencies
    that matter for the label it is testing.

    Returns ``(scores, outcome)`` where ``scores`` maps a label to
    ``{"peak", "psr", "lag", "template_rank"}`` — the winning template's
    figures — and ``outcome`` is one of the module constants.
    """
    scores = {}
    outcome = SCORED
    for label, spec in templates_by_label.items():
        band = spec["band"]
        query, _ = band_limit(patch, F, band[0], band[1])
        best = None
        for rank, template in enumerate(spec["templates"]):
            if query.shape[1] < template.shape[1]:
                continue
            surface = zncc_surface(query, template)
            if surface.size < min_support_ratio * template.shape[1]:
                outcome = INSUFFICIENT_SUPPORT
            peak, lag, psr = peak_psr(surface, template.shape[1])
            if best is None or peak > best["peak"]:
                best = {"peak": peak, "psr": psr, "lag": lag,
                        "template_rank": rank}
        if best is None:
            return {}, TOO_SHORT
        scores[label] = best
    return scores, outcome


# Breaking near-ties by PSR: see the sandbox README.  Measured on 17 labels it
# cuts the 3->26 confusion from 26 cases to 11 but raises 18->1 from 14 to 18,
# for a net +0.9 points that is unstable across the threshold.  That is within
# noise, so it stays off by default.
PSR_TIE_BREAK_DELTA = 0.06


def decide(scores, assigned_label, tie_break_psr=False,
           delta=PSR_TIE_BREAK_DELTA):
    """Turn a score vector into ``(predicted, margin)``.

    The margin — the assigned label's peak minus the best competitor's — is
    the production statistic.  Because all labels are scored against the same
    patch, recording gain, noise floor and bird condition are common-mode
    across the comparisons and cancel in the difference; no absolute
    calibration is needed.
    """
    if not scores:
        return None, float("nan")
    predicted = max(scores, key=lambda l: scores[l]["peak"])

    if tie_break_psr:
        # Peak ZNCC rewards the *least specific* template: one holding only a
        # syllable's sustained part matches anything containing that part, and
        # matches it at many lags.  PSR sees through that — a broad match has a
        # peak that does not stand out — but PSR's own scale depends so heavily
        # on template type that ranking by it alone is useless (0.4% accuracy).
        # So it is consulted only among near-tied peaks.
        best = scores[predicted]["peak"]
        tied = [l for l in scores if best - scores[l]["peak"] <= delta]
        finite = [l for l in tied if np.isfinite(scores[l]["psr"])]
        if len(tied) > 1 and finite:
            predicted = max(finite, key=lambda l: scores[l]["psr"])
    if assigned_label not in scores:
        return predicted, float("nan")
    mine = scores[assigned_label]["peak"]
    others = [s["peak"] for l, s in scores.items() if l != assigned_label]
    if not others:
        return predicted, float("nan")
    return predicted, float(mine - max(others))
