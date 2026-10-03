"""Raw spectrogram for the scoring path.

Deliberately *not* the viewer's ``_compute_spectrogram_norm``.  That function
clips at the 5th/96th percentile of the whole recording before normalising —
a non-linear, recording-dependent transform.  In a loud file it crushes quiet
syllables toward zero before any comparison happens, and no amount of
z-scoring downstream can invert a clip.  ZNCC removes gain and offset by
construction, so this path needs no normalisation at all.
"""
import numpy as np
from scipy.signal import spectrogram, windows


# Same DPSS parameters as the rest of the codebase.  The hop is 2 ms, not the
# viewer's 1 ms: the analysis window is 512 samples = 11.6 ms at 44.1 kHz, so
# consecutive 1 ms frames overlap by 91% and carry almost no independent
# information.
NFFT = 512
HOP_SECONDS = 0.002
DPSS_NW = 4
DPSS_KMAX = 2

_TAPERS = {}


def _tapers(nfft, nw, kmax):
    key = (nfft, nw, kmax)
    if key not in _TAPERS:
        _TAPERS[key] = windows.dpss(nfft, nw, Kmax=kmax)
    return _TAPERS[key]


def compute(y, fs, nfft=NFFT, hop_seconds=HOP_SECONDS):
    """Two-taper log-power spectrogram.  Returns ``(S, F, T)``.

    The result stops at ``log1p(power)`` with no clipping or rescaling.
    """
    hop = max(1, int(round(hop_seconds * fs)))
    tapers = _tapers(nfft, DPSS_NW, DPSS_KMAX)

    F, T, S1 = spectrogram(y, fs, window=tapers[0], noverlap=nfft - hop,
                           nfft=nfft, nperseg=nfft, mode="complex")
    _, _, S2 = spectrogram(y, fs, window=tapers[1], noverlap=nfft - hop,
                           nfft=nfft, nperseg=nfft, mode="complex")
    power = np.abs(S1) ** 2
    del S1
    power += np.abs(S2) ** 2
    del S2
    return np.log1p(power, out=power), F, T
