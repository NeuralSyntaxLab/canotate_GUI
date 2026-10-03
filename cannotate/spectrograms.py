"""
The viewer's spectrogram: computing it, and computing only part of it.

This is the *display* spectrogram — two DPSS tapers, log power, then
percentile-clipped onto [0, 1] over the whole recording.  It is deliberately
not what :mod:`cannotate.xcorr.spectro` produces for scoring: that path needs
an untouched log-power surface, because a percentile clip is a non-linear,
recording-dependent transform that no downstream normalisation can invert.

Template building and syllable suggestions used to live here, on a DTW
distance over spectrograms resampled to a fixed column count.  They now live
in :mod:`cannotate.template_store` and :mod:`cannotate.xcorr_suggestions`, on
the measured cross-correlation path.  What is left is the spectrogram
machinery the viewer itself draws with, plus the annotation walk that decides
which instances are worth reading audio for.
"""
import numpy as np
from scipy.signal import spectrogram, windows


# ── Low-level helpers ────────────────────────────────────────────────

def extract_syllable_spectrogram(S_norm, T, t_start, t_end):
    """Extract normalised spectrogram columns for [t_start, t_end]."""
    if S_norm is None or T is None:
        return None
    indices = np.where((T >= t_start) & (T <= t_end))[0]
    if indices.size == 0:
        return None
    return S_norm[:, indices]


def _dpss_windows(nfft):
    """The two DPSS tapers every spectrogram in this module is built from."""
    E = windows.dpss(nfft, 4, Kmax=2)
    return E[0], E[1]


def _fine_spectrogram_frames(y, fs, nfft, w1, w2, hop):
    """Raw (un-normalised) log-power spectrogram of *y* at the given hop.

    Returns ``(S_log, F)``.  Split out so that the full-file path, the cropped
    path and the statistics pass all go through exactly one implementation --
    and so tests can count how much audio the cropped path actually processes.
    """
    noverlap = nfft - hop
    F, _T, S1 = spectrogram(y, fs, window=w1, noverlap=noverlap,
                            nfft=nfft, nperseg=nfft)
    _F, _T, S2 = spectrogram(y, fs, window=w2, noverlap=noverlap,
                             nfft=nfft, nperseg=nfft)
    # Free each intermediate as soon as it is consumed and scale in place:
    # for a long recording every one of these arrays is hundreds of MB, so
    # holding them simultaneously is what drives the process out of memory.
    S_power = (np.abs(S1) + np.abs(S2)) / 2
    del S1, S2
    S_log = np.log10(1 + S_power + 1e-12)
    del S_power
    return S_log, F


def _stats_from_log(S_log):
    """The four constants that map a raw log-power array onto [0, 1].

    Mirrors the in-place arithmetic of :func:`compute_spectrogram_for_audio`
    exactly, so applying them with :func:`_apply_normalisation` reproduces it.
    """
    eps = 1e-8
    minv = float(S_log.min())
    maxv = float(S_log.max())
    scaled = (S_log - minv) / (maxv - minv + eps)
    vmin = float(np.percentile(scaled, 5)) + eps
    vmax = float(np.percentile(scaled, 97)) + eps
    return minv, maxv, vmin, vmax


def _apply_normalisation(S_log, stats):
    """Map raw log-power onto [0, 1] with pre-computed *stats*, in place."""
    minv, maxv, vmin, vmax = stats
    eps = 1e-8
    S_log -= minv
    S_log /= (maxv - minv + eps)
    S_log -= vmin
    S_log /= (vmax - vmin) if vmax > vmin else 1.0
    return np.clip(S_log, 0, 1, out=S_log)


def frame_times(n_samples, fs, nfft, hop):
    """The frame centre times scipy's ``spectrogram`` will report.

    Replicates scipy's own expression rather than an equivalent one, so that
    selecting columns by ``t0 <= T <= t1`` picks precisely the same frames here
    as it does from a full-file spectrogram -- the cropped path must not drift
    by a frame at a boundary.
    """
    return np.arange(nfft / 2, n_samples - nfft / 2 + 1, hop) / float(fs)


def compute_spectrogram_for_audio(y, fs, nfft=512):
    """Compute normalised spectrogram (same pipeline as *load_existing_data*).

    Returns (S_norm, T, F).
    """
    hop = int(0.001 * fs)
    w1, w2 = _dpss_windows(nfft)
    S_log, F = _fine_spectrogram_frames(y, fs, nfft, w1, w2, hop)
    S_norm = _apply_normalisation(S_log, _stats_from_log(S_log))
    return S_norm, frame_times(len(y), fs, nfft, hop), F


# How much coarser the statistics pass is than the fine pass.  The four
# normalisation constants are distribution statistics of the whole recording,
# so they survive decimation: measured over real songs, a 4x coarser hop costs
# a quarter of the time and shifts the resulting [0, 1] mapping by under 1% --
# far less than the run-to-run spread already introduced by the random choice
# of template instances.
STATS_HOP_MULT = 4

# Wanted ranges closer together than this many samples are spectrogrammed in
# one pass: below roughly a window length the two passes would overlap anyway,
# and one scipy call on a slightly wider slice beats two on narrow ones.
def _merge_gap_samples(nfft):
    return 4 * nfft

# Above this fraction of the recording, cropping cannot pay for itself.  The
# full-file path is then used instead -- which also makes the patches *exactly*
# what the old code produced, with no estimated statistics involved.
DENSE_COVERAGE_FRACTION = 0.6


def compute_spectrogram_for_ranges(y, fs, ranges, nfft=512):
    """Normalised spectrogram columns for *ranges* only, without doing the rest.

    *ranges* is a list of ``(t_start, t_end)`` in seconds.  Returns a list of
    ``(S_norm_patch, T_patch)`` in the same order, where each patch holds
    exactly the columns a full-file spectrogram would place inside that range.

    This exists because building templates needs a few seconds out of each
    recording, while :func:`compute_spectrogram_for_audio` spectrogrammed all
    of it -- on a real corpus ~88% wasted work, and the bulk of the startup
    rebuild.  The fine spectrogram is computed only over the wanted spans; the
    normalisation constants come from a cheaper coarse-hop pass over the whole
    recording (see :data:`STATS_HOP_MULT`), because they are whole-file
    statistics and a crop cannot know them.

    When the wanted spans cover most of the recording
    (:data:`DENSE_COVERAGE_FRACTION`) there is nothing to save, so the ordinary
    full-file path is used and the patches are bit-identical to the old ones.
    """
    if not ranges:
        return []

    hop = int(0.001 * fs)
    n = len(y)
    T_all = frame_times(n, fs, nfft, hop)
    n_frames = T_all.size

    # Which full-file frame indices each requested range owns.  Computed from
    # the frame-time grid alone -- no audio touched yet.
    spans = []
    for t0, t1 in ranges:
        if n_frames == 0:
            spans.append(None)
            continue
        idx = np.where((T_all >= t0) & (T_all <= t1))[0]
        spans.append((int(idx[0]), int(idx[-1])) if idx.size else None)

    covered = sum(hi - lo + 1 for s in spans if s is not None
                  for lo, hi in (s,))
    if n_frames == 0:
        empty = np.zeros((nfft // 2 + 1, 0))
        return [(empty.copy(), T_all[:0].copy()) for _ in ranges]

    if covered >= DENSE_COVERAGE_FRACTION * n_frames:
        S_norm, T, _F = compute_spectrogram_for_audio(y, fs, nfft=nfft)
        out = []
        for span in spans:
            if span is None:
                out.append((S_norm[:, :0].copy(), T[:0].copy()))
            else:
                lo, hi = span
                out.append((S_norm[:, lo:hi + 1].copy(), T[lo:hi + 1].copy()))
        return out

    w1, w2 = _dpss_windows(nfft)

    # Statistics pass: whole recording, coarse hop.
    coarse_hop = hop * STATS_HOP_MULT
    if n >= nfft + coarse_hop:
        S_coarse, _F = _fine_spectrogram_frames(y, fs, nfft, w1, w2, coarse_hop)
        stats = _stats_from_log(S_coarse)
        del S_coarse
    else:
        S_log, _F = _fine_spectrogram_frames(y, fs, nfft, w1, w2, hop)
        stats = _stats_from_log(S_log)
        del S_log

    # Group the wanted spans into as few contiguous passes as possible.
    order = sorted((i for i, s in enumerate(spans) if s is not None),
                   key=lambda i: spans[i][0])
    gap = _merge_gap_samples(nfft)
    chunks = []  # [(frame_lo, frame_hi, [range index, ...]), ...]
    for i in order:
        lo, hi = spans[i]
        if chunks and lo * hop - chunks[-1][1] * hop <= gap:
            c_lo, c_hi, members = chunks[-1]
            chunks[-1] = (c_lo, max(c_hi, hi), members + [i])
        else:
            chunks.append((lo, hi, [i]))

    out = [None] * len(ranges)
    for i, span in enumerate(spans):
        if span is None:
            out[i] = (np.zeros((nfft // 2 + 1, 0)), T_all[:0].copy())

    for c_lo, c_hi, members in chunks:
        # Frame j of the full file reads samples [j*hop, j*hop + nfft), so a
        # slice starting at c_lo*hop puts full-file frame c_lo at local frame
        # 0 -- identical samples through an identical window, hence identical
        # values.  Any other start would shift the grid.
        s0 = c_lo * hop
        s1 = min(n, c_hi * hop + nfft)
        S_log, _F = _fine_spectrogram_frames(y[s0:s1], fs, nfft, w1, w2, hop)
        S_chunk = _apply_normalisation(S_log, stats)
        for i in members:
            lo, hi = spans[i]
            a, b = lo - c_lo, hi - c_lo + 1
            out[i] = (S_chunk[:, a:b].copy(), T_all[lo:hi + 1].copy())
        del S_chunk
    return out
