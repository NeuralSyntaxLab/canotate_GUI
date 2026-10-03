"""
Utility functions and constants shared across the Cannotate GUI modules.
"""
import os
import warnings

import numpy as np
from scipy.signal import spectrogram as sp_spectrogram, ellip, filtfilt


# ── NN model configuration ──────────────────────────────────────────
# The model zoo needs torch. Annotating by hand does not, so the import is
# optional: without torch the GUI still opens and every manual feature works,
# and only the NN-assisted buttons are greyed out. NN_IMPORT_ERROR carries the
# reason so the UI can explain itself rather than failing silently.
try:
    from bird_song_detector.model_zoo import ConvMLP, ClassificationCnn  # noqa: F401
except ImportError as _nn_err:
    ConvMLP = ClassificationCnn = None
    NN_AVAILABLE = False
    NN_IMPORT_ERROR = str(_nn_err)
else:
    NN_AVAILABLE = True
    NN_IMPORT_ERROR = None

NN_MODEL_ARCHITECTURES = {
    "ConvMLP (input_size=256)": (ConvMLP, {"input_size": 256}),
} if NN_AVAILABLE else {}

# Model weights ship inside the project so it stays self-contained. Point
# CANNOTATE_MODELS_DIR elsewhere to use a shared weights folder instead.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NN_WEIGHTS_DIR = os.path.normpath(
    os.environ.get("CANNOTATE_MODELS_DIR", os.path.join(_PROJECT_ROOT, "models")))

# Default ONNX segmentation model (ConvMLP exported from bird_song_detector).
ONNX_DEFAULT_WEIGHTS = os.path.join(
    NN_WEIGHTS_DIR, "weights_best_epoch4_vacc91.onnx")


# ── Robust WAV reading ──────────────────────────────────────────────
def _spectral_flatness(signal):
    """Wiener entropy: geometric_mean(PSD) / arithmetic_mean(PSD).

    Higher value → flatter spectrum → more noise-like.
    Lower value  → tonal / harmonic structure → more song-like.
    """
    n = min(len(signal), 8192)
    psd = np.abs(np.fft.rfft(signal[:n].astype(np.float64))) ** 2 + 1e-10
    return float(np.exp(np.mean(np.log(psd))) / np.mean(psd))


def _identify_channels(data):
    """Return ``(song_idx, noise_idx)`` for a 2-D stereo array.

    Uses spectral flatness: the channel with the flattest spectrum is white
    noise; the most tonal channel is the song microphone.
    """
    flatness = [_spectral_flatness(data[:, i]) for i in range(data.shape[1])]
    noise_idx = int(np.argmax(flatness))
    song_idx = int(np.argmin(flatness))
    return song_idx, noise_idx


def _load_raw(path):
    """Return ``(fs, data)`` without any channel selection."""
    from scipy.io import wavfile
    try:
        fs, data = wavfile.read(path)
        return fs, data
    except Exception as scipy_err:
        try:
            import soundfile as sf
        except ImportError:
            raise scipy_err
        try:
            data, fs = sf.read(path, dtype='float32', always_2d=False)
            return fs, data
        except Exception:
            raise scipy_err


def read_wav(path):
    """Read an audio file, returning the **song** channel as a 1-D array.

    For stereo files the channel with the lowest spectral flatness (most
    tonal, least noise-like) is selected.  Use :func:`read_wav_channels`
    when you also need the white-noise / trigger channel.
    """
    fs, data = _load_raw(path)
    if data.ndim == 2:
        song_idx, _ = _identify_channels(data)
        data = data[:, song_idx]
    return fs, data


def read_wav_channels(path):
    """Read a stereo WAV and return ``(fs, song_channel, noise_channel)``.

    Channel assignment is based on spectral flatness: the flattest channel is
    treated as white noise / trigger; the most tonal channel is the song mic.
    For mono files ``noise_channel`` is ``None``.
    """
    fs, data = _load_raw(path)
    if data.ndim == 1:
        return fs, data, None
    song_idx, noise_idx = _identify_channels(data)
    return fs, data[:, song_idx].astype(np.float64), data[:, noise_idx].astype(np.float64)


# ── Pretty sonogram ─────────────────────────────────────────────────
def generate_pretty_sonogram(signal, fs=None, **kwargs):
    """
    based on zftftb_pretty_sonogram.m
    Compute a 2-taper spectrogram with Gaussian windows (and its derivative).
    """
    if fs is None:
        fs = 48_000.0

    overlap_ms = kwargs.get('overlap', 7)
    len_ms     = kwargs.get('len', 70)
    tscale_ms  = kwargs.get('tscale', 2)
    postproc   = kwargs.get('postproc', 'y')
    nfft_param = kwargs.get('nfft', None)
    zeropad_arg = kwargs.get('zeropad', None)
    norm_amp   = bool(kwargs.get('norm_amp', 0))
    filtering  = kwargs.get('filtering', None)
    clipping   = kwargs.get('clipping', [-2, 2])
    saturation = kwargs.get('saturation', 0.8)
    units      = kwargs.get('units', 'ln')
    _          = kwargs.get('low', None), kwargs.get('high', None)

    x = np.asarray(signal, dtype=np.float64).flatten()
    win_len   = int(round((len_ms / 1e3) * fs))
    noverlap  = int(round((overlap_ms / 1e3) * fs))

    if x.size <= win_len:
        warnings.warn("Signal shorter than window; truncating.")
        diff   = win_len - noverlap
        win_len = x.size // 3
        noverlap = win_len - diff
        nfft_param = None

    def _nextpow2(n):
        return int(np.ceil(np.log2(n))) if n > 0 else 0

    nfft = 2 ** _nextpow2(win_len) if nfft_param is None else 2 ** _nextpow2(int(nfft_param))

    if zeropad_arg == 0:
        zp, autopad = int(round(win_len / 2)), True
    elif zeropad_arg is not None:
        zp, autopad = int(zeropad_arg), False
    else:
        zp, autopad = None, False
    if zp is not None:
        x = np.concatenate((np.zeros(zp), x, np.zeros(zp)))

    if norm_amp:
        x /= np.max(np.abs(x))
    if filtering is not None:
        b, a = ellip(5, 0.2, 40, filtering / (fs / 2), btype='highpass')
        x = filtfilt(b, a, x)

    half  = win_len / 2.0
    tvec  = np.arange(win_len) - (half - 1)
    sigma = (tscale_ms / 1e3) * fs
    w     = np.exp(-(tvec / sigma) ** 2)
    dw    = -2 * w * (tvec / (sigma ** 2))

    f, t, S1 = sp_spectrogram(x, fs=fs, window=w, nperseg=win_len,
                               noverlap=noverlap, nfft=nfft, mode='complex')
    _, _, S2  = sp_spectrogram(x, fs=fs, window=dw, nperseg=win_len,
                               noverlap=noverlap, nfft=nfft, mode='complex')

    M = (np.abs(S1) ** 2 + np.abs(S2) ** 2)
    if units.lower() == 'ln':
        img = np.log(M)
    elif units.lower() == 'db':
        img = 20 * np.log10(M)
    else:
        img = M

    if autopad:
        t = t - zp / fs

    return img, f, t


# ── Annotation segment helpers ───────────────────────────────────────
def clean_segments(segments):
    """Convert segment tuples to JSON-safe native types.

    Accepts 3-tuples (t_start, t_end, label) or 4-tuples where the 4th
    element is the AI-generated flag (True).  The flag is preserved so that
    the GUI can distinguish AI annotations from manual ones.

    Segments come back sorted by start time — edits (splits, new draws) can
    append pieces out of chronological order, and downstream code that walks
    the list assuming ascending time (e.g. gap-based merging) would silently
    misbehave on an unsorted list.
    """
    result = []
    for seg in segments:
        t_start = float(seg[0]) if seg[0] is not None else 0.0
        t_end   = float(seg[1]) if seg[1] is not None else 0.0
        label   = int(seg[2])   if seg[2] is not None else 999
        if len(seg) > 3 and seg[3]:
            result.append([t_start, t_end, label, True])
        else:
            result.append([t_start, t_end, label])
    result.sort(key=lambda s: s[0])
    return result


def binary_to_segments(seg_vector, annot_vector, fs):
    """Recover segments from the sample-level binary vector format.

    Lossy, and never to be used to read an annotation list that already
    exists: segments that touch (end of one == start of the next) share one
    uninterrupted run of samples, so they come back as a single segment
    carrying the first one's label.
    """
    seg_vector = np.array(seg_vector)
    annot_vector = np.array(annot_vector)
    if seg_vector.ndim != annot_vector.ndim:
        raise ValueError("seg_vector and annot_vector must have the same number of dimensions")
    changes = np.diff(np.concatenate(([0], seg_vector, [0])))
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    segments = []
    for s, e in zip(starts, ends):
        t_start = s / fs
        t_end   = e / fs
        sub = annot_vector[s:e]
        annot = int(sub[np.nonzero(sub)[0][0]]) if np.any(sub) else 999
        segments.append((t_start, t_end, annot))
    return segments


def segments_to_binary(segments, total_samples, fs):
    seg_vec = np.zeros(total_samples, dtype=int)
    annot_vec = np.zeros(total_samples, dtype=int)
    for seg in segments:
        t_s, t_e, lab = seg[0], seg[1], seg[2]
        i_s = int(round(t_s * fs))
        i_e = int(round(t_e * fs))
        seg_vec[i_s:i_e] = 1
        if lab is not None:
            annot_vec[i_s:i_e] = lab
    return seg_vec, annot_vec
