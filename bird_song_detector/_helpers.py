"""Two helpers lifted out of the full bird_song_detector package.

``InferenceToAnnotationData`` is the only thing this project needs from
bird_song_detector, and it reaches for exactly two functions that live in
files dedicated to *training*:

    load_from_wav        - bird_song_detector/data_loader.py   (2848 lines)
    normalize_per_segment - bird_song_detector/training_loop.py (1094 lines)

Vendoring those two files whole would drag h5py, pandas, scikit-learn,
seaborn and tensorboard into a tool that only ever annotates. Both functions
are short and self-contained, so they are reproduced here verbatim instead.

See PROVENANCE.md for the upstream commit these were taken from.
"""
import numpy as np
import torch
from scipy.io import wavfile


def load_from_wav(file_path, segment_size=256):
    """
    Loads a WAV file and returns segments of audio data following the pkl format.
    returns: segments_array: a tensor of shape (num_segments, segment_size) containing the audio segments
             labels: a tensor of shape (num_segments,) containing the labels (999 as placeholder)
             indices: a tensor of shape (num_segments,) containing the original indices of the segments in the audio file
             metadata: a dictionary containing metadata about the segments (segment_size and stride)
    """
    fs, data = wavfile.read(file_path)
    # Generate segments with no overlap
    num_segments = len(data) // segment_size
    segments = data[:num_segments * segment_size].reshape(num_segments, segment_size).copy()
    segments_array = torch.tensor(segments, dtype=torch.float16)

    return segments_array, torch.ones(segments_array.shape[0])*999, torch.tensor(np.arange(num_segments), dtype=torch.int32), \
                                                                    {"segment_size": segment_size, "stride": segment_size, "fs": fs}


def normalize_per_segment(inputs, take_power=True):
    """Normalize each segment independently: subtract its mean, divide by its std.
    Works for both (B, T) waveform and (B, C, F, T) spectrogram inputs.
    Mimics real-time usage where only the current segment is available."""
    if take_power:
        inputs = inputs ** 2
    dims = tuple(range(1, inputs.dim()))
    mean = inputs.mean(dim=dims, keepdim=True)
    std  = inputs.std(dim=dims, keepdim=True).clamp(min=1e-6)

    return (inputs - mean) / std
