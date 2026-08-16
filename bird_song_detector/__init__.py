"""bird_song_detector - vendored subset.

This is NOT the full bird_song_detector package. It carries only the pieces
the Cannotate annotation tool imports:

    model_zoo.ConvMLP, model_zoo.ClassificationCnn
    nn_segmentation.InferenceToAnnotationData

The upstream ``__init__`` also re-exported ``load_from_wav``, ``load_data``,
``AudioAugmentation`` and ``compute_spectrogram``, which pulled in h5py and
matplotlib at import time. Nothing here uses them, so they are omitted.

See PROVENANCE.md before editing anything in this directory.
"""

from bird_song_detector.model_zoo import ConvMLP, ClassificationCnn

__all__ = [
    "ConvMLP",
    "ClassificationCnn",
]
