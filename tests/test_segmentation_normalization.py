"""The segmenter's input must be normalized exactly once.

``InferenceToAnnotationData.run_inference`` applies ``normalize_per_segment``
before handing segments to the model. That function is the one
``bird_song_detector``'s training loop applied, so it is the normalization the
weights were trained against. An ONNX wrapper that normalizes *again* feeds the
network input of a kind it has never seen.

``full_annotation_pipeline.OnnxSegmentationModel`` used to do exactly that: its
``preprocess_input`` squared and z-scored on top of ``run_inference``'s own
squaring and z-scoring. The viewer's wrapper did not. So Full Auto Annotate ran
the segmenter on doubly-normalized input while Run NN Segmentation ran it on
correctly-normalized input, and the two buttons disagreed about the same song.

These tests pin the single-normalization contract and the single wrapper.

Runs under pytest or directly:  python tests/test_segmentation_normalization.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from bird_song_detector._helpers import load_from_wav, normalize_per_segment
from bird_song_detector.nn_segmentation import InferenceToAnnotationData

SEGMENT_SIZE = 256
N_SEGMENTS = 40
FS = 44100


class _RecordingSession:
    """Stands in for an ``onnxruntime.InferenceSession`` and keeps what it was fed.

    The real package is not installed in every environment this suite runs in,
    and the question under test -- what reaches the network -- is answerable
    without it.
    """

    def __init__(self, path, providers=None):
        self.path = path
        self.providers = providers
        self.received = []

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def run(self, output_names, feed):
        x = np.asarray(feed["input"])
        self.received.append(x.copy())
        return [np.zeros((x.shape[0], 1), dtype=np.float32)]


@pytest.fixture
def fake_onnxruntime(monkeypatch):
    """Install a stand-in ``onnxruntime`` that records every session it builds."""
    module = types.ModuleType("onnxruntime")
    built = []

    class InferenceSession(_RecordingSession):
        def __init__(self, path, providers=None):
            super().__init__(path, providers)
            built.append(self)

    module.InferenceSession = InferenceSession
    module.get_available_providers = lambda: ["CPUExecutionProvider"]
    module.built = built
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return module


@pytest.fixture
def wav_path(tmp_path):
    """A wav long enough to make several whole 256-sample segments."""
    rng = np.random.default_rng(13)
    audio = rng.integers(-8000, 8000, SEGMENT_SIZE * N_SEGMENTS, dtype=np.int16)
    path = tmp_path / "bird_0001_2024_07_05_06_58_44.wav"
    wavfile.write(path, FS, audio)
    return path


def test_onnx_segmenter_receives_input_normalized_exactly_once(
    fake_onnxruntime, wav_path
):
    """What reaches the ONNX session is run_inference's normalization, nothing more."""
    from full_annotation_pipeline import OnnxSegmentationModel

    model = OnnxSegmentationModel("unused.onnx")
    InferenceToAnnotationData(model, torch.device("cpu")).run_inference(str(wav_path))

    session = fake_onnxruntime.built[-1]
    assert len(session.received) == 1, "expected one forward pass over the file"

    data, _, _, _ = load_from_wav(str(wav_path))
    expected = normalize_per_segment(data.float()).numpy()
    np.testing.assert_allclose(session.received[0], expected, rtol=1e-6, atol=1e-6)


def test_both_call_sites_share_one_onnx_wrapper():
    """Two copies of this wrapper is how the two buttons came to disagree."""
    from cannotate.spectrogram_viewer import OnnxSegmentationModel as viewer_cls
    from full_annotation_pipeline import OnnxSegmentationModel as pipeline_cls

    assert viewer_cls is pipeline_cls


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
