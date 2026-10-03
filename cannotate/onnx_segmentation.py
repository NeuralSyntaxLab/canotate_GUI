"""The one ONNX wrapper the segmentation path uses.

There used to be two copies of this class -- one in ``spectrogram_viewer`` and
one in ``full_annotation_pipeline`` -- and they drifted. The pipeline's copy
grew a ``preprocess_input`` that squared and z-scored its input, which
``InferenceToAnnotationData.run_inference`` had already done via
``normalize_per_segment``. The result was that Full Auto Annotate ran the
segmenter on doubly-normalized input while Run NN Segmentation ran it on
correctly-normalized input, so the same song segmented differently depending on
which button was pressed.

``normalize_per_segment`` is the normalization ``bird_song_detector``'s training
loop applied, so it is what the weights were trained against, and it belongs to
``run_inference`` alone. This wrapper therefore passes its input straight
through: it is a *runtime* adapter, not a preprocessing step.

The module is deliberately free of GUI imports so the headless pipeline can use
it without a display.
"""

# torch may be absent: manual annotation never needs it, and the GUI greys the
# NN buttons out rather than failing to start. The annotations below are quoted
# for the same reason -- a bare torch.Tensor annotation is evaluated at class
# definition time and would raise on a machine without the deep-learning stack.
try:
    import torch
except ImportError:  # pragma: no cover - exercised only on torch-less machines
    torch = None


class OnnxSegmentationModel:
    """Wraps an ONNX InferenceSession to look like a PyTorch model.

    Compatible with ``InferenceToAnnotationData.run_inference()``, which calls
    ``model.eval()``, ``model.to(device)``, and ``model(tensor)`` and then
    applies ``torch.sigmoid`` on the result.

    The ONNX graph accepts ``[batch, 256]`` float32 and returns raw logits
    ``[batch, 1]`` — identical contract to the PyTorch ConvMLP.
    """

    def __init__(self, onnx_path):
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        try:
            available = ort.get_available_providers()
            for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider"):
                if p in available:
                    providers.insert(0, p)
                    break
        except Exception:
            pass
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def eval(self):
        return self

    def to(self, device):
        return self  # onnxruntime selects its own execution provider

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        x_np = x.cpu().numpy().astype("float32")
        outputs = self._session.run(None, {self._input_name: x_np})
        return torch.from_numpy(outputs[0])
