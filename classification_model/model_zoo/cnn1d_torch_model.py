from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import json
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


NORMALIZATION_MODES = {"segment_peak", "train_zscore", "none"}


def make_normalization_config(
    mode: str = "segment_peak",
    mean: Optional[float] = None,
    std: Optional[float] = None,
    eps: float = 1e-8,
) -> Dict:
    """Create a JSON-serializable waveform normalization config.

    Kept byte-compatible with tweetynet_flow so the same trained-model
    metadata.json round-trips identically across training and this runtime.
    """
    mode = str(mode)
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {mode!r}. Expected one of {sorted(NORMALIZATION_MODES)}")

    cfg: Dict[str, object] = {"mode": mode, "eps": float(eps)}
    if mode == "train_zscore":
        if mean is None or std is None:
            raise ValueError("train_zscore normalization requires mean and std")
        std = float(std)
        if not np.isfinite(std) or std <= float(eps):
            raise ValueError(f"Invalid train_zscore std={std}. It must be finite and > eps={eps}.")
        cfg["mean"] = float(mean)
        cfg["std"] = std
    return cfg


def compute_train_zscore_config(train_segments: np.ndarray, eps: float = 1e-8) -> Dict:
    """Compute global z-score parameters from training segments only."""
    x = np.asarray(train_segments, dtype=np.float32)
    if x.size == 0:
        raise ValueError("Cannot compute train_zscore normalization from an empty training array")
    mean = float(np.mean(x))
    std = float(np.std(x))
    return make_normalization_config(mode="train_zscore", mean=mean, std=std, eps=eps)


def normalize_segment(seg: np.ndarray, normalization_config: Optional[Dict] = None) -> np.ndarray:
    """Normalize one waveform segment according to a parameterized config.

    Supported modes
    ---------------
    segment_peak
        Subtract this segment's mean and divide by its max absolute amplitude.
    train_zscore
        Subtract a fixed training-set mean and divide by a fixed training-set std.
    none
        Return the segment as float32 without scaling.
    """
    cfg = normalization_config or {"mode": "segment_peak", "eps": 1e-8}
    mode = str(cfg.get("mode", "segment_peak"))
    eps = float(cfg.get("eps", 1e-8))

    seg = np.asarray(seg, dtype=np.float32)

    if mode == "none":
        return seg

    if mode == "segment_peak":
        seg = seg - np.mean(seg)
        peak = np.max(np.abs(seg))
        if peak > eps:
            seg = seg / peak
        return seg.astype(np.float32)

    if mode == "train_zscore":
        if "mean" not in cfg or "std" not in cfg:
            raise ValueError("train_zscore normalization requires 'mean' and 'std' in normalization_config")
        std = float(cfg["std"])
        if std <= eps:
            raise ValueError(f"Invalid train_zscore std={std}; expected > eps={eps}")
        return ((seg - float(cfg["mean"])) / std).astype(np.float32)

    raise ValueError(f"Unknown normalization mode: {mode!r}. Expected one of {sorted(NORMALIZATION_MODES)}")


def normalize_segments(segments: np.ndarray, normalization_config: Optional[Dict] = None) -> np.ndarray:
    """Normalize a [n_segments, n_time] array using the same config."""
    x = np.asarray(segments, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected segments with shape [n_samples, n_time], got {x.shape}")

    cfg = normalization_config or {"mode": "segment_peak", "eps": 1e-8}
    mode = str(cfg.get("mode", "segment_peak"))
    eps = float(cfg.get("eps", 1e-8))

    if mode == "none":
        return x.astype(np.float32, copy=False)

    if mode == "segment_peak":
        out = x - np.mean(x, axis=1, keepdims=True)
        peak = np.max(np.abs(out), axis=1, keepdims=True)
        out = np.divide(out, peak, out=np.zeros_like(out, dtype=np.float32), where=peak > eps)
        return out.astype(np.float32, copy=False)

    if mode == "train_zscore":
        if "mean" not in cfg or "std" not in cfg:
            raise ValueError("train_zscore normalization requires 'mean' and 'std' in normalization_config")
        std = float(cfg["std"])
        if std <= eps:
            raise ValueError(f"Invalid train_zscore std={std}; expected > eps={eps}")
        return ((x - float(cfg["mean"])) / std).astype(np.float32)

    raise ValueError(f"Unknown normalization mode: {mode!r}. Expected one of {sorted(NORMALIZATION_MODES)}")


@dataclass
class Conv1DHyperparams:
    learning_rate: float = 2e-4
    batch_size: int = 64
    epochs: int = 40
    weight_decay: float = 1e-4
    dropout_rate: List[float] = field(default_factory=lambda: [0.13, 0.40, 0.40, 0.16])
    kernel_size: List[int] = field(default_factory=lambda: [16, 8, 4, 4])
    strides: List[int] = field(default_factory=lambda: [1, 2, 4, 2])
    hidden_dim: int = 32
    leaky_relu_slope: float = 0.01
    early_stopping_patience: int = 8
    use_batch_norm: bool = True
    normalize_waveform: bool = True
    # Present so metadata.json from newer tweetynet_flow training round-trips here.
    # "segment_peak" (default) matches deployed models; "train_zscore" uses persisted
    # mean/std carried in metadata's normalization_config.
    normalization_mode: str = "segment_peak"
    normalization_eps: float = 1e-8
    class_weighting: str = "balanced"  # "balanced" or "none"
    sample_secs: Optional[float] = None  # if None, infer from median GT segment duration
    max_per_class: Optional[int] = None
    random_seed: int = 13

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")


class SegmentDataset(Dataset):
    def __init__(self, segments: np.ndarray, labels: np.ndarray):
        if segments.ndim != 2:
            raise ValueError(f"Expected segments with shape [n_samples, n_time], got {segments.shape}")
        self.X = torch.from_numpy(segments[:, None, :].astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class Conv1DCNN(nn.Module):
    """Lean 1D CNN for classifying fixed-length waveform segments.

    Input shape: [batch, 1, n_samples]
    Output shape: [batch, n_classes]
    """

    def __init__(self, input_num_samples: int, n_classes: int, hp: Conv1DHyperparams):
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2 for multiclass classification")

        channels = [1, 64, 48, 24, 20]
        ks = hp.kernel_size
        st = hp.strides
        dr = hp.dropout_rate
        if not (len(ks) == len(st) == len(dr) == 4):
            raise ValueError("kernel_size, strides, and dropout_rate must all have length 4")

        def block(in_ch: int, out_ch: int, kernel_size: int, stride: int, dropout: float, use_bias: bool = True):
            layers = [nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=use_bias)]
            if hp.use_batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.extend([
                nn.LeakyReLU(negative_slope=hp.leaky_relu_slope, inplace=True),
                nn.Dropout(dropout),
            ])
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(channels[0], channels[1], ks[0], st[0], dr[0], use_bias=False),
            block(channels[1], channels[2], ks[1], st[1], dr[1]),
            block(channels[2], channels[3], ks[2], st[2], dr[2]),
            block(channels[3], channels[4], ks[3], st[3], dr[3]),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_num_samples)
            flattened = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, hp.hidden_dim),
            nn.LeakyReLU(negative_slope=hp.leaky_relu_slope, inplace=True),
            nn.Linear(hp.hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class NumpyLabelEncoder:
    def __init__(self):
        self.classes_: Optional[np.ndarray] = None
        self.class_to_idx: Dict[str, int] = {}

    def fit(self, labels: np.ndarray) -> "NumpyLabelEncoder":
        classes = np.unique(labels.astype(str))
        self.classes_ = classes
        self.class_to_idx = {label: idx for idx, label in enumerate(classes.tolist())}
        return self

    def transform(self, labels: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Encoder has not been fit")
        return np.asarray([self.class_to_idx[str(label)] for label in labels], dtype=np.int64)

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Encoder has not been fit")
        return self.classes_[indices]

    def save(self, path: Path) -> None:
        if self.classes_ is None:
            raise RuntimeError("Encoder has not been fit")
        payload = {"classes": self.classes_.tolist()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NumpyLabelEncoder":
        """Restore the saved class order *verbatim*.

        The saved order **is** the model's mapping from output index to label:
        index i of the final Linear layer means ``classes[i]``. So it must be
        adopted as-is, never re-derived. ``fit`` cannot be reused here because
        it sorts (and de-duplicates) - for a model whose classes were not saved
        in lexicographic order that silently shifts every prediction onto the
        wrong label, with no error anywhere to show it.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"label_encoder.json not found next to the model: {path}. "
                "It holds the output-index -> label mapping and is written "
                "beside the weights by Conv1DClassifier.save; a model folder "
                "cannot be used without the one that was saved with it."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = [str(c) for c in payload["classes"]]
        if not labels:
            raise ValueError(f"{path} lists no classes")
        if len(set(labels)) != len(labels):
            dupes = sorted({l for l in labels if labels.count(l) > 1})
            raise ValueError(
                f"{path} maps several output indices to the same label(s) "
                f"{dupes}; the mapping is ambiguous."
            )
        enc = cls()
        enc.classes_ = np.asarray(labels, dtype=str)
        enc.class_to_idx = {label: idx for idx, label in enumerate(labels)}
        return enc


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")



model_params_new_kernel_1 = {"conv_channels":[1,64, 24, 12], "kernels": [7,5,3], "strides": [2,2,2], "dropouts": [0.2,0.4, 0.4]}


model_params_new = {"conv_channels":[1,64, 48, 24, 12], "kernels": [4,2,2,2], "strides": [1,2,4,2], "dropouts": [0.13,0.4,0.4,0.12]} 


class _MPSSafeAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    def forward(self, x):
        if x.device.type == "mps":
            return super().forward(x.cpu()).to(x.device)
        return super().forward(x)


class ConvMLP(nn.Module):
    def __init__(self, input_size=256, num_classes=1):
        """
        Parameters
        ----------
        input_size  : int – waveform segment length in samples (default 256).
        model_params: dict – conv-block config (channels/kernels/strides/dropouts).
        num_classes : int – number of output logits (default 1, preserving the
                            original binary single-logit behaviour). Pass >=2 to
                            use the network for multiclass classification with
                            CrossEntropyLoss/argmax downstream.

        forward() returns RAW LOGITS — callers are expected to apply
        torch.sigmoid (binary, num_classes==1) or torch.softmax (multiclass)
        themselves. This lets us pair the model with BCEWithLogitsLoss or
        CrossEntropyLoss directly, both of which are numerically more stable
        than applying the activation in forward and then doing BCELoss / CE
        on top.
        """
        super(ConvMLP, self).__init__()

        model_params = model_params_new_kernel_1
        channels = model_params["conv_channels"]
        kernels = model_params["kernels"]
        strides = model_params["strides"]
        dropouts = model_params["dropouts"]
        num_blocks = len(channels) - 1

        blocks = [self.block(channels[i], channels[i+1], kernels[i], strides[i], dropouts[i], batch_norm=True) for i in range(num_blocks)]

        with torch.no_grad():
            sample_input = torch.randn(1, 1, input_size)
            conv_out = nn.Sequential(*blocks)(sample_input)
            target_L = conv_out.shape[2]
            self.flat_features = conv_out.shape[1] * target_L

        blocks.append(_MPSSafeAdaptiveAvgPool1d(target_L))
        blocks.append(nn.Flatten())
        self.audio_encoder = nn.Sequential(*blocks)

        self.num_classes = num_classes
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.flat_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def block(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, dropout: float, use_bias: bool = True, batch_norm: bool = False):
            layers = [nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=use_bias)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.extend([
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
            ])
            return nn.Sequential(*layers)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.audio_encoder(x)
        x = self.mlp_layers(x)
        return x  # raw logits; apply sigmoid (binary) / softmax (multiclass) at use-site
