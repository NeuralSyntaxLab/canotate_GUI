import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


model_params = {"conv_channels":[1,64, 24, 12], "kernels": [7, 5, 3], "strides": [2,2,2], "dropouts": [0.2,0.4, 0.4]}
model_params_new = {"conv_channels":[1,64, 48, 24, 12], "kernels": [4,2,2,2], "strides": [1,2,4,2], "dropouts": [0.13,0.4,0.4,0.12]} 
model_params_list = [model_params, model_params_new]

class _MPSSafeAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    def forward(self, x):
        if x.device.type == "mps":
            return super().forward(x.cpu()).to(x.device)
        return super().forward(x)


class ConvMLP(nn.Module):
    def __init__(self, input_size=256, model_params_idx=0, num_classes=1):
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
        if model_params_idx is None:
            model_params_idx = 0
        model_params = model_params_list[model_params_idx]
        print(f"Using model_params_idx={model_params_idx}: {model_params}")
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
            print(f"Conv output shape: {conv_out.shape}, flat_features: {self.flat_features}")

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


class SincConv_fast(nn.Module):
    """First-layer parametrized sinc bandpass filterbank. Learns only
    low-cutoff + bandwidth per filter (2 params/filter). Mono input only."""
    @staticmethod
    def to_mel(hz):  return 2595 * np.log10(1 + hz / 700)
    @staticmethod
    def to_hz(mel):  return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=44100,
                 stride=1, padding=0, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size + (1 - kernel_size % 2)  # force odd
        self.stride, self.padding = stride, padding
        self.sample_rate = sample_rate
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz

        low_hz, high_hz = 30, sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin = torch.linspace(0, self.kernel_size // 2 - 1, steps=self.kernel_size // 2)
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size))
        n = (self.kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    def forward(self, x):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        f_low = torch.matmul(low, self.n_)
        f_high = torch.matmul(high, self.n_)
        left = ((torch.sin(f_high) - torch.sin(f_low)) / (self.n_ / 2)) * self.window_
        center = 2 * band.view(-1, 1)
        right = torch.flip(left, dims=[1])
        filters = torch.cat([left, center, right], dim=1) / (2 * band[:, None])
        filters = filters.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=self.stride, padding=self.padding)

class DWSep(nn.Module):
    def __init__(self, i, o, k=9, pool=4, dropout=0.0):
        super().__init__()
        self.dw = nn.Conv1d(i, i, k, padding=k // 2, groups=i, bias=False)
        self.pw = nn.Conv1d(i, o, 1, bias=False)
        self.bn = nn.BatchNorm1d(o)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.pw(self.dw(x))))
        return self.drop(self.pool(x))


sinc_model_params_list = [
    {"sinc_filters": 64, "sinc_kernel": 129,   # 129 taps ≈ 2.9 ms; use 65 if input_size=256
     "conv_channels": [64, 128], "kernels": [9, 9],
     "pools": [4, 4, 4], "dropout": 0.2},
]


class SincEventNet(nn.Module):
    def __init__(self, input_size=661, model_params_idx=0, num_classes=1, sample_rate=44100):
        super().__init__()
        p = sinc_model_params_list[model_params_idx or 0]
        Fn, K = p["sinc_filters"], p["sinc_kernel"]
        c1, c2 = p["conv_channels"]; k1, k2 = p["kernels"]
        p0, p1, p2 = p["pools"]; d = p["dropout"]

        self.sinc = SincConv_fast(Fn, K, sample_rate=sample_rate, padding=K // 2)
        self.sinc_post = nn.Sequential(
            nn.BatchNorm1d(Fn), nn.LeakyReLU(0.01, inplace=True), nn.MaxPool1d(p0))
        self.block1 = DWSep(Fn, c1, k1, p1, d)
        self.block2 = DWSep(c1, c2, k2, p2, d)
        self.num_classes = num_classes
        self.head = nn.Linear(c2, num_classes)

        with torch.no_grad():
            f = self._features(torch.randn(1, 1, input_size))
            print(f"[SincEventNet] pre-GAP feature shape: {tuple(f.shape)}")

    def _features(self, x):
        return self.block2(self.block1(self.sinc_post(self.sinc(x))))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self._features(x)
        x = x.mean(dim=-1)          # MPS-safe global average pool
        return self.head(x)         # raw logits
    
fc_model_params_list = [
    # 5 blocks: conv_channels has n_blocks + 1 entries (leading 1 = mono input),
    # while kernels / strides / pools each have exactly n_blocks entries.
    # Total temporal downsampling = prod(strides) * prod(pools) = 4 * 16 = 64.
    {"conv_channels": [1, 256, 128, 64, 64, 32], "kernels": [16, 8, 8, 4, 4],
     "strides": [1, 2, 2, 1, 1], "pools": [1, 4, 1, 4, 1], "dropout": 0.3},
]


def validate_fc_arch(arch):
    """Check that a FullyConvEventNet architecture spec is internally consistent.

    ``FullyConvEventNet`` builds one block per entry of ``kernels`` and indexes
    ``strides``/``pools`` with the same counter, so mismatched list lengths
    surface as a bare ``IndexError`` deep inside ``__init__``. Validating up
    front turns that into an actionable message.

    Args:
        arch: dict with keys 'conv_channels', 'kernels', 'strides', 'pools',
            'dropout'.

    Returns:
        The number of blocks the spec describes.

    Raises:
        ValueError: if a key is missing, if 'kernels'/'strides'/'pools' do not
            all have the same length, or if 'conv_channels' does not have
            exactly one more entry than there are blocks.
    """
    required_keys = ("conv_channels", "kernels", "strides", "pools", "dropout")
    missing = [key for key in required_keys if key not in arch]
    if missing:
        raise ValueError(
            f"FullyConvEventNet arch is missing key(s) {missing}; "
            f"expected all of {list(required_keys)}."
        )

    num_blocks = len(arch["kernels"])
    per_block_lengths = {name: len(arch[name]) for name in ("kernels", "strides", "pools")}
    if len(set(per_block_lengths.values())) != 1:
        raise ValueError(
            f"FullyConvEventNet arch has inconsistent per-block list lengths "
            f"{per_block_lengths}; 'kernels', 'strides' and 'pools' must all have "
            f"one entry per block."
        )
    if len(arch["conv_channels"]) != num_blocks + 1:
        raise ValueError(
            f"FullyConvEventNet arch has {len(arch['conv_channels'])} conv_channels "
            f"but describes {num_blocks} blocks; conv_channels needs {num_blocks + 1} "
            f"entries (the leading entry is the input channel count)."
        )
    return num_blocks


class FullyConvEventNet(nn.Module):
    def __init__(self, input_size=661, model_params_idx=0, num_classes=1, arch=None,
                 verbose=True):
        """Fully convolutional 1-D event classifier over raw waveform segments.

        Args:
            input_size: waveform segment length in samples; used only for the
                pre-head feature-shape probe.
            model_params_idx: index into ``fc_model_params_list``; ignored when
                ``arch`` is given.
            num_classes: number of output logits (1 for binary).
            arch: optional architecture spec dict overriding
                ``fc_model_params_list[model_params_idx]``. Same keys, so a
                hyperparameter search can build arbitrary variants without
                mutating module-level state. Validated by ``validate_fc_arch``.
            verbose: print the pre-head feature shape. Set False when building
                many models in a loop (a search sweep) to keep the log readable.
        """
        super().__init__()
        p = fc_model_params_list[model_params_idx or 0] if arch is None else arch
        validate_fc_arch(p)
        ch, ks, st, po, d = (p["conv_channels"], p["kernels"],
                             p["strides"], p["pools"], p["dropout"])
        blocks = []
        for i in range(len(ks)):
            blocks += [
                nn.Conv1d(ch[i], ch[i + 1], ks[i], stride=st[i],
                          padding=ks[i] // 2, bias=False),
                nn.BatchNorm1d(ch[i + 1]),
                nn.LeakyReLU(0.01, inplace=True),
            ]
            if po[i] > 1:
                blocks.append(nn.MaxPool1d(po[i]))
        self.features = nn.Sequential(*blocks)
        self.drop = nn.Dropout1d(d)
        self.classifier = nn.Conv1d(ch[-1], num_classes, 1)   # 1x1 conv -> fully conv
        self.num_classes = num_classes
        # Recorded so ONNX export can recover the segment length from the module
        # itself when checkpoint metadata is unavailable (see compare_models).
        self.input_size = input_size

        if verbose:
            with torch.no_grad():
                f = self.features(torch.randn(1, 1, input_size))
                print(f"[FullyConvEventNet] pre-head feature shape: {tuple(f.shape)}")

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.drop(x)
        x = self.classifier(x)        # (B, num_classes, L)
        return x.mean(dim=-1)         # global avg pool over time -> (B, num_classes) logits


WAVEFORM_MODEL_NAMES = ["ConvMLP", "SincEventNet", "FullyConvEventNet"]



def build_waveform_model(name, input_size, num_classes, sample_rate=44100,
                         model_params_idx=None, fc_arch=None):
    """Instantiate one waveform model by name.

    Shared by training_loop.main() and compare_models so the two entry points
    cannot drift apart.

    Args:
        name: one of WAVEFORM_MODEL_NAMES.
        input_size: waveform segment length in samples (pinned to the data).
        num_classes: number of output logits (1 for binary).
        sample_rate: audio sample rate in Hz; SincEventNet's filterbank only.
        model_params_idx: conv-config index. Only ConvMLP has more than one
            parameter set, so the other two are always built with index 0.
        fc_arch: optional FullyConvEventNet architecture spec dict (see
            ``validate_fc_arch``) that replaces ``fc_model_params_list``. Used by
            the hyperparameter search and by configs it writes out; ignored for
            the other model names.

    Returns:
        An un-initialised nn.Module on CPU.

    Raises:
        ValueError: if ``name`` is not a known waveform model, or if ``fc_arch``
            is given for a model other than FullyConvEventNet.
    """
    if fc_arch is not None and name != "FullyConvEventNet":
        raise ValueError(
            f"fc_arch only applies to FullyConvEventNet, but name={name!r}; "
            f"drop 'fc_arch' from the config or switch model_name."
        )
    if name == "ConvMLP":
        return ConvMLP(input_size=input_size, num_classes=num_classes,
                       model_params_idx=model_params_idx or 0)
    if name == "SincEventNet":
        return SincEventNet(input_size=input_size, num_classes=num_classes,
                            model_params_idx=0, sample_rate=sample_rate)
    if name == "FullyConvEventNet":
        return FullyConvEventNet(input_size=input_size, num_classes=num_classes,
                                 model_params_idx=0, arch=fc_arch)
    raise ValueError(f"Unknown waveform model name: {name!r}. "
                     f"Expected one of {WAVEFORM_MODEL_NAMES}.")


class ClassificationCnn(nn.Module):
        
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(p=0.1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        
        self.flattened_size = self._get_flattened_size(input_shape)
        self.fc = nn.Linear(self.flattened_size, num_classes)
        # NOTE: forward() returns raw logits — pair with BCEWithLogitsLoss
        # (num_classes==1) or CrossEntropyLoss (num_classes>=2). Apply the
        # appropriate activation only at inference / thresholding time.

    def _get_flattened_size(self, input_shape):
        x = torch.randn(1, *input_shape)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.dropout3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        return x.size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.dropout3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # raw logits
    

#for testing compilation of the model
if __name__ == "__main__":
    batch_size, input_size = 8, 512
    sample_input = torch.randn(batch_size, input_size)  # (B, T) waveform batch
    builders = [
        ("ConvMLP", lambda: ConvMLP(input_size=input_size, model_params_idx=0, num_classes=1)),
        ("SincEventNet", lambda: SincEventNet(input_size=input_size, model_params_idx=0, num_classes=1)),
        ("FullyConvEventNet", lambda: FullyConvEventNet(input_size=input_size, model_params_idx=0, num_classes=1)),
    ]
    for model_name, build_model in builders:
        model = build_model()
        output = model(sample_input)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{model_name}: output shape {tuple(output.shape)}, params {num_params:,}")