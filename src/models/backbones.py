"""RawNet2 (ASVspoof 2021 baseline layout) so official pretrained weights load with strict=True.

Layer names mirror the official `model.py` (Sinc_conv, first_bn, block0..5, fc_attention0..5,
bn_before_gru, gru, fc1_gru, fc2_gru). With a 64,600-sample input the last residual block
outputs [B, 128, 29]. NOTE: the spec states D=256; native RawNet2 gives D=128 (see README).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """Fixed (non-learnable) mel-spaced sinc band-pass filterbank, as in the 2021 baseline."""

    def __init__(self, out_channels=20, kernel_size=1024, sample_rate=16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels, self.kernel_size, self.sample_rate = out_channels, kernel_size, sample_rate
        nfft = 512
        f = int(sample_rate / 2) * np.linspace(0, 1, int(nfft / 2) + 1)
        fmel = 2595 * np.log10(1 + f / 700)
        mel_edges = np.linspace(fmel.min(), fmel.max(), out_channels + 1)
        hz = 700 * (10 ** (mel_edges / 2595) - 1)
        hsupp = np.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        bank = np.zeros((out_channels, kernel_size))
        win = np.hamming(kernel_size)
        for i in range(out_channels):
            fmin, fmax = hz[i], hz[i + 1]
            h_high = (2 * fmax / sample_rate) * np.sinc(2 * fmax * hsupp / sample_rate)
            h_low = (2 * fmin / sample_rate) * np.sinc(2 * fmin * hsupp / sample_rate)
            bank[i] = win * (h_high - h_low)
        # non-persistent: official checkpoints contain no tensors for Sinc_conv
        self.register_buffer("filters", torch.tensor(bank, dtype=torch.float32).view(out_channels, 1, kernel_size),
                             persistent=False)

    def forward(self, x):
        return F.conv1d(x, self.filters)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], kernel_size=3, padding=1)
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=1)
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        out = x
        if not self.first:
            out = self.lrelu(self.bn1(x))
        # Official baseline applies conv1 to `x` (pre-BN), kept for weight/behaviour parity.
        out = self.conv1(x)
        out = self.conv2(self.lrelu(self.bn2(out)))
        if self.downsample:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)


class RawNet2(nn.Module):
    FILTS = [20, [20, 20], [20, 128], [128, 128]]

    def __init__(self, first_conv=1024, gru_node=1024, nb_gru_layer=3, nb_fc_node=1024, nb_classes=2):
        super().__init__()
        f = self.FILTS
        self.Sinc_conv = SincConv(out_channels=f[0], kernel_size=first_conv)
        self.first_bn = nn.BatchNorm1d(f[0])
        self.selu = nn.SELU(inplace=True)
        specs = [(f[1], True), (f[1], False), (f[2], False), (f[3], False), (f[3], False), (f[3], False)]
        for i, (nf, first) in enumerate(specs):
            setattr(self, f"block{i}", nn.Sequential(Residual_block(nf, first=first)))
            # official checkpoint stores these as nn.Sequential(nn.Linear) -> keys "fc_attention{i}.0.*"
            setattr(self, f"fc_attention{i}", nn.Sequential(nn.Linear(nf[-1], nf[-1])))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.sig = nn.Sigmoid()
        self.bn_before_gru = nn.BatchNorm1d(f[3][-1])
        self.gru = nn.GRU(f[3][-1], gru_node, nb_gru_layer, batch_first=True)
        self.fc1_gru = nn.Linear(gru_node, nb_fc_node)
        self.fc2_gru = nn.Linear(nb_fc_node, nb_classes)

    def frame_features(self, x):
        """[B, L] -> [B, 128, T] (T=29 for L=64600)."""
        x = x.view(x.shape[0], 1, -1)
        x = self.selu(self.first_bn(F.max_pool1d(torch.abs(self.Sinc_conv(x)), 3)))
        for i in range(6):
            xi = getattr(self, f"block{i}")(x)
            y = self.avgpool(xi).view(xi.size(0), -1)
            y = self.sig(getattr(self, f"fc_attention{i}")(y)).view(xi.size(0), xi.size(1), -1)
            x = xi * y + y  # Feature Map Scaling
        return x

    def forward(self, x):
        x = self.selu(self.bn_before_gru(self.frame_features(x))).transpose(1, 2)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        return self.fc2_gru(self.fc1_gru(x[:, -1, :]))  # logits [B, 2]; official: index 1 = bonafide


class RawNet2Encoder(nn.Module):
    """Shared encoder phi: [B, L] -> [B, T, D]. Optional linear projection to proj_dim."""

    def __init__(self, rawnet: RawNet2 | None = None, proj_dim: int | None = None):
        super().__init__()
        self.rawnet = rawnet or RawNet2()
        self.proj = nn.Linear(128, proj_dim) if proj_dim else None
        self.out_dim = proj_dim or 128

    def forward(self, x):
        z = self.rawnet.frame_features(x).transpose(1, 2)  # [B, T, 128]
        return self.proj(z) if self.proj is not None else z


def load_pretrained_rawnet2(path: str, map_location="cpu") -> RawNet2:
    model = RawNet2()
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model
