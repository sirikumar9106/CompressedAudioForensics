"""EXP-008 detector: WavLM-Base+ with the top N transformer layers fine-tuned.

  waveform -> WavLM (bottom layers frozen, top N trainable) -> learned softmax weights over all 13 hidden
  states (keeps the "all layers matter" finding of EXP-004) -> attentive statistics pooling -> MLP -> 2 logits
Labels: 1 = real (bona fide), 0 = AI. Score = logit[1] - logit[0] (higher = more real).
"""
import torch
import torch.nn as nn


class AttentiveStatsPool(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h):                       # h [B, T, D]
        w = torch.softmax(self.att(h), dim=1)   # [B, T, 1]
        mu = (w * h).sum(1)
        sd = torch.sqrt(((w * (h - mu.unsqueeze(1)) ** 2).sum(1)).clamp(min=1e-6))
        return torch.cat([mu, sd], -1)


class WavLMDetector(nn.Module):
    def __init__(self, name="microsoft/wavlm-base-plus", n_trainable=4, dropout=0.2, low_branch=False, n_low=7):
        super().__init__()
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained(name)
        self.wavlm.config.apply_spec_augment = False
        self.wavlm.config.layerdrop = 0.0          # keep a fixed set of hidden states for the layer weights
        for p in self.wavlm.parameters():
            p.requires_grad = False
        layers = self.wavlm.encoder.layers
        for l in layers[len(layers) - n_trainable:]:
            for p in l.parameters():
                p.requires_grad = True
        n_states = len(layers) + 1
        d = self.wavlm.config.hidden_size
        self.layer_w = nn.Parameter(torch.zeros(n_states))
        self.pool = AttentiveStatsPool(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 2))
        self.n_low = n_low if low_branch else 0
        if low_branch:   # EXP-010: separate judge on the low "sound texture" layers (CNN + layers 1..n_low-1)
            self.low_w = nn.Parameter(torch.zeros(n_low))
            self.low_pool = AttentiveStatsPool(d)
            self.low_head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2))

    def from_states(self, hs, layer_w=None, low_w=None, parts=False):
        """hs: stacked hidden states [L, B, T, D]. Returns logits (and per-branch logits if parts)."""
        lw = self.layer_w if layer_w is None else layer_w
        h = (torch.softmax(lw, 0)[:, None, None, None] * hs).sum(0)
        main = self.head(self.pool(h))
        if not self.n_low:
            return (main, main, None) if parts else main
        lo_w = self.low_w if low_w is None else low_w
        hl = (torch.softmax(lo_w, 0)[:, None, None, None] * hs[:self.n_low]).sum(0)
        low = self.low_head(self.low_pool(hl))
        return (main + low, main, low) if parts else main + low

    def forward(self, x, parts=False):           # x [B, L] 16 kHz, peak-normalised
        hs = torch.stack(self.wavlm(x, output_hidden_states=True).hidden_states, 0)
        return self.from_states(hs, parts=parts)

    def param_groups(self, lr_backbone, lr_head):
        bb = [p for p in self.wavlm.parameters() if p.requires_grad]
        hd = [self.layer_w] + list(self.pool.parameters()) + list(self.head.parameters())
        if self.n_low:
            hd += [self.low_w] + list(self.low_pool.parameters()) + list(self.low_head.parameters())
        return [{"params": bb, "lr": lr_backbone}, {"params": hd, "lr": lr_head}]
