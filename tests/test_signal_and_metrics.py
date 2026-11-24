"""mu-law numerics, audio windowing, metrics vs scikit-learn."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data.audio_processing import fix_length, peak_normalize, preprocess
from src.evaluation.metrics import bootstrap_ci, eer, roc_auc, threshold_rates
from src.models.degradation_engine import apply_gaussian_noise, apply_mulaw_quantization, apply_uniform_quantization


def test_mulaw_stability():
    for x in [torch.zeros(2, 64600), torch.ones(2, 64600) * 3.0, torch.full((2, 64600), 0.3),
              torch.tensor([[float("nan"), float("inf"), -float("inf"), 0.5]])]:
        y = apply_mulaw_quantization(x)
        assert y.shape == x.shape and torch.isfinite(y).all() and y.abs().max() <= 1.0


def test_mulaw_256_levels_and_error():
    x = torch.linspace(-1, 1, 100000)
    y = apply_mulaw_quantization(x)
    assert len(torch.unique(y)) <= 256
    assert (y - x).abs().max() < 0.04          # coarse steps near full scale
    small = torch.linspace(-0.01, 0.01, 1000)
    assert (apply_mulaw_quantization(small) - small).abs().max() < 5e-4  # fine steps near zero
    assert torch.equal(apply_mulaw_quantization(x), y)  # deterministic


def test_other_perturbations():
    x = torch.randn(1, 1000) * 0.2
    assert len(torch.unique(apply_uniform_quantization(x))) <= 256
    assert torch.equal(apply_gaussian_noise(x, seed=1), apply_gaussian_noise(x, seed=1))


def test_windowing():
    w, m = fix_length(np.random.randn(1000).astype(np.float32))
    assert w.shape == (64600,) and m.sum() == 1000
    w, m = fix_length(np.random.randn(100000).astype(np.float32))
    assert w.shape == (64600,) and m.all()
    w, _ = fix_length(np.ones(5, np.float32), pad="zero")
    assert w[5:].sum() == 0
    assert abs(np.abs(preprocess(np.random.randn(70000))).max() - 0.95) < 1e-5
    assert np.all(peak_normalize(np.zeros(10)) == 0)


def test_metrics():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(300), np.zeros(700)].astype(int)
    s = np.r_[rng.normal(1, 1, 300), rng.normal(0, 1, 700)]
    assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-9
    assert eer(y, np.r_[np.ones(300), np.zeros(700)])[0] == 0.0
    e, _ = eer(y, s)
    assert 0.25 < e < 0.40  # d'=1 -> EER ~0.31
    r = threshold_rates(y, s, 0.5)
    assert r["TP_spoof"] + r["FN_spoof"] == 700 and r["TN_bona"] + r["FP_bona"] == 300
    p, lo, hi = bootstrap_ci(y, s, roc_auc, n_iter=200)
    assert lo <= p <= hi
