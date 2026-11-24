"""Feature dims, weight sharing, inference decoupling, benchmark isolation (spec Section 33)."""
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from src.app.inference_engine import predict_authenticity
from src.evaluation.calibrator import PlattCalibrator
from src.models.backbones import RawNet2, RawNet2Encoder
from src.models.delta_net import DeltaNet
from src.models.differential_head import build_differential_features, feature_slices
from src.utils.benchmark_guard import assert_not_locked

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DeltaNet(RawNet2Encoder(proj_dim=256)).eval()


def test_rawnet2_frame_shape():
    z = RawNet2Encoder(proj_dim=None).eval()(torch.randn(2, 64600) * 0.1)
    assert z.shape == (2, 29, 128)


def test_rawnet2_full_forward():
    assert RawNet2().eval()(torch.randn(2, 64600) * 0.1).shape == (2, 2)


def test_feature_dims_1025(model):
    f = model.features(torch.randn(4, 64600) * 0.1)
    assert f.shape == (4, 1025)
    assert not torch.isnan(f).any()


def test_feature_slices():
    zy, zyp = torch.randn(3, 29, 256), torch.randn(3, 29, 256)
    f = build_differential_features(zy, zyp)
    s = feature_slices(256)
    assert torch.allclose(f[:, s["z_y"]], zy.mean(1))
    assert torch.allclose(f[:, s["z_yp"]], zyp.mean(1))
    assert torch.allclose(f[:, s["delta_mean"]], (zy - zyp).mean(1), atol=1e-6)
    assert torch.allclose(f[:, s["delta_std"]], (zy - zyp).std(1, unbiased=False), atol=1e-6)
    assert f[:, s["cos_dist"]].shape == (3, 1)


def test_identical_inputs_zero_delta():
    z = torch.randn(2, 29, 256)
    f = build_differential_features(z, z.clone())
    assert torch.allclose(f[:, 512:], torch.zeros(2, 513), atol=1e-5)


def test_weight_sharing(model):
    # one encoder module object => both branches use the same parameter tensors
    enc_params = {id(p) for p in model.encoder.parameters()}
    all_params = [id(p) for p in model.parameters()]
    assert len(all_params) == len(set(all_params))
    assert sum(1 for name, _ in model.named_children() if "encoder" in name) == 1
    assert enc_params <= set(all_params)


def test_gradients_reach_encoder(model):
    model.train()
    model(torch.randn(2, 64600) * 0.1).sum().backward()
    assert model.encoder.rawnet.block5[0].conv2.weight.grad is not None
    model.zero_grad(); model.eval()


def test_training_and_inference_share_feature_fn():
    src = inspect.getsource(predict_authenticity)
    assert "model.features(" in src
    assert "build_differential_features" in inspect.getsource(DeltaNet.features)


def test_inference_decoupling(model):
    params = list(inspect.signature(predict_authenticity).parameters)
    assert params == ["uploaded_audio_waveform", "model", "calibrator"]
    out = predict_authenticity(np.random.randn(30000).astype(np.float32) * 0.1, model, PlattCalibrator())
    assert 0 <= out["calibrated_bonafide_probability"] <= 1
    assert out["secondary_perturbation_applied"] == "mulaw8"


def test_benchmark_isolation():
    with pytest.raises(PermissionError):
        assert_not_locked("data/manifests/evaluation_manifest.csv")
    for d in ["src/data", "src/training", "experiments"]:
        for f in (ROOT / d).glob("*.py"):
            assert "evaluation_manifest.csv\"" not in f.read_text() or f.name == "run_phase0_audit.py", f


def test_checkpoint_key_layout():
    """Regression (EXP-001): official RawNet2 checkpoint stores FMS layers as fc_attention{i}.0.*"""
    keys = set(RawNet2().state_dict().keys())
    assert "fc_attention0.0.weight" in keys and "fc_attention5.0.bias" in keys
    assert not any(k.startswith("Sinc_conv") for k in keys)  # fixed filterbank, no stored tensors
