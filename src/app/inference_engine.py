"""Single-file production inference. Accepts ONLY the uploaded audio — no clean reference x."""
import numpy as np
import torch

from src.data.audio_processing import preprocess


def generate_safety_advisory(pred_label: str, margin: float) -> list[str]:
    adv = ["Automated detection is probabilistic; treat this as an advisory, not proof."]
    if pred_label != "Bona Fide Human" or margin < 0.3:
        adv = ["Do NOT send money, transfers or gift cards based on this audio.",
               "Call the person back on a known, trusted number.",
               "Ask a private question only the real contact would know."] + adv
    return adv


@torch.no_grad()
def predict_authenticity(uploaded_audio_waveform: np.ndarray, model, calibrator) -> dict:
    model.eval()
    y = torch.from_numpy(preprocess(np.asarray(uploaded_audio_waveform, np.float32), mode="center")).unsqueeze(0)
    y = y.to(next(model.parameters()).device)
    f = model.features(y)  # same feature function as training
    assert f.shape == (1, model.feat_dim)
    lg = model.head(f)
    s = (lg[0, 1] - lg[0, 0]).item()
    p = float(calibrator.predict_proba(s))
    label = "Bona Fide Human" if p >= 0.5 else "Likely AI-Generated"
    margin = round(abs(p - 0.5) * 2.0, 3)
    return {"prediction": label, "calibrated_bonafide_probability": round(p, 4),
            "calibrated_spoof_probability": round(1 - p, 4), "decision_margin": margin,
            "secondary_perturbation_applied": model.perturbation_name,
            "safety_advisory": generate_safety_advisory(label, margin)}
