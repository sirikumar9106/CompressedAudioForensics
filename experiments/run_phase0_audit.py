"""Phase 0: environment check, repo/data inventory, locked-manifest metadata, historical metrics audit.

Usage:  python experiments/run_phase0_audit.py [--pred results/raw_predictions_Model_A_orig.csv]
Writes results/phase0_report.json
"""
import argparse
import csv
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.benchmark_guard import manifest_metadata  # noqa: E402


def env_report() -> dict:
    r = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ["torch", "torchaudio", "numpy", "scipy", "sklearn", "soundfile", "yaml", "pytest", "gradio"]:
        try:
            r[m] = getattr(importlib.import_module(m), "__version__", "installed")
        except Exception:
            r[m] = "MISSING"
    try:
        import torch
        r["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            r["gpu"] = torch.cuda.get_device_name(0)
            r["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            r["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    ff = shutil.which("ffmpeg")
    r["ffmpeg"] = ff or "MISSING"
    if ff:
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        r["ffmpeg_encoders"] = {e: (e in enc) for e in ["libmp3lame", "libopus", "aac"]}
    return r


def inventory() -> dict:
    need = {
        "pretrained_rawnet2": "models/pre_trained_DF_RawNet2.pth",
        "manifest_2019_train": "data/manifests/asvspoof2019_train.csv",
        "manifest_2019_dev": "data/manifests/asvspoof2019_dev.csv",
        "historical_predictions_model_A": "results/raw_predictions_Model_A_orig.csv",
    }
    return {k: {"path": v, "exists": (ROOT / v).exists()} for k, v in need.items()}


def historical_audit(pred_path: Path) -> dict:
    """Recompute EER and fixed-threshold FPR/FNR under BOTH score directions and report which
    interpretation reproduces the inherited 14.80% / 1.80% / 27.60% figures."""
    if not pred_path.exists():
        return {"status": "SKIPPED", "reason": f"{pred_path} not found"}
    import numpy as np
    from src.evaluation.metrics import eer, roc_auc, threshold_rates
    rows = list(csv.DictReader(open(pred_path, newline="")))
    cols = list(rows[0].keys())
    lab_col = next((c for c in cols if c.lower() in ("label", "true_label", "target", "y_true", "key")), None)
    sc_col = next((c for c in cols if c.lower() in ("score", "bonafide_score", "prob", "prob_bonafide",
                                                    "logit", "cm_score", "pred_score")), None)
    if lab_col is None or sc_col is None:
        return {"status": "NEEDS_MAPPING", "columns": cols,
                "reason": "Could not auto-detect label/score columns; tell Claude which is which."}
    raw = [r[lab_col].strip().lower() for r in rows]
    y = np.array([1 if v in ("1", "bonafide", "bona-fide", "real") else 0 for v in raw])
    s = np.array([float(r[sc_col]) for r in rows])
    thr_fixed = 0.5 if (s.min() >= 0 and s.max() <= 1) else 0.0
    out = {"status": "DONE", "n": len(y), "n_bonafide": int(y.sum()), "label_col": lab_col, "score_col": sc_col,
           "score_range": [float(s.min()), float(s.max())], "fixed_threshold": thr_fixed, "directions": {}}
    for name, ss, t in [("higher=bonafide", s, thr_fixed), ("higher=spoof", -s, -thr_fixed)]:
        e, et = eer(y, ss)
        out["directions"][name] = {"EER": e, "AUC": roc_auc(y, ss), "at_fixed_threshold": threshold_rates(y, ss, t)}
    out["note"] = ("Compare FPR/FNR above with historical 1.80% / 27.60%. If 27.60% matches FNR (spoof missed) "
                   "at the fixed threshold, it is a threshold/calibration effect, not an EER error.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "results/raw_predictions_Model_A_orig.csv"))
    a = ap.parse_args()
    rep = {"environment": env_report(), "inventory": inventory(),
           "locked_manifest": manifest_metadata(str(ROOT / "data/manifests/evaluation_manifest.csv")),
           "historical_audit": historical_audit(Path(a.pred))}
    out = ROOT / "results/phase0_report.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
