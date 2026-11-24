"""Deployment model (method F from EXP-005/006) — build once from cached features, then load.

F  = linear probe on frozen WavLM-Base+ all-layer mean+std features, trained on MP3-32k clips
     plus RawBoost-style augmented copies; Platt calibrator per bitrate class.
U  = "I don't know" mode: probe trained on clean+64+32+16k, one pooled calibrator.
Reference scores (validation set) are kept per calibrator for the explainability chart.
"""
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "models" / "compressed_forensics_F.joblib"
CLASSES = ["clean", "64k", "32k", "16k", "8k"]


def build(npz_path=ROOT / "results" / "exp005_feats.npz", out=ARTIFACT):
    sys.path.insert(0, str(ROOT / "experiments"))
    from run_ssl_probe import fit_probe
    from src.evaluation.calibrator import PlattCalibrator

    F = dict(np.load(npz_path, allow_pickle=True))
    ytr, yva = F["ytr"], F["yva"]
    key = lambda split, c: f"{split}_{None if c == 'clean' else c}"
    _, mF = fit_probe(np.vstack([F["tr_32k"], F["tr_32k_aug"]]), np.tile(ytr, 2), F["va_32k"], yva)
    mix = ["clean", "64k", "32k", "16k"]
    _, mU = fit_probe(np.vstack([F[key("tr", c)] for c in mix]), np.tile(ytr, 4),
                      np.vstack([F[key("va", c)] for c in mix]), np.tile(yva, 4))
    cal, ref = {}, {}
    for c in CLASSES:
        s = mF.decision_function(F[key("va", c)])
        p = PlattCalibrator().fit(s, yva)
        cal[c] = (p.a, p.b)
        ref[c] = {"real": s[yva == 1].astype(float), "fake": s[yva == 0].astype(float)}
    sU = mU.decision_function(np.vstack([F[key("va", c)] for c in CLASSES]))
    yU = np.tile(yva, len(CLASSES))
    pU = PlattCalibrator().fit(sU, yU)
    cal["unknown"] = (pU.a, pU.b)
    ref["unknown"] = {"real": sU[yU == 1].astype(float), "fake": sU[yU == 0].astype(float)}
    art = {"probe_F": mF, "probe_U": mU, "cal": cal, "ref": ref, "version": "F-1.0 (EXP-005/006)"}
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, out)
    return art


def load():
    return joblib.load(ARTIFACT) if ARTIFACT.exists() else build()


if __name__ == "__main__":
    build()
    print("saved", ARTIFACT)
