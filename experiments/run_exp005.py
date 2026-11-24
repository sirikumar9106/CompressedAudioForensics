"""EXP-005: is EXP-004d's winner (A) beaten by stronger alternatives? Same 1,000 test files & conditions.

All variants use frozen WavLM-Base+ all-layer mean+std features.
  A : linear probe trained on 32k + per-bitrate calibration                 (EXP-004d winner, reference)
  D : linear probe trained on clean+64+32+16k + per-bitrate calibration     (A + B combined)
  E : small MLP (1 hidden layer) trained on 32k + per-bitrate calibration
  F : linear probe trained on 32k + RawBoost-style augmented copies + per-bitrate calibration
  U : "Not sure" mode — D's probe with ONE pooled calibrator (bitrate unknown)
Writes results/exp005.json and caches features in results/exp005_feats.npz (reused by later experiments).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.signal import firwin, lfilter
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_phase2_pilot import sample_split  # noqa: E402
from run_ssl_probe import fit_probe, wavlm_feats  # noqa: E402
from src.data.audio_processing import load_audio, peak_normalize, preprocess  # noqa: E402
from src.data.offline_cache import cache_path  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402

BRS = ["64k", "32k", "16k", "8k"]
RNG = np.random.default_rng(42)


def rawboost_lite(w):
    """RawBoost-style nuisance: random band-pass coloration (linear convolutive) + stationary noise (10-35 dB SNR)."""
    lo = RNG.uniform(50, 400); hi = RNG.uniform(3000, 7900)
    h = firwin(RNG.integers(31, 129) | 1, [lo, hi], pass_zero=False, fs=16000)
    y = lfilter(h, 1.0, w)
    snr = RNG.uniform(10, 35)
    y = y + RNG.standard_normal(len(y)) * np.sqrt(np.mean(y ** 2) / 10 ** (snr / 10))
    return peak_normalize(y.astype(np.float32))


def fit_mlp(Xtr, ytr, Xva, yva):
    best = (-1, None)
    for a in [1e-4, 1e-3, 1e-2]:
        m = make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=a, max_iter=300, early_stopping=True,
                                                          random_state=0)).fit(Xtr, ytr)
        s = m.predict_proba(Xva)[:, 1]
        auc = roc_auc(yva, s)
        if auc > best[0]:
            best = (auc, m)
    m = best[1]
    m.decision_function = lambda X, m=m: np.log(np.clip(m.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)) - \
        np.log(np.clip(m.predict_proba(X)[:, 0], 1e-7, 1 - 1e-7))
    return m


def main():
    from transformers import WavLMModel
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    cr, dev = str(ROOT / cfg["cache_root"]), ("cuda" if torch.cuda.is_available() else "cpu")
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    test = sample_split(ev, {"n_train": 0, "n_val": 0, "n_test": 500}, 7)["test"]  # same 1,000 as EXP-004d
    s = sample_split(list(csv.DictReader(open(ROOT / "data/manifests/asvspoof2019_train.csv"))),
                     {"n_train": 300, "n_val": 100, "n_test": 0}, cfg["seed"])
    tr, va = s["train"], s["val"]
    lab = lambda rows: np.array([int(r["label"]) for r in rows])
    ytr, yva, yte = lab(tr), lab(va), lab(test)
    single = lambda r, br: r["path"] if br is None else str(cache_path(cr, "mp3", br, r["file_id"]))
    double = lambda r, br: str(cache_path(cr, "mp3x2from64k", br, r["file_id"]))
    wl = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval()
    feats = lambda rows, fn, br, aug=False: wavlm_feats(
        wl, False, [(rawboost_lite if aug else (lambda w: w))(preprocess(load_audio(fn(r, br)))) for r in rows], dev).mean(1)

    print("[1/3] train/val features", flush=True)
    F = {}
    for b in [None, "64k", "32k", "16k"]:
        F[f"tr_{b}"] = feats(tr, single, b)
    for b in [None] + BRS:
        F[f"va_{b}"] = feats(va, single, b)
    F["tr_32k_aug"] = feats(tr, single, "32k", aug=True)
    print("[2/3] test features (9 conditions x 1,000 files)", flush=True)
    conds = [("clean", single, None)] + [(f"single_{b}", single, b) for b in BRS] + [(f"double_{b}", double, b) for b in BRS]
    for name, fn, b in conds:
        F[f"te_{name}"] = feats(test, fn, b)
        print(f"   {name} done", flush=True)
    np.savez_compressed(ROOT / "results/exp005_feats.npz", **F, ytr=ytr, yva=yva, yte=yte)

    mix = [None, "64k", "32k", "16k"]
    _, mA = fit_probe(F["tr_32k"], ytr, F["va_32k"], yva)
    _, mD = fit_probe(np.vstack([F[f"tr_{b}"] for b in mix]), np.tile(ytr, 4),
                      np.vstack([F[f"va_{b}"] for b in mix]), np.tile(yva, 4))
    mE = fit_mlp(F["tr_32k"], ytr, F["va_32k"], yva)
    _, mF = fit_probe(np.vstack([F["tr_32k"], F["tr_32k_aug"]]), np.tile(ytr, 2), F["va_32k"], yva)
    per_br = lambda m: {b: PlattCalibrator().fit(m.decision_function(F[f"va_{b}"]), yva) for b in [None] + BRS}
    cal = {"A": per_br(mA), "D": per_br(mD), "E": per_br(mE), "F": per_br(mF)}
    model = {"A": mA, "D": mD, "E": mE, "F": mF, "U": mD}
    pooled = PlattCalibrator().fit(mD.decision_function(np.vstack([F[f"va_{b}"] for b in [None] + BRS])),
                                   np.tile(yva, 5))

    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(yte), len(yte)) for _ in range(1000)]
    print("[3/3] scoring", flush=True)
    out = []
    for name, _, b in conds:
        X = F[f"te_{name}"]
        for v, m in model.items():
            p = (pooled if v == "U" else cal[v][b]).predict_proba(m.decision_function(X))
            acc = (p >= 0.5) == yte
            bs = [acc[i].mean() for i in boot]
            out.append({"variant": v, "condition": name, "acc": float(acc.mean()),
                        "acc_lo": float(np.quantile(bs, .025)), "acc_hi": float(np.quantile(bs, .975)),
                        "auc": roc_auc(yte, p), "eer": eer(yte, p)[0],
                        "false_alarm_rate": float((p[yte == 1] < 0.5).mean()), "miss_rate": float((p[yte == 0] >= 0.5).mean())})
            r = out[-1]
            print(f"  {name:10} {v}  acc {r['acc']*100:5.1f}% [{r['acc_lo']*100:4.1f}-{r['acc_hi']*100:4.1f}]  AUC {r['auc']:.4f}  "
                  f"false-alarm {r['false_alarm_rate']*100:4.1f}%  miss {r['miss_rate']*100:4.1f}%", flush=True)
    json.dump(out, open(ROOT / "results/exp005.json", "w"), indent=2)
    print("EXP005_DONE", flush=True)


if __name__ == "__main__":
    main()
