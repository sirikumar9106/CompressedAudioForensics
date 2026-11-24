"""EXP-006: real-world robustness of the EXP-005 design (F = RawBoost-aug WavLM probe + per-bitrate calibration).

Part 1  McNemar paired test, F vs A, on the same 1,000 files (all 9 EXP-005 conditions).
Part 2  Wrong dropdown: MP3 audio at true bitrate T scored with the calibrator for chosen bitrate S (+ "Not sure").
Part 3  Unseen codecs: Opus 64/32/16/8k and AAC 64/32/16k (never used for training or calibration);
        user picks the matching nominal bitrate from the MP3-calibrated dropdown, or "Not sure".
Reuses results/exp005_feats.npz. Writes results/exp006.json
"""
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_phase2_pilot import sample_split  # noqa: E402
from run_ssl_probe import fit_probe, wavlm_feats  # noqa: E402
from src.data.audio_processing import load_audio, preprocess  # noqa: E402
from src.data.offline_cache import cache_path, degrade_file  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import roc_auc  # noqa: E402

BRS = [None, "64k", "32k", "16k", "8k"]
NEW = [("opus", b) for b in ["64k", "32k", "16k", "8k"]] + [("aac", b) for b in ["64k", "32k", "16k"]]


def main():
    F = dict(np.load(ROOT / "results/exp005_feats.npz", allow_pickle=True))
    ytr, yva, yte = F["ytr"], F["yva"], F["yte"]
    tr = lambda b: F[f"tr_{b}"]; va = lambda b: F[f"va_{b}"]

    _, mA = fit_probe(tr("32k"), ytr, va("32k"), yva)
    _, mF = fit_probe(np.vstack([tr("32k"), F["tr_32k_aug"]]), np.tile(ytr, 2), va("32k"), yva)
    mix = [None, "64k", "32k", "16k"]
    _, mU = fit_probe(np.vstack([tr(b) for b in mix]), np.tile(ytr, 4), np.vstack([va(b) for b in mix]), np.tile(yva, 4))
    calA = {b: PlattCalibrator().fit(mA.decision_function(va(b)), yva) for b in BRS}
    calF = {b: PlattCalibrator().fit(mF.decision_function(va(b)), yva) for b in BRS}
    calU = PlattCalibrator().fit(mU.decision_function(np.vstack([va(b) for b in BRS])), np.tile(yva, 5))
    out = {"mcnemar": [], "mismatch": [], "codecs": []}

    print("[1/3] McNemar F vs A", flush=True)
    conds = ["clean"] + [f"{m}_{b}" for m in ("single", "double") for b in ("64k", "32k", "16k", "8k")]
    for c in conds:
        b = None if c == "clean" else c.split("_")[1]
        X = F[f"te_{c}"]
        okA = (calA[b].predict_proba(mA.decision_function(X)) >= 0.5) == yte
        okF = (calF[b].predict_proba(mF.decision_function(X)) >= 0.5) == yte
        n01, n10 = int((~okA & okF).sum()), int((okA & ~okF).sum())  # F right & A wrong / A right & F wrong
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        out["mcnemar"].append({"condition": c, "acc_A": float(okA.mean()), "acc_F": float(okF.mean()),
                               "F_only_correct": n01, "A_only_correct": n10, "p_value": float(p)})
        print(f"  {c:10} A {okA.mean()*100:5.1f}  F {okF.mean()*100:5.1f}  F-only {n01:3d}  A-only {n10:3d}  p={p:.4f}", flush=True)

    print("[2/3] wrong dropdown choice (F)", flush=True)
    choices = BRS + ["notsure"]
    for t in ["64k", "32k", "16k", "8k"]:
        X = F[f"te_single_{t}"]
        for s in choices:
            p = calU.predict_proba(mU.decision_function(X)) if s == "notsure" else calF[s].predict_proba(mF.decision_function(X))
            out["mismatch"].append({"true": t, "chosen": "clean" if s is None else s, "acc": float(((p >= 0.5) == yte).mean()),
                                    "false_alarm": float((p[yte == 1] < 0.5).mean()), "miss": float((p[yte == 0] >= 0.5).mean())})
        print("  true", t, " ".join(f"{r['chosen']}:{r['acc']*100:.1f}" for r in out["mismatch"][-6:]), flush=True)

    print("[3/3] unseen codecs: Opus / AAC", flush=True)
    from transformers import WavLMModel
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    cr, dev = str(ROOT / cfg["cache_root"]), ("cuda" if torch.cuda.is_available() else "cpu")
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    test = sample_split(ev, {"n_train": 0, "n_val": 0, "n_test": 500}, 7)["test"]
    assert (np.array([int(r["label"]) for r in test]) == yte).all()
    jobs = [(r, c, b) for c, b in NEW for r in test if not cache_path(cr, c, b, r["file_id"]).exists()]
    print(f"  encoding {len(jobs)} files", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: degrade_file(j[0]["path"], cache_path(cr, j[1], j[2], j[0]["file_id"]), j[1], j[2]), jobs))
    wl = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval()
    for c, b in NEW:
        X = wavlm_feats(wl, False, [preprocess(load_audio(str(cache_path(cr, c, b, r["file_id"])))) for r in test], dev).mean(1)
        for mode, p in [("F + matching dropdown", calF[b].predict_proba(mF.decision_function(X))),
                        ("A + matching dropdown", calA[b].predict_proba(mA.decision_function(X))),
                        ("Not sure", calU.predict_proba(mU.decision_function(X)))]:
            out["codecs"].append({"codec": c, "bitrate": b, "mode": mode, "acc": float(((p >= 0.5) == yte).mean()),
                                  "auc": roc_auc(yte, p), "false_alarm": float((p[yte == 1] < 0.5).mean()),
                                  "miss": float((p[yte == 0] >= 0.5).mean())})
            r = out["codecs"][-1]
            print(f"  {c} {b:4} {mode:22} acc {r['acc']*100:5.1f}  AUC {r['auc']:.4f}  FA {r['false_alarm']*100:4.1f}  miss {r['miss']*100:4.1f}", flush=True)
    json.dump(out, open(ROOT / "results/exp006.json", "w"), indent=2)
    print("EXP006_DONE", flush=True)


if __name__ == "__main__":
    main()
