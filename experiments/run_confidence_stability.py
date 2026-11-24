"""EXP-004b: does a file's confidence stay stable as compression increases?

Same 200 unseen-attack test files scored at clean, MP3 320/128/64/32/16/8 kbps.
Detectors (each Platt-calibrated on its own 2019 LA train-split validation set):
  - Model A (official RawNet2)                      calibrated on clean val
  - WavLM-Base+ layer-avg probe trained on CLEAN    calibrated on clean val
  - WavLM-Base+ layer-avg probe trained on MP3 32k  calibrated on 32k val
Writes results/confidence_stability.csv (per file x condition) and results/confidence_stability.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_phase2_pilot import sample_split  # noqa: E402
from run_ssl_probe import fit_probe, wavlm_feats  # noqa: E402
from src.data.audio_processing import load_audio, preprocess  # noqa: E402
from src.data.offline_cache import cache_path  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402
from src.models.backbones import load_pretrained_rawnet2  # noqa: E402

CONDS = [None, "320k", "128k", "64k", "32k", "16k", "8k"]
MODEL = "microsoft/wavlm-base-plus"


def main():
    from transformers import WavLMModel
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    cr, dev = str(ROOT / cfg["cache_root"]), ("cuda" if torch.cuda.is_available() else "cpu")
    P = lambda r, br: r["path"] if br is None else str(cache_path(cr, "mp3", br, r["file_id"]))
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    test = sample_split(ev, cfg, cfg["seed"])["test"]
    s = sample_split(list(csv.DictReader(open(ROOT / "data/manifests/asvspoof2019_train.csv"))),
                     {"n_train": 300, "n_val": 100, "n_test": 0}, cfg["seed"])
    tr, va = s["train"], s["val"]
    ytr, yva = [np.array([int(r["label"]) for r in x]) for x in (tr, va)]
    yte = np.array([int(r["label"]) for r in test])

    rawnet = load_pretrained_rawnet2(str(ROOT / cfg["pretrained"])).to(dev).eval()
    wl = WavLMModel.from_pretrained(MODEL).to(dev).eval()
    W = lambda rows, br: wavlm_feats(wl, False, [preprocess(load_audio(P(r, br))) for r in rows], dev).mean(1)

    @torch.no_grad()
    def modelA(rows, br):
        out = []
        for r in rows:
            lg = rawnet(torch.from_numpy(preprocess(load_audio(P(r, br)))).unsqueeze(0).to(dev))
            out.append((lg[0, 1] - lg[0, 0]).item())
        return np.array(out)

    det = {}
    print("calibrating Model A", flush=True)
    det["Model A (official)"] = (None, PlattCalibrator().fit(modelA(va, None), yva))
    for tag, br in [("WavLM-B+ probe (train clean)", None), ("WavLM-B+ probe (train 32k)", "32k")]:
        print("training", tag, flush=True)
        _, m = fit_probe(W(tr, br), ytr, W(va, br), yva)
        det[tag] = (m, PlattCalibrator().fit(m.decision_function(W(va, br)), yva))

    probs = {k: {} for k in det}
    for br in CONDS:
        c = "clean" if br is None else br
        print("scoring", c, flush=True)
        F = W(test, br)
        sA = modelA(test, br)
        for k, (m, cal) in det.items():
            raw = sA if m is None else m.decision_function(F)
            probs[k][c] = cal.predict_proba(raw)

    rows, summ = [], []
    for k, d in probs.items():
        p0 = d["clean"]
        for c, p in d.items():
            summ.append({"detector": k, "condition": c, "auc": roc_auc(yte, p), "eer": eer(yte, p)[0],
                         "acc@0.5": float(((p >= 0.5) == yte).mean()),
                         "mean_abs_shift_vs_clean": float(np.abs(p - p0).mean()),
                         "p95_abs_shift_vs_clean": float(np.quantile(np.abs(p - p0), 0.95)),
                         "decision_flip_rate_vs_clean": float(((p >= 0.5) != (p0 >= 0.5)).mean()),
                         "mean_p_bonafide_on_bonafide": float(p[yte == 1].mean()),
                         "mean_p_bonafide_on_spoof": float(p[yte == 0].mean())})
            for r, pi in zip(test, p):
                rows.append({"detector": k, "condition": c, "file_id": r["file_id"], "label": r["label"],
                             "p_bonafide": round(float(pi), 5)})
            print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in summ[-1].items()}), flush=True)
    with open(ROOT / "results/confidence_stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(summ, open(ROOT / "results/confidence_stability.json", "w"), indent=2)
    print("STABILITY_DONE", flush=True)


if __name__ == "__main__":
    main()
