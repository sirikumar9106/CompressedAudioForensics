"""EXP-004c: re-compression (forwarding) robustness.

Same 200 unseen-attack test files. Each file is first encoded to MP3 64k (first send), decoded, then
re-encoded at 320/128/64/32/16/8k (forward). Scores compared against SINGLE compression at the same
bitrate (EXP-004b). Same detectors and calibration as EXP-004b.
Writes results/double_compression.json
"""
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
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
from src.data.offline_cache import cache_path, degrade_file  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402
from src.models.backbones import load_pretrained_rawnet2  # noqa: E402

FIRST = "64k"
SECOND = ["320k", "128k", "64k", "32k", "16k", "8k"]


def main():
    from transformers import WavLMModel
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    cr, dev = str(ROOT / cfg["cache_root"]), ("cuda" if torch.cuda.is_available() else "cpu")
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    test = sample_split(ev, cfg, cfg["seed"])["test"]
    s = sample_split(list(csv.DictReader(open(ROOT / "data/manifests/asvspoof2019_train.csv"))),
                     {"n_train": 300, "n_val": 100, "n_test": 0}, cfg["seed"])
    tr, va = s["train"], s["val"]
    ytr, yva = [np.array([int(r["label"]) for r in x]) for x in (tr, va)]
    yte = np.array([int(r["label"]) for r in test])

    single = lambda r, br: r["path"] if br is None else str(cache_path(cr, "mp3", br, r["file_id"]))
    double = lambda r, br: str(cache_path(cr, f"mp3x2from{FIRST}", br, r["file_id"]))
    jobs = [(r, br) for br in SECOND for r in test if not Path(double(r, br)).exists()]
    print(f"re-encoding {len(jobs)} files (64k -> X)", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: degrade_file(single(j[0], FIRST), Path(double(*j)), "mp3", j[1]), jobs))

    rawnet = load_pretrained_rawnet2(str(ROOT / cfg["pretrained"])).to(dev).eval()
    wl = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval()
    W = lambda rows, fn, br: wavlm_feats(wl, False, [preprocess(load_audio(fn(r, br))) for r in rows], dev).mean(1)

    @torch.no_grad()
    def modelA(rows, fn, br):
        return np.array([(lambda lg: (lg[0, 1] - lg[0, 0]).item())(
            rawnet(torch.from_numpy(preprocess(load_audio(fn(r, br)))).unsqueeze(0).to(dev))) for r in rows])

    det = {"Model A (official)": (None, PlattCalibrator().fit(modelA(va, single, None), yva))}
    for tag, br in [("WavLM-B+ probe (train clean)", None), ("WavLM-B+ probe (train 32k)", "32k")]:
        print("training", tag, flush=True)
        _, m = fit_probe(W(tr, single, br), ytr, W(va, single, br), yva)
        det[tag] = (m, PlattCalibrator().fit(m.decision_function(W(va, single, br)), yva))

    out = []
    for mode, fn in [("single", single), ("double", double)]:
        for br in SECOND:
            print("scoring", mode, br, flush=True)
            F, sA = W(test, fn, br), modelA(test, fn, br)
            for k, (m, cal) in det.items():
                p = cal.predict_proba(sA if m is None else m.decision_function(F))
                pr, pf = float(p[yte == 1].mean()), float(p[yte == 0].mean())
                out.append({"detector": k, "mode": mode, "bitrate": br, "auc": roc_auc(yte, p), "eer": eer(yte, p)[0],
                            "acc@0.5": float(((p >= 0.5) == yte).mean()), "mean_p_real_on_real": pr,
                            "mean_p_real_on_fake": pf, "confidence_gap": pr - pf})
                print(json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in out[-1].items()}), flush=True)
    json.dump(out, open(ROOT / "results/double_compression.json", "w"), indent=2)
    print("DOUBLE_DONE", flush=True)


if __name__ == "__main__":
    main()
