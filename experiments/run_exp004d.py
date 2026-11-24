"""EXP-004d: small tweaks for 16 kbps robustness. Compared on measured results only.

Test  : 1,000 unseen-attack files (2019 LA eval A07-A18 w/o A16/A19; 500 real / 500 fake).
Train : 2019 LA train (A01-A06), 300+100 per class (train/val).
Conditions: clean; single MP3 64/32/16/8k; re-compressed 64k -> 64/32/16/8k.
Variants (all on frozen WavLM-Base+, all-layer mean+std):
  BASE : probe trained on 32k, one Platt calibrator (val 32k)            [= EXP-004c setup]
  A    : BASE probe + per-bitrate calibrator chosen by the final bitrate  (user dropdown)
  B    : probe trained on mixed clean+64k+32k+16k, one Platt (mixed val)
  C    : BASE + parallel band-limited branch (4 kHz low-pass -> WavLM), features concatenated
Writes results/exp004d.json
"""
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_phase2_pilot import sample_split  # noqa: E402
from run_ssl_probe import fit_probe, wavlm_feats  # noqa: E402
from src.data.audio_processing import load_audio, preprocess  # noqa: E402
from src.data.offline_cache import cache_path, degrade_file  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402

N_TEST = 500  # per class
BRS = ["64k", "32k", "16k", "8k"]
SOS = butter(8, 4000, btype="low", fs=16000, output="sos")


def main():
    from transformers import WavLMModel
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    cr, dev = str(ROOT / cfg["cache_root"]), ("cuda" if torch.cuda.is_available() else "cpu")
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    test = sample_split(ev, {"n_train": 0, "n_val": 0, "n_test": N_TEST}, 7)["test"]
    s = sample_split(list(csv.DictReader(open(ROOT / "data/manifests/asvspoof2019_train.csv"))),
                     {"n_train": 300, "n_val": 100, "n_test": 0}, cfg["seed"])
    tr, va = s["train"], s["val"]
    lab = lambda rows: np.array([int(r["label"]) for r in rows])
    ytr, yva, yte = lab(tr), lab(va), lab(test)

    single = lambda r, br: r["path"] if br is None else str(cache_path(cr, "mp3", br, r["file_id"]))
    double = lambda r, br: str(cache_path(cr, "mp3x2from64k", br, r["file_id"]))
    jobs = [(r["path"], single(r, b), b) for b in BRS for r in tr + va + test if not Path(single(r, b)).exists()]
    print(f"[1/4] single encodes needed: {len(jobs)}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: degrade_file(j[0], Path(j[1]), "mp3", j[2]), jobs))
    jobs = [(single(r, "64k"), double(r, b), b) for b in BRS for r in test if not Path(double(r, b)).exists()]
    print(f"[2/4] re-compression encodes needed: {len(jobs)}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: degrade_file(j[0], Path(j[1]), "mp3", j[2]), jobs))

    wl = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(dev).eval()

    def feats(rows, fn, br, lowpass=False):
        ws = []
        for r in rows:
            w = preprocess(load_audio(fn(r, br)))
            if lowpass:
                w = sosfiltfilt(SOS, w).astype(np.float32)
            ws.append(w)
        return wavlm_feats(wl, False, ws, dev).mean(1)

    print("[3/4] extracting train/val features", flush=True)
    F = {}
    for b in [None, "64k", "32k", "16k"]:
        F["tr", b] = feats(tr, single, b)
    for b in [None] + BRS:
        F["va", b] = feats(va, single, b)
    F["trLP"], F["vaLP"] = feats(tr, single, "32k", True), feats(va, single, "32k", True)

    _, base = fit_probe(F["tr", "32k"], ytr, F["va", "32k"], yva)
    cal_base = PlattCalibrator().fit(base.decision_function(F["va", "32k"]), yva)
    cal_A = {b: PlattCalibrator().fit(base.decision_function(F["va", b]), yva) for b in [None] + BRS}
    Xb = np.vstack([F["tr", b] for b in [None, "64k", "32k", "16k"]]); yb = np.tile(ytr, 4)
    Vb = np.vstack([F["va", b] for b in [None, "64k", "32k", "16k"]]); vb = np.tile(yva, 4)
    _, mB = fit_probe(Xb, yb, Vb, vb)
    cal_B = PlattCalibrator().fit(mB.decision_function(Vb), vb)
    _, mC = fit_probe(np.hstack([F["tr", "32k"], F["trLP"]]), ytr, np.hstack([F["va", "32k"], F["vaLP"]]), yva)
    cal_C = PlattCalibrator().fit(mC.decision_function(np.hstack([F["va", "32k"], F["vaLP"]])), yva)

    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(yte), len(yte)) for _ in range(1000)]

    def metrics(p):
        acc = ((p >= 0.5) == yte)
        bs = [acc[i].mean() for i in boot]
        return {"acc": float(acc.mean()), "acc_lo": float(np.quantile(bs, .025)), "acc_hi": float(np.quantile(bs, .975)),
                "auc": roc_auc(yte, p), "eer": eer(yte, p)[0], "p_real_on_real": float(p[yte == 1].mean()),
                "p_real_on_fake": float(p[yte == 0].mean()), "false_alarm_rate": float((p[yte == 1] < 0.5).mean()),
                "miss_rate": float((p[yte == 0] >= 0.5).mean())}

    print("[4/4] scoring 1,000 test files per condition", flush=True)
    out = []
    conds = [("clean", single, None)] + [(f"single_{b}", single, b) for b in BRS] + [(f"double_{b}", double, b) for b in BRS]
    for name, fn, b in conds:
        X, XL = feats(test, fn, b), feats(test, fn, b, True)
        sb = base.decision_function(X)
        res = {"BASE": cal_base.predict_proba(sb), "A": cal_A[b].predict_proba(sb),
               "B": cal_B.predict_proba(mB.decision_function(X)),
               "C": cal_C.predict_proba(mC.decision_function(np.hstack([X, XL])))}
        for v, p in res.items():
            out.append({"variant": v, "condition": name, **metrics(p)})
            m = out[-1]
            print(f"  {name:10} {v:4}  acc {m['acc']*100:5.1f}% [{m['acc_lo']*100:4.1f}-{m['acc_hi']*100:4.1f}]  "
                  f"AUC {m['auc']:.4f}  false-alarm {m['false_alarm_rate']*100:4.1f}%  miss {m['miss_rate']*100:4.1f}%", flush=True)
    json.dump(out, open(ROOT / "results/exp004d.json", "w"), indent=2)
    print("EXP004D_DONE", flush=True)


if __name__ == "__main__":
    main()
