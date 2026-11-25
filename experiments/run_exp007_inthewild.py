"""EXP-007: "before" baseline — the deployed model (method F, exact app pipeline) on In-the-Wild.

In-the-Wild (Müller et al. 2022): real + deepfake speech of 58 public figures from online videos. Never used
for training or calibration (test only). Balanced random sample of N_PER_CLASS real + N_PER_CLASS fake.
Scored with app profiles 'clean' and 'unknown', as-is and with leading/trailing silence trimmed.
Writes results/exp007.json and results/exp007_files.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_exp007a_diagnose import decode, score, trim_silence  # noqa: E402
from src.app import server  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402

N_PER_CLASS = 1000
PROFILES = ["clean", "unknown"]


def main():
    metas = list((ROOT / "data/raw/in_the_wild").rglob("meta.csv"))
    if not metas:
        sys.exit("In-the-Wild meta.csv not found - run download_datasets.bat first")
    meta = metas[0]
    rows = list(csv.DictReader(open(meta, encoding="utf-8")))
    lab = lambda r: 1 if r["label"].strip().lower().startswith("bona") else 0
    rng = np.random.default_rng(2026)
    real = [r for r in rows if lab(r) == 1]; fake = [r for r in rows if lab(r) == 0]
    print(f"In-the-Wild: {len(real)} real, {len(fake)} fake clips, {len({r['speaker'] for r in rows})} speakers", flush=True)
    pick = list(rng.choice(real, N_PER_CLASS, replace=False)) + list(rng.choice(fake, N_PER_CLASS, replace=False))
    server._load()

    recs = []
    for i, r in enumerate(pick, 1):
        x = decode(meta.parent / r["file"])
        if len(x) < 1600:
            continue
        rec = {"file": r["file"], "speaker": r["speaker"], "label": lab(r), "sec": round(len(x) / 16000, 2)}
        xt = trim_silence(x)
        for c in PROFILES:
            p, _ = score(x, c)
            rec[f"p_{c}"] = float(np.median(p))
        p, _ = score(xt, "clean")
        rec["p_clean_trimmed"] = float(np.median(p))
        recs.append(rec)
        if i % 200 == 0:
            print(f"  {i}/{len(pick)} scored", flush=True)

    y = np.array([r["label"] for r in recs])
    summary = {}
    for k in [f"p_{c}" for c in PROFILES] + ["p_clean_trimmed"]:
        p = np.array([r[k] for r in recs])
        summary[k] = {"accuracy": float(((p >= .5) == y).mean()), "false_alarm_real_called_AI": float((p[y == 1] < .5).mean()),
                      "miss_AI_called_real": float((p[y == 0] >= .5).mean()), "auc": roc_auc(y, p), "eer": eer(y, p)[0]}
        s = summary[k]
        print(f"  {k:18} acc {s['accuracy']*100:5.1f}%  false alarms {s['false_alarm_real_called_AI']*100:5.1f}%  "
              f"misses {s['miss_AI_called_real']*100:5.1f}%  AUC {s['auc']:.3f}  EER {s['eer']*100:4.1f}%", flush=True)
    secs = np.array([r["sec"] for r in recs])
    summary["duration_sec"] = {"mean": float(secs.mean()), "median": float(np.median(secs)), "max": float(secs.max())}
    summary["n"] = {"real": int(y.sum()), "fake": int((1 - y).sum())}
    json.dump(summary, open(ROOT / "results/exp007.json", "w"), indent=2)
    with open(ROOT / "results/exp007_files.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    print("EXP007_DONE", flush=True)


if __name__ == "__main__":
    main()
