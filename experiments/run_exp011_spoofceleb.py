"""EXP-011 (test only, no training): deployed EXP-010 on SpoofCeleb evaluation attack a23.

The SpoofCeleb stream (download_spoofceleb_eval.py) stopped after saving evaluation attack a23 (6,792 AI clips)
and before the evaluation real voices. Real side: 2,000 In-the-Wild real clips (celebrities from online videos,
the closest real-speech match to SpoofCeleb's VoxCeleb source; random, seed 11). Neither set was ever trained on.
Conditions: original audio, and Opus 16 kbps (WhatsApp-like). Writes results/exp011.json, results/exp011_log.txt.
"""
import csv, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import run_exp010b_double as base  # noqa: E402
from run_exp010b_double import load, score_file, decode  # noqa: E402
from src.data.offline_cache import degrade_file  # noqa: E402
from src.data.audio_processing import load_audio  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402

base.LOG = ROOT / "results" / "exp011_log.txt"
log = base.log
A23 = ROOT / "data/raw/spoofceleb/spoofceleb/flac/evaluation/a23"
CACHE = ROOT / "data/cache/exp011_opus16k"
N_REAL = 2000


def opus16(p, tag):
    dst = CACHE / f"{tag}_{p.stem}.wav"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        degrade_file(str(p), dst, "opus", "16k")
    return dst


def main():
    base.LOG.write_text("")
    fake = sorted(A23.glob("*/*.flac"))
    meta = next((ROOT / "data/raw/in_the_wild").rglob("meta.csv"))
    rows = [r for r in csv.DictReader(open(meta, encoding="utf-8")) if r["label"].strip().lower().startswith("bona")]
    real = [meta.parent / r["file"] for r in random.Random(11).sample(rows, N_REAL)]
    log(f"AI: {len(fake)} SpoofCeleb evaluation clips (attack a23, {len({p.parent.name for p in fake})} speakers); "
        f"real: {len(real)} In-the-Wild clips")
    with ThreadPoolExecutor(8) as ex:
        fo = list(ex.map(lambda p: opus16(p, "sc"), fake))
        ro = list(ex.map(lambda p: opus16(p, "itw"), real))
    log("Opus 16 kbps copies ready")
    model, c = load(ROOT / "models/exp010_wavlm_ft.pt"); a, b = c["platt"]; th = c["threshold"]
    res = {"n_ai": len(fake), "n_real": len(real), "conditions": {}}
    for name, F, R in (("original", fake, real), ("Opus 16k", fo, ro)):
        Sf = np.array([score_file(model, load_audio(str(p))) for p in F])
        Sr = np.array([score_file(model, decode(p) if name == "original" else load_audio(str(p))) for p in R])
        S = np.concatenate([Sr, Sf]); Y = np.r_[np.ones(len(Sr)), np.zeros(len(Sf))]
        P = 1 / (1 + np.exp(-(a * S + b)))
        det, fa = float((P[Y == 0] < th).mean()), float((P[Y == 1] < th).mean())
        r = {"ai_detected": det, "miss": 1 - det, "false_alarm": fa,
             "balanced_acc": (det + 1 - fa) / 2, "auc": roc_auc(Y, S), "eer": eer(Y, S)[0],
             "ai_score_median": float(np.median(Sf)), "real_score_median": float(np.median(Sr))}
        res["conditions"][name] = r
        log(f"  {name:9}  AI detected {det*100:5.1f}%  real called AI {fa*100:4.1f}%  balanced acc "
            f"{r['balanced_acc']*100:5.1f}%  AUC {r['auc']:.3f}  EER {r['eer']*100:4.1f}%")
    json.dump(res, open(ROOT / "results/exp011.json", "w"), indent=2)
    log("EXP011_DONE")


if __name__ == "__main__":
    main()
