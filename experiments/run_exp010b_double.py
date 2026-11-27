"""EXP-010b (backup model, NOT deployed): continue training a COPY of the EXP-010 model on a small,
deliberately asymmetric set of double-compressed clips (same codec + bitrate applied twice: 64k, 32k, 16k),
then compare it with EXP-010 on identical tests. EXP-010 itself is never modified.

Train: 1,500 AI + 1,000 real clips drawn from the EXP-010 training manifest (60/40, no test data),
       each with one double-compressed copy (mp3/opus/aac x 64k/32k/16k, applied twice).
Tests (both models, same files): In-the-Wild 2,000 clean + double 64k/32k/16k; MLAAD held-out systems clean +
       double 16k; ASVspoof 2019 eval 1,000 clean + MP3 16k. McNemar test EXP-010 vs EXP-010b per condition.
Writes models/exp010b_wavlm_ft.pt, results/exp010b.json, results/exp010b_log.txt
"""
import csv, hashlib, json, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.data.audio_processing import N_SAMPLES, SR, load_audio, peak_normalize  # noqa: E402
from src.data.offline_cache import degrade_file, cache_path  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402
from src.models.wavlm_detector import WavLMDetector  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = ROOT / "results" / "exp010b_log.txt"
CACHE = ROOT / "data" / "cache" / "exp010b"
N_FAKE, N_REAL, EPOCHS, BATCH, LR_BB, LR_HEAD = 1500, 1000, 2, 12, 5e-6, 1e-4
CODECS = ["mp3", "opus", "aac"]
BRS = ["64k", "32k", "16k"]
rng = random.Random(10)


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


def double(src, codec, br):
    dst = CACHE / f"x2_{codec}_{br}" / (hashlib.md5(str(src).encode()).hexdigest()[:16] + ".wav")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".pass1.wav")
        try:
            degrade_file(str(src), tmp, codec, br)
            degrade_file(str(tmp), dst, codec, br)
        finally:
            tmp.unlink(missing_ok=True)
    return dst


def encode_all(jobs, what):
    log(f"encoding {len(jobs)} double-compressed copies ({what})")
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda j: double(*j), jobs))
    return out


def decode(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def crop(x, r):
    if len(x) < N_SAMPLES:
        x = np.pad(x, (0, N_SAMPLES - len(x)), mode="reflect" if len(x) > N_SAMPLES // 2 else "constant")
    s = r.integers(0, len(x) - N_SAMPLES + 1)
    return peak_normalize(x[s:s + N_SAMPLES].astype(np.float32))


class Train(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r, g = self.rows[i], np.random.default_rng()
        p = r["x2"] if g.random() < 0.7 else r["path"]          # 70% double-compressed, 30% original
        return torch.from_numpy(crop(load_audio(p), g)), int(r["label"])


@torch.no_grad()
def score_file(model, x):
    from src.app.server import windows
    ws = [peak_normalize(w.astype(np.float32)) for _, _, w in windows(x)]
    out = []
    for i in range(0, len(ws), 16):
        t = torch.from_numpy(np.stack(ws[i:i + 16])).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            lg = model(t).float()
        out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    return float(np.median(np.concatenate(out)))


def load(path):
    ck = torch.load(path, map_location="cpu")
    m = WavLMDetector(**ck.get("arch", {})); m.load_state_dict(ck["state_dict"])
    return m.to(DEV).eval(), ck


def mcnemar(a, b):
    from math import comb
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum()); n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
    return n10, n01, p


def main():
    LOG.write_text("")
    tr = list(csv.DictReader(open(ROOT / "data/manifests/exp010_train.csv")))
    va = list(csv.DictReader(open(ROOT / "data/manifests/exp010_val.csv")))
    fake = rng.sample([r for r in tr if r["label"] == "0"], N_FAKE)
    real = rng.sample([r for r in tr if r["label"] == "1"], N_REAL)
    rows = fake + real
    for r in rows:
        r["codec"], r["br"] = rng.choice(CODECS), rng.choice(BRS)
    vrows = rng.sample([r for r in va if r["label"] == "0"], 300) + rng.sample([r for r in va if r["label"] == "1"], 300)
    for r in vrows:
        r["codec"], r["br"] = rng.choice(CODECS), rng.choice(BRS)
    xs = encode_all([(r["path"], r["codec"], r["br"]) for r in rows + vrows], "train + calibration set")
    for r, x in zip(rows + vrows, xs):
        r["x2"] = str(x)
    log(f"train: {N_FAKE} AI + {N_REAL} real (asymmetric 60/40); bitrates " +
        ", ".join(f"{b}: {sum(r['br'] == b for r in rows)}" for b in BRS))

    # ---- clone EXP-010 and continue training the copy
    model, ck = load(ROOT / "models/exp010_wavlm_ft.pt")
    for p in model.wavlm.parameters():
        p.requires_grad = False
    for l in model.wavlm.encoder.layers[-ck["arch"].get("n_trainable", 8):]:
        for p in l.parameters():
            p.requires_grad = True
    opt = torch.optim.AdamW(model.param_groups(LR_BB, LR_HEAD), weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    dl = DataLoader(Train(rows), BATCH, shuffle=True, num_workers=4, drop_last=True)
    lossf = nn.CrossEntropyLoss()
    for ep in range(1, EPOCHS + 1):
        model.train(); model.wavlm.eval(); tot, t0 = 0.0, time.time()
        for i, (x, y) in enumerate(dl, 1):
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = lossf(model(x.to(DEV)).float(), y.to(DEV))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 5.0)
            scaler.step(opt); scaler.update(); tot += loss.item()
        log(f"epoch {ep}: loss {tot / len(dl):.4f}  ({(time.time() - t0) / 60:.1f} min)")
    model.eval()

    # ---- calibrate the clone on clean + double-compressed validation clips (EXP-010 keeps its own calibration)
    s, y = [], []
    for r in vrows:
        for p in (r["path"], r["x2"]):
            s.append(score_file(model, load_audio(p))); y.append(int(r["label"]))
    s, y = np.array(s), np.array(y)
    cal = PlattCalibrator().fit(s, y); pv = cal.predict_proba(s)
    ts = np.unique(pv); fa = np.array([(pv[y == 1] < t).mean() for t in ts]); ms = np.array([(pv[y == 0] >= t).mean() for t in ts])
    thr = float(ts[np.argmin(np.abs(fa - ms))])
    out = ROOT / "models/exp010b_wavlm_ft.pt"
    torch.save({"state_dict": model.state_dict(), "platt": (cal.a, cal.b), "threshold": thr, "threshold_strict": thr,
                "version": "EXP010B", "arch": ck["arch"], "note": "backup for heavily re-compressed audio; not deployed"}, out)
    log(f"saved {out.name}  (threshold P(real) >= {thr:.3f})")
    del model; torch.cuda.empty_cache()

    # ---- identical tests for both models
    e7 = list(csv.DictReader(open(ROOT / "results/exp007_files.csv")))
    meta = next((ROOT / "data/raw/in_the_wild").rglob("meta.csv")).parent
    itw = [(meta / r["file"], int(r["label"])) for r in e7]
    hm = [(Path(r["path"]), int(r["label"])) for r in csv.DictReader(open(ROOT / "data/manifests/exp010_test_mlaad_heldout.csv"))]
    import yaml
    from run_phase2_pilot import sample_split
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    a19 = sample_split(ev, {"n_train": 0, "n_val": 0, "n_test": 500}, 7)["test"]

    conds = {"In-the-Wild clean": itw}
    for b in BRS:
        files = [(p, l) for p, l in itw]
        enc = encode_all([(p, CODECS[i % 3], b) for i, (p, _) in enumerate(files)], f"In-the-Wild double {b}")
        conds[f"In-the-Wild double {b}"] = [(e, l) for e, (_, l) in zip(enc, files)]
    conds["MLAAD held-out clean"] = hm
    enc = encode_all([(p, CODECS[i % 3], "16k") for i, (p, _) in enumerate(hm)], "MLAAD held-out double 16k")
    conds["MLAAD held-out double 16k"] = [(e, l) for e, (_, l) in zip(enc, hm)]
    conds["ASVspoof 2019 clean"] = [(Path(r["path"]), int(r["label"])) for r in a19]
    conds["ASVspoof 2019 MP3 16k"] = [(cache_path(str(ROOT / "data/cache"), "mp3", "16k", r["file_id"]), int(r["label"])) for r in a19]

    res = {"train": {"ai": N_FAKE, "real": N_REAL}, "threshold_010b": thr, "conditions": {}}
    preds = {}
    for tag, path in (("EXP-010", ROOT / "models/exp010_wavlm_ft.pt"), ("EXP-010b", out)):
        model, c = load(path)
        a, b = c["platt"]; th = c["threshold"]
        log(f"Testing {tag}")
        for name, files in conds.items():
            S = np.array([score_file(model, decode(p) if "In-the-Wild" in name and "double" not in name else load_audio(str(p))) for p, _ in files])
            Y = np.array([l for _, l in files]); P = 1 / (1 + np.exp(-(a * S + b)))
            ok = (P >= th) == (Y == 1); preds[(tag, name)] = ok
            r = {"acc": float(ok.mean()), "false_alarm": float((P[Y == 1] < th).mean()), "miss": float((P[Y == 0] >= th).mean()),
                 "auc": roc_auc(Y, S), "eer": eer(Y, S)[0]}
            res["conditions"].setdefault(name, {})[tag] = r
            log(f"  {name:28} acc {r['acc']*100:5.1f}%  FA {r['false_alarm']*100:5.1f}%  miss {r['miss']*100:5.1f}%  EER {r['eer']*100:4.1f}%")
        del model; torch.cuda.empty_cache()
    log("McNemar EXP-010 vs EXP-010b (files only 010 got right / only 010b got right, p):")
    for name in conds:
        n10, n01, p = mcnemar(preds[("EXP-010", name)], preds[("EXP-010b", name)])
        res["conditions"][name]["mcnemar"] = {"only_010": n10, "only_010b": n01, "p": p}
        log(f"  {name:28} {n10:4d} / {n01:4d}   p = {p:.3g}")
    json.dump(res, open(ROOT / "results/exp010b.json", "w"), indent=2)
    log("EXP010B_DONE")


if __name__ == "__main__":
    main()
