"""EXP-009b: compression-robust FEATURE SELECTION for the image CNN, then combination with WavLM (method F).

Step 1  Feature audit (no training): each of the 3 views (mel / fine spectrum / phase IF) is cut into 16 equal frequency slices
        (mel: mel-spaced 0-8 kHz; fine spectrum and phase: 250 Hz slices of 0-4 kHz). On 400 training clips, each band gets
          stability      = mean correlation between the clean and the 16k/8k-compressed version of the same clip
          discriminative = |AUC - 0.5| * 2 of the band's per-clip mean/std, measured on COMPRESSED audio
        weight = stability x discriminative. Only bands with weight >= 25% of the best band are kept, and each kept
        band is scaled by its weight, so the CNN sees mainly what survives compression AND separates real from fake.
Step 2  Retrain ResNet-18 on the selected/weighted image with clean + MP3 32k/16k/8k copies (same 4,960 clips).
Step 3  Combination with WavLM F on the SAME 1,000 test files x 9 conditions:
          - average of calibrated log-odds (no extra training)
          - per-bitrate logistic fusion fitted on the 200 validation clips
        McNemar paired test fused vs F alone.
Writes results/exp009b.json, models/exp009b_image_cnn.pt
"""
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import yaml
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_exp009a_image import FingerprintImage, build_cnn, crop, spec_augment  # noqa: E402
from run_phase2_pilot import sample_split  # noqa: E402
from src.app import model_store  # noqa: E402
from src.data.audio_processing import load_audio  # noqa: E402
from src.data.offline_cache import cache_path, degrade_file  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BRS = ["64k", "32k", "16k", "8k"]
EPOCHS, BATCH, LR, NB = 15, 32, 3e-4, 16
LOG = ROOT / "results" / "exp009b_log.txt"
random.seed(19); np.random.seed(19); torch.manual_seed(19)


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


class RawViews(FingerprintImage):
    """Same three views as EXP-009a but WITHOUT resizing/normalising: returns per-view [B, F, T] at 0-8 kHz."""
    @torch.no_grad()
    def views(self, x):
        img = super().forward(x)   # [B,3,128,400], frequency rows flipped (low at bottom) -> undo flip
        return img.flip(2)          # rows 0..127 = low..high frequency


class SelectedImage(nn.Module):
    """Keeps only selected bands of each view, scales them by their weight, re-stacks to 3 x 128 x 400."""
    def __init__(self, weights):   # weights [3, NB] (0 = dropped)
        super().__init__()
        self.base = RawViews()
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x):
        v = self.base.views(x)                                   # [B,3,128,400]
        rows = 128 // NB
        chans = []
        for c in range(3):
            keep = [b for b in range(NB) if self.w[c, b] > 0]
            if not keep:
                chans.append(torch.zeros_like(v[:, 0])); continue
            parts = [v[:, c, b * rows:(b + 1) * rows] * self.w[c, b] for b in keep]
            sel = torch.cat(parts, 1)                            # [B, rows*len(keep), 400]
            chans.append(Fn.interpolate(sel.unsqueeze(1), size=(128, 400), mode="bilinear", align_corners=False).squeeze(1))
        img = torch.stack(chans, 1).flip(2)
        mu = img.mean((2, 3), keepdim=True); sd = img.std((2, 3), keepdim=True) + 1e-5
        return (img - mu) / sd


def main():
    LOG.write_text("")
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    cr = str(ROOT / cfg["cache_root"])
    P = lambda r, br: r["path"] if br is None else str(cache_path(cr, "mp3", br, r["file_id"]))
    D = lambda r, br: str(cache_path(cr, "mp3x2from64k", br, r["file_id"]))
    tr_all = list(csv.DictReader(open(ROOT / "data/manifests/asvspoof2019_train.csv")))
    pv = sample_split(tr_all, {"n_train": 300, "n_val": 100, "n_test": 0}, cfg["seed"])
    val_ids = {r["file_id"] for r in pv["val"]}
    pool = [r for r in tr_all if r["file_id"] not in val_ids]
    rng = random.Random(9)
    real = [r for r in pool if r["label"] == "1"]; fake = rng.sample([r for r in pool if r["label"] == "0"], len(real))
    train = real + fake                                           # same 4,960 clips as EXP-009a
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    test = sample_split(ev, {"n_train": 0, "n_val": 0, "n_test": 500}, 7)["test"]
    yva = np.array([int(r["label"]) for r in pv["val"]]); yte = np.array([int(r["label"]) for r in test])

    jobs = [(r, b) for r in train for b in ("32k", "16k", "8k") if not Path(P(r, b)).exists()]
    log(f"encoding {len(jobs)} MP3 copies (32k/16k/8k) of the training clips")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: degrade_file(j[0]["path"], Path(P(*j)), "mp3", j[1]), jobs))

    # ---------------- Step 1: feature audit
    log("Step 1: feature audit on 400 training clips (stability under compression x real/fake separation)")
    raw = RawViews().to(DEV)
    audit = rng.sample(real, 200) + rng.sample(fake, 200)
    ya = np.array([int(r["label"]) for r in audit])
    get = lambda br: torch.cat([raw.views(torch.from_numpy(np.stack([crop(load_audio(P(r, br)), False) for r in audit[i:i + 50]])).to(DEV)).cpu()
                                for i in range(0, len(audit), 50)])
    Vc, V16, V8 = get(None), get("16k"), get("8k")
    rows = 128 // NB
    stab = np.zeros((3, NB)); disc = np.zeros((3, NB))
    for c in range(3):
        for b in range(NB):
            sl = slice(b * rows, (b + 1) * rows)
            a = Vc[:, c, sl].flatten(1); s = []
            for Vx in (V16, V8):
                x = Vx[:, c, sl].flatten(1)
                am, xm = a - a.mean(1, keepdim=True), x - x.mean(1, keepdim=True)
                s.append(((am * xm).sum(1) / (am.norm(dim=1) * xm.norm(dim=1) + 1e-8)).mean().item())
            stab[c, b] = max(min(s), 0)
            band = V16[:, c, sl]
            d = [abs(roc_auc(ya, band.mean((1, 2)).numpy()) - .5) * 2, abs(roc_auc(ya, band.std((1, 2)).numpy()) - .5) * 2]
            disc[c, b] = max(d)
    score = stab * disc
    w = np.where(score >= 0.25 * score.max(), score / score.max(), 0.0)
    names = ["mel", "fine-spectrum", "phase-IF"]
    for c in range(3):
        log(f"  {names[c]:14} kept slices (0 = lowest): {[b for b in range(NB) if w[c, b] > 0]}  "
            f"mean stability {stab[c].mean():.2f}  mean separation {disc[c].mean():.2f}")

    # ---------------- Step 2: retrain CNN on the selected image
    front = SelectedImage(w).to(DEV)
    cnn, arch = build_cnn(); cnn = cnn.to(DEV)
    opt = torch.optim.AdamW(cnn.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (len(train) // BATCH), pct_start=0.15)
    lossf = nn.CrossEntropyLoss()

    @torch.no_grad()
    def score_cnn(rows_, fn, br):
        cnn.eval(); out = []
        for i in range(0, len(rows_), 64):
            x = torch.from_numpy(np.stack([crop(load_audio(fn(r, br)), False) for r in rows_[i:i + 64]])).to(DEV)
            lg = cnn(front(x)).float(); out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
        return np.concatenate(out)

    log(f"Step 2: training {arch} on the selected image (clean + MP3 32k/16k/8k copies)")
    best = 1e9
    for ep in range(1, EPOCHS + 1):
        cnn.train(); random.shuffle(train); tot = 0.0
        for i in range(0, len(train) - BATCH + 1, BATCH):
            rb = train[i:i + BATCH]
            x = torch.from_numpy(np.stack([crop(load_audio(P(r, random.choice([None, "32k", "16k", "8k"]))), True) for r in rb])).to(DEV)
            y = torch.tensor([int(r["label"]) for r in rb], device=DEV)
            loss = lossf(cnn(spec_augment(front(x))), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        sv = np.concatenate([score_cnn(pv["val"], P, b) for b in (None, "16k", "8k")])
        e = eer(np.tile(yva, 3), sv)[0]
        log(f"  epoch {ep:2d}: loss {tot / (len(train) // BATCH):.4f}  val EER (clean+16k+8k) {e * 100:.2f}%")
        if e <= best:
            best = e; torch.save(cnn.state_dict(), ROOT / "models/exp009b_image_cnn.pt")
    cnn.load_state_dict(torch.load(ROOT / "models/exp009b_image_cnn.pt", map_location=DEV))

    # ---------------- Step 3: combination with WavLM F
    log("Step 3: combination with WavLM F (scores from the saved F model + EXP-005 features)")
    art = model_store.load(); probeF = art["probe_F"]
    Fz = dict(np.load(ROOT / "results/exp005_feats.npz", allow_pickle=True))
    assert (Fz["yte"] == yte).all() and (Fz["yva"] == yva).all(), "file order mismatch with EXP-005"
    key = lambda b: "clean" if b is None else b
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    calI, fus = {}, {}
    for b in [None] + BRS:
        sF = probeF.decision_function(Fz[f"va_{b}"]); sI = score_cnn(pv["val"], P, b)
        calI[b] = PlattCalibrator().fit(sI, yva)
        aF, bF = art["cal"][key(b)]
        X = np.c_[logit(1 / (1 + np.exp(-(aF * sF + bF)))), logit(calI[b].predict_proba(sI))]
        fus[b] = LogisticRegression(C=1.0).fit(X, yva)
    conds = [("clean", P, None)] + [(f"single_{b}", P, b) for b in BRS] + [(f"double_{b}", D, b) for b in BRS]
    rb = np.random.default_rng(0); boot = [rb.integers(0, len(yte), len(yte)) for _ in range(1000)]
    out = []

    def add(method, c, p, extra=None):
        ok = (p >= .5) == yte; bs = [ok[i].mean() for i in boot]
        r = {"method": method, "condition": c, "acc": float(ok.mean()), "acc_lo": float(np.quantile(bs, .025)),
             "acc_hi": float(np.quantile(bs, .975)), "auc": roc_auc(yte, p), "eer": eer(yte, p)[0],
             "false_alarm": float((p[yte == 1] < .5).mean()), "miss": float((p[yte == 0] >= .5).mean())}
        r.update(extra or {}); out.append(r)
        log(f"  {c:10} {method:34} acc {r['acc']*100:5.1f}%  EER {r['eer']*100:4.1f}%  FA {r['false_alarm']*100:4.1f}%  "
            f"miss {r['miss']*100:4.1f}%" + (f"  McNemar vs F p={r['p_vs_F']:.3f} (fused-only {r['fused_only']}, F-only {r['F_only']})" if extra else ""))
        return ok

    for c, fn, b in conds:
        sF = probeF.decision_function(Fz[f"te_{c}"]); aF, bF = art["cal"][key(b)]
        pF = 1 / (1 + np.exp(-(aF * sF + bF)))
        sI = score_cnn(test, fn, b); pI = calI[b].predict_proba(sI)
        okF = add("WavLM F (reference)", c, pF)
        add("Image CNN, selected features", c, pI)
        pAvg = 1 / (1 + np.exp(-(logit(pF) + logit(pI)) / 2))
        add("Combined: average", c, pAvg)
        pLR = fus[b].predict_proba(np.c_[logit(pF), logit(pI)])[:, 1]
        okC = (pLR >= .5) == yte
        n01, n10 = int((okC & ~okF).sum()), int((~okC & okF).sum())
        add("Combined: learned weights", c, pLR, {"p_vs_F": float(binomtest(n01, n01 + n10, .5).pvalue) if n01 + n10 else 1.0,
                                                  "fused_only": n01, "F_only": n10})
    json.dump({"bands": {"stability": stab.tolist(), "separation": disc.tolist(), "weight": w.tolist()}, "rows": out},
              open(ROOT / "results/exp009b.json", "w"), indent=2)
    log("EXP009B_DONE")


if __name__ == "__main__":
    main()
