"""EXP-008: fine-tune WavLM-Base+ (top 4 layers + layer weights + attentive pooling) on the merged, balanced
ASVspoof 5 + Fake-or-Real + LibriSeVoc manifest (exp008_data.py), then evaluate on data NEVER used for training:
  * In-the-Wild: the SAME 2,000 clips as EXP-007 (direct before/after comparison), app-style 4 s windows + median
  * ASVspoof 2019 eval (EXP-005 1,000-file set): clean and MP3 16 kbps, to check lab performance is kept
Threshold: Platt calibration on the validation set, operating point chosen so <= 5% of real validation voices
are called AI. Writes models/exp008_wavlm_ft.pt and results/exp008.json
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from src.data.audio_processing import N_SAMPLES, load_audio, peak_normalize  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402
from src.models.wavlm_detector import WavLMDetector  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = sys.argv[1] if len(sys.argv) > 1 else "exp008"
# EXP-010: 8 fine-tuned layers + low-layer "texture" branch (diagnostics showed the top layer alone decides)
ARCH = {"n_trainable": 8, "low_branch": True} if TAG == "exp010" else {}
EPOCHS, BATCH, LR_BB, LR_HEAD, MAX_FA = (3, 12, 2e-5, 1e-3, 0.05) if TAG == "exp010" else (4, 16, 2e-5, 1e-3, 0.05)
LOG = ROOT / "results" / f"{TAG}_log.txt"


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


def crop(x, rng, train):
    if len(x) < N_SAMPLES:
        x = np.pad(x, (0, N_SAMPLES - len(x)), mode="reflect" if len(x) > 1 else "constant") if len(x) > N_SAMPLES // 2 \
            else np.pad(x, (0, N_SAMPLES - len(x)))
    s = rng.integers(0, len(x) - N_SAMPLES + 1) if train else (len(x) - N_SAMPLES) // 2
    return peak_normalize(x[s:s + N_SAMPLES].astype(np.float32))


class Mix(Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows) * (1 if self.train else 2)

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else i)
        r = self.rows[i % len(self.rows)]
        use_codec = rng.random() < 0.5 if self.train else i >= len(self.rows)   # val: every clip clean AND codec
        x = load_audio(r["codec_path"] if use_codec else r["path"])
        return torch.from_numpy(crop(x, rng, self.train)), int(r["label"])


@torch.no_grad()
def scores(model, loader):
    model.eval(); S, Y = [], []
    for x, y in loader:
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            lg = model(x.to(DEV)).float()
        S.append((lg[:, 1] - lg[:, 0]).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(S), np.concatenate(Y)


@torch.no_grad()
def score_file(model, x):
    from src.app.server import windows
    ws = torch.stack([torch.from_numpy(peak_normalize(w.astype(np.float32))) for _, _, w in windows(x)]).to(DEV)
    with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
        lg = model(ws).float()
    return float(np.median((lg[:, 1] - lg[:, 0]).cpu().numpy()))


def report(name, y, s, cal, thr):
    p = cal.predict_proba(s)
    r = {"auc": roc_auc(y, s), "eer": eer(y, s)[0], "accuracy": float(((p >= thr) == y).mean()),
         "false_alarm_real_called_AI": float((p[y == 1] < thr).mean()), "miss_AI_called_real": float((p[y == 0] >= thr).mean())}
    log(f"  {name:28} acc {r['accuracy']*100:5.1f}%  false alarms {r['false_alarm_real_called_AI']*100:5.1f}%  "
        f"misses {r['miss_AI_called_real']*100:5.1f}%  AUC {r['auc']:.3f}  EER {r['eer']*100:4.1f}%")
    return r


def main():
    LOG.write_text("")
    rd = lambda s: list(csv.DictReader(open(ROOT / f"data/manifests/{TAG}_{s}.csv")))
    tr, va = rd("train"), rd("val")
    log(f"train {len(tr)} clips, val {len(va)} clips, device {DEV}")
    for ds in sorted({r["dataset"] for r in tr}):
        log(f"   {ds}: train {sum(r['dataset'] == ds for r in tr)}, val {sum(r['dataset'] == ds for r in va)}")
    torch.manual_seed(8)
    dl_tr = DataLoader(Mix(tr, True), BATCH, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(Mix(va, False), 32, num_workers=4)
    model = WavLMDetector(**ARCH).to(DEV)
    opt = torch.optim.AdamW(model.param_groups(LR_BB, LR_HEAD), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_BB, LR_HEAD], total_steps=EPOCHS * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    lossf = nn.CrossEntropyLoss()
    best, out = 1e9, ROOT / "models" / f"{TAG}_wavlm_ft.pt"
    hist = []
    for ep in range(1, EPOCHS + 1):
        model.train(); model.wavlm.eval()   # keep frozen layers' dropout off
        t0, tot = time.time(), 0.0
        for i, (x, y) in enumerate(dl_tr, 1):
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = lossf(model(x.to(DEV)).float(), y.to(DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
            if i % 100 == 0:
                log(f"  epoch {ep} step {i}/{len(dl_tr)}  loss {tot / i:.4f}  ({(time.time() - t0) / i:.2f} s/step)")
        s, y = scores(model, dl_va)
        e, a = eer(y, s)[0], roc_auc(y, s)
        hist.append({"epoch": ep, "train_loss": tot / len(dl_tr), "val_eer": e, "val_auc": a})
        log(f"epoch {ep}: train loss {tot / len(dl_tr):.4f}  val EER {e*100:.2f}%  val AUC {a:.4f}")
        if e < best:
            best = e
            torch.save(model.state_dict(), out)
    model.load_state_dict(torch.load(out, map_location=DEV))

    s, y = scores(model, dl_va)
    cal = PlattCalibrator().fit(s, y)
    pv = cal.predict_proba(s)
    thr_strict = float(np.quantile(pv[y == 1], MAX_FA))          # <= 5% of real validation voices fall below
    ts = np.unique(pv); fa = np.array([(pv[y == 1] < t).mean() for t in ts]); ms = np.array([(pv[y == 0] >= t).mean() for t in ts])
    thr = float(ts[np.argmin(np.abs(fa - ms))])                  # balanced (EER) operating point -> deployed
    torch.save({"state_dict": model.state_dict(), "platt": (cal.a, cal.b), "threshold": thr,
                "threshold_strict": thr_strict, "version": TAG.upper(), "arch": ARCH}, out)
    log(f"operating points: balanced/EER P(real) >= {thr:.3f} (deployed),  strict <=5% FA: {thr_strict:.3f}")
    res = {"history": hist, "threshold": thr, "threshold_strict": thr_strict,
           "val": report("validation (mixed)", y, s, cal, thr)}

    log("Test 1: In-the-Wild, same 2,000 clips as EXP-007 (never trained on)")
    from run_exp007a_diagnose import decode
    e7 = list(csv.DictReader(open(ROOT / "results/exp007_files.csv")))
    meta = next((ROOT / "data/raw/in_the_wild").rglob("meta.csv")).parent
    S, Y = [], []
    for i, r in enumerate(e7, 1):
        S.append(score_file(model, decode(meta / r["file"]))); Y.append(int(r["label"]))
        if i % 500 == 0:
            log(f"   {i}/{len(e7)}")
    res["in_the_wild"] = report("In-the-Wild (EXP-007 set)", np.array(Y), np.array(S), cal, thr)
    res["in_the_wild_strict"] = report("In-the-Wild, strict threshold", np.array(Y), np.array(S), cal, thr_strict)

    log("Test 2: ASVspoof 2019 eval, EXP-005 1,000-file set (never trained on)")
    import yaml
    from run_phase2_pilot import sample_split
    from src.data.offline_cache import cache_path
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    test = sample_split(ev, {"n_train": 0, "n_val": 0, "n_test": 500}, 7)["test"]
    for name, fn in [("ASVspoof 2019 clean", lambda r: r["path"]),
                     ("ASVspoof 2019 MP3 16k", lambda r: str(cache_path(str(ROOT / "data/cache"), "mp3", "16k", r["file_id"])))]:
        S = np.array([score_file(model, load_audio(fn(r))) for r in test]); Y = np.array([int(r["label"]) for r in test])
        res[name] = report(name, Y, S, cal, thr)
    rw = sorted(p for p in (ROOT / "data/real_world").glob("*") if p.is_file() and p.name != ".gitkeep")
    if rw:
        log("Test 3: your own recordings in data/real_world (all assumed REAL)")
        res["real_world"] = []
        for f in rw:
            p_ = float(cal.predict_proba(np.array([score_file(model, decode(f))]))[0])
            res["real_world"].append({"file": f.name, "p_real": p_, "verdict": "real" if p_ >= thr else "AI"})
            log(f"  {f.name}: P(real) = {p_:.3f} -> {'REAL' if p_ >= thr else 'AI'}")
    hp = ROOT / "data/manifests/exp010_test_mlaad_heldout.csv"
    if hp.exists():
        log("Test 4: MLAAD English, TTS systems held out from training (unseen generators) vs unused real audiobook speech")
        ht = list(csv.DictReader(open(hp)))
        S = np.array([score_file(model, load_audio(r["path"])) for r in ht]); Y = np.array([int(r["label"]) for r in ht])
        res["mlaad_heldout"] = report("MLAAD held-out systems", Y, S, cal, thr)
        res["mlaad_heldout_strict"] = report("MLAAD held-out, strict", Y, S, cal, thr_strict)
        per = {}
        for r, sc in zip(ht, S):
            if r["label"] == "0":
                per.setdefault(r["model"], []).append(float(cal.predict_proba(np.array([sc]))[0] < thr))
        res["mlaad_heldout_per_system_detected"] = {k: float(np.mean(v)) for k, v in per.items()}
        for k, v in sorted(per.items()):
            log(f"     {k:32} AI detected {np.mean(v)*100:5.1f}%  (n={len(v)})")
    lab = [(p, 1) for p in (ROOT / "data/real_world/real").glob("*") if p.is_file()] + \
          [(p, 0) for p in (ROOT / "data/real_world/ai").glob("*") if p.is_file()]
    if lab:
        log("Test 5: your labelled live-test clips (data/real_world/real and /ai)")
        res["live_test"] = []
        for f, l in lab:
            sc = score_file(model, decode(f)); p_ = float(cal.predict_proba(np.array([sc]))[0])
            v = "real" if p_ >= thr_strict else "AI"
            res["live_test"].append({"file": f.name, "truth": "real" if l else "AI", "score": sc, "p_real": p_, "verdict": v})
            log(f"  [{'REAL' if l else ' AI '}] {f.name[:45]:45} score {sc:6.2f}  P(real) {p_:.3f} -> {v}  {'OK' if (v == 'real') == bool(l) else 'WRONG'}")
    json.dump(res, open(ROOT / f"results/{TAG}.json", "w"), indent=2)
    log("EXP008_DONE")


if __name__ == "__main__":
    main()
