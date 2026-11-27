"""EXP-010-SC: a NEW model (EXP-010 architecture, trained from scratch) on EXP-010's data + SpoofCeleb.

EXP-010 (models/exp010_wavlm_ft.pt) is never modified. Output: models/exp010_sc_wavlm_ft.pt (not deployed).

Training data
  * EXP-010 training manifest (39,472 clips: ASVspoof 5, Fake-or-Real, LibriSeVoc, ASVspoof 2019, MLAAD)
  * SpoofCeleb train: SC_REAL real (a00) + SC_FAKE AI clips spread evenly over a01..a10 (random, seed 27)
Compression schedule (applied on the fly, nothing written to disk), per clip:
  original + 64k + 32k always; 16k for P16 of clips; 8k for P8 of clips; codec per copy random mp3/opus/aac.
Validation / calibration: EXP-010 val manifest + 1,000 real + 1,000 AI SpoofCeleb development clips,
  each original and at a random 16/32 kbps codec.
Tests (both EXP-010 and the new model, same files): In-the-Wild 2,000 (EXP-007 set), MLAAD 31 held-out
  systems 2,000, SpoofCeleb evaluation 1,500 real + 1,500 AI (all eval attacks), ASVspoof 2019 1,000;
  each original and Opus 16 kbps. Results: results/exp010_sc.json, log results/exp010_sc_log.txt.
"""
import csv, json, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.data.audio_processing import N_SAMPLES, SR, load_audio, peak_normalize  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402
from src.models.wavlm_detector import WavLMDetector  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SC = ROOT / "data/raw/spoofceleb/spoofceleb/flac"
LOG = ROOT / "results/exp010_sc_log.txt"
OUT = ROOT / "models/exp010_sc_wavlm_ft.pt"
ARCH = {"n_trainable": 8, "low_branch": True}                     # same as EXP-010
SC_REAL, SC_FAKE, P16, P8 = 60000, 60000, 0.35, 0.10
BATCH, LR_BB, LR_HEAD, WORKERS = 12, 2e-5, 1e-3, 8
FMT = {"mp3": ("libmp3lame", "mp3", "mp3"), "opus": ("libopus", "ogg", "ogg"), "aac": ("aac", "adts", "aac")}  # encoder, mux, demux
CODECS = list(FMT)
rng = random.Random(27)


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


def codec(x, name, br):
    """Encode a 16 kHz float clip with a real codec and decode it back (in memory)."""
    enc, fmt, dfmt = FMT[name]
    a = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                        "-i", "pipe:0", "-c:a", enc, "-b:a", br, "-f", fmt, "pipe:1"],
                       input=x.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    b = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", dfmt, "-i", "pipe:0", "-ac", "1",
                        "-ar", str(SR), "-f", "f32le", "pipe:1"], input=a, capture_output=True, check=True).stdout
    y = np.frombuffer(b, dtype=np.float32)
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def crop(x, g, train=True):
    if len(x) < N_SAMPLES:
        x = np.pad(x, (0, N_SAMPLES - len(x)), mode="reflect" if len(x) > N_SAMPLES // 2 else "constant")
    s = g.integers(0, len(x) - N_SAMPLES + 1) if train else (len(x) - N_SAMPLES) // 2
    return x[s:s + N_SAMPLES].astype(np.float32)


class Items(Dataset):
    """items: (path, label, codec_name or None, bitrate or None)"""
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab, c, br = self.items[i]
        g = np.random.default_rng() if self.train else np.random.default_rng(i)
        try:
            x = crop(load_audio(str(p)), g, self.train)
            if c:
                x = codec(x, c, br)
        except Exception:
            x = np.zeros(N_SAMPLES, np.float32)       # unreadable file: silent clip (logged count stays tiny)
        return torch.from_numpy(peak_normalize(x)), lab


def expand(clips):
    items = []
    for p, lab in clips:
        items.append((p, lab, None, None))
        for br in ("64k", "32k"):
            items.append((p, lab, rng.choice(CODECS), br))
        if rng.random() < P16:
            items.append((p, lab, rng.choice(CODECS), "16k"))
        if rng.random() < P8:
            items.append((p, lab, rng.choice(CODECS), "8k"))
    return items


def sc_split(split, n_real, n_fake):
    real = sorted((SC / split / "a00").glob("*/*.flac"))
    atts = sorted(d for d in (SC / split).iterdir() if d.is_dir() and d.name != "a00")
    per = n_fake // max(1, len(atts))
    fake = []
    for d in atts:
        fs = sorted(d.glob("*/*.flac"))
        fake += [(f, 0, d.name) for f in rng.sample(fs, min(per, len(fs)))]
    real = [(f, 1, "a00") for f in rng.sample(real, min(n_real, len(real)))]
    return real, fake, [d.name for d in atts]


@torch.no_grad()
def batch_scores(model, dl):
    model.eval(); s, y = [], []
    for x, lab in dl:
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            lg = model(x.to(DEV)).float()
        s.append((lg[:, 1] - lg[:, 0]).cpu().numpy()); y.append(lab.numpy())
    return np.concatenate(s), np.concatenate(y)


class FileSet(Dataset):
    """Whole-file scoring for tests: returns all 4 s windows (50% overlap) of one file, optional Opus 16k."""
    def __init__(self, files, opus):
        self.files, self.opus = files, opus

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p, lab = self.files[i]
        try:
            x = load_audio(str(p))
            if self.opus:
                x = codec(x, "opus", "16k")
        except Exception:
            x = np.zeros(N_SAMPLES, np.float32)
        if len(x) < N_SAMPLES:
            x = np.pad(x, (0, N_SAMPLES - len(x)))
        st = list(range(0, len(x) - N_SAMPLES + 1, N_SAMPLES // 2)) or [0]
        w = np.stack([peak_normalize(x[s:s + N_SAMPLES].astype(np.float32)) for s in st])
        return torch.from_numpy(w), lab


def first(b):                                         # top-level so Windows worker processes can pickle it
    return b[0]


@torch.no_grad()
def file_scores(model, files, opus):
    dl = DataLoader(FileSet(files, opus), 1, num_workers=WORKERS, collate_fn=first)
    out = []
    for w, _ in dl:
        sc = []
        for k in range(0, len(w), 16):
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                lg = model(w[k:k + 16].to(DEV)).float()
            sc.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
        out.append(float(np.median(np.concatenate(sc))))
    return np.array(out)


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = WavLMDetector(**ck.get("arch", {})); m.load_state_dict(ck["state_dict"])
    return m.to(DEV).eval(), ck


def main():
    LOG.write_text("")
    # ---------------- data
    tr10 = [(Path(r["path"]), int(r["label"])) for r in csv.DictReader(open(ROOT / "data/manifests/exp010_train.csv"))]
    va10 = [(Path(r["path"]), int(r["label"])) for r in csv.DictReader(open(ROOT / "data/manifests/exp010_val.csv"))]
    r_tr, f_tr, att_tr = sc_split("train", SC_REAL, SC_FAKE)
    r_dv, f_dv, _ = sc_split("development", 1000, 1000)
    clips = tr10 + [(p, l) for p, l, _ in r_tr + f_tr]
    items = expand(clips); rng.shuffle(items)
    log(f"train clips: {len(clips)} (EXP-010 data {len(tr10)}, SpoofCeleb {len(r_tr)} real + {len(f_tr)} AI "
        f"from {', '.join(att_tr)}) -> {len(items)} training examples after compression schedule")
    log("  examples per condition: " + ", ".join(f"{k}: {sum(1 for it in items if (it[3] or 'original') == k)}"
                                              for k in ("original", "64k", "32k", "16k", "8k")))
    vclips = va10 + [(p, l) for p, l, _ in r_dv + f_dv]
    vitems = [(p, l, None, None) for p, l in vclips] + [(p, l, rng.choice(CODECS), rng.choice(["16k", "32k"])) for p, l in vclips]
    log(f"validation: {len(vclips)} clips x (original + 16/32k copy) = {len(vitems)}")

    # ---------------- model (fresh, same architecture as EXP-010)
    model = WavLMDetector(**ARCH).to(DEV)
    dl = DataLoader(Items(items, True), BATCH, shuffle=False, num_workers=WORKERS, drop_last=True,
                    persistent_workers=True, prefetch_factor=4)
    dv = DataLoader(Items(vitems, False), 32, num_workers=WORKERS)
    opt = torch.optim.AdamW(model.param_groups(LR_BB, LR_HEAD), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_BB, LR_HEAD], total_steps=len(dl), pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    lossf = nn.CrossEntropyLoss()
    model.train(); model.wavlm.eval()
    t0, tot = time.time(), 0.0
    for i, (x, y) in enumerate(dl, 1):
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            loss = lossf(model(x.to(DEV)).float(), y.to(DEV))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 5.0)
        scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        if i % 200 == 0:
            el = time.time() - t0
            log(f"  step {i}/{len(dl)}  loss {tot / i:.4f}  {el / i:.2f} s/step  ETA {(len(dl) - i) * el / i / 3600:.1f} h")
        if i % 5000 == 0:
            torch.save({"state_dict": model.state_dict(), "step": i, "arch": ARCH}, ROOT / "models/exp010_sc_partial.pt")
            model.train(); model.wavlm.eval()
    log(f"training done in {(time.time() - t0) / 3600:.1f} h")

    # ---------------- calibration on validation
    s, y = batch_scores(model, dv)
    cal = PlattCalibrator().fit(s, y); pv = cal.predict_proba(s)
    ts = np.unique(pv); fa = np.array([(pv[y == 1] < t).mean() for t in ts]); ms = np.array([(pv[y == 0] >= t).mean() for t in ts])
    thr = float(ts[np.argmin(np.abs(fa - ms))])
    log(f"validation: EER {eer(y, s)[0] * 100:.2f}%  AUC {roc_auc(y, s):.4f}  threshold P(real) >= {thr:.3f}")
    torch.save({"state_dict": model.state_dict(), "platt": (cal.a, cal.b), "threshold": thr, "threshold_strict": thr,
                "version": "EXP010-SC", "arch": ARCH, "note": "EXP-010 data + SpoofCeleb; not deployed"}, OUT)
    (ROOT / "models/exp010_sc_partial.pt").unlink(missing_ok=True)
    log(f"saved {OUT.name}")
    del model; torch.cuda.empty_cache()

    # ---------------- tests
    e7 = list(csv.DictReader(open(ROOT / "results/exp007_files.csv")))
    meta = next((ROOT / "data/raw/in_the_wild").rglob("meta.csv")).parent
    import yaml
    from run_phase2_pilot import sample_split
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    r_ev, f_ev, att_ev = sc_split("evaluation", 1500, 1500)
    sets = {
        "In-the-Wild": [(meta / r["file"], int(r["label"])) for r in e7],
        "MLAAD 31 unseen systems": [(Path(r["path"]), int(r["label"])) for r in csv.DictReader(open(ROOT / "data/manifests/exp010_test_mlaad_heldout.csv"))],
        "SpoofCeleb evaluation": [(p, l) for p, l, _ in r_ev + f_ev],
        "ASVspoof 2019": [(Path(r["path"]), int(r["label"])) for r in sample_split(ev, {"n_train": 0, "n_val": 0, "n_test": 500}, 7)["test"]],
    }
    eatt = [a for _, _, a in r_ev + f_ev]
    log(f"SpoofCeleb evaluation sample: {len(r_ev)} real + {len(f_ev)} AI from {', '.join(att_ev)}")
    res = {"thr": thr, "tests": {}}
    for tag, path in (("EXP-010", ROOT / "models/exp010_wavlm_ft.pt"), ("EXP-010-SC", OUT)):
        model, ck = load_model(path); a, b = ck["platt"]; th = ck["threshold"]
        log(f"Testing {tag}")
        for name, files in sets.items():
            for opus in (False, True):
                cond = f"{name} | {'Opus 16k' if opus else 'original'}"
                S = file_scores(model, files, opus); Y = np.array([l for _, l in files])
                P = 1 / (1 + np.exp(-(a * S + b)))
                r = {"acc": float(((P >= th) == (Y == 1)).mean()), "false_alarm": float((P[Y == 1] < th).mean()),
                     "miss": float((P[Y == 0] >= th).mean()), "auc": roc_auc(Y, S), "eer": eer(Y, S)[0]}
                if name == "SpoofCeleb evaluation":
                    r["per_attack_detected"] = {k: float((P[np.array([x == k for x in eatt])] < th).mean()) for k in att_ev}
                res["tests"].setdefault(cond, {})[tag] = r
                log(f"  {cond:40} acc {r['acc']*100:5.1f}%  real->AI {r['false_alarm']*100:5.1f}%  "
                    f"AI->real {r['miss']*100:5.1f}%  EER {r['eer']*100:4.1f}%")
                if "per_attack_detected" in r:
                    log("      AI detected per attack: " + "  ".join(f"{k} {v*100:.0f}%" for k, v in r["per_attack_detected"].items()))
                json.dump(res, open(ROOT / "results/exp010_sc.json", "w"), indent=2)
        del model; torch.cuda.empty_cache()
    log("EXP010_SC_DONE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        log("CRASHED:\n" + traceback.format_exc())
        raise
