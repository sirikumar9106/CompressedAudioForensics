"""EXP-009a: "multi-view fingerprint image" + CNN, alone (no WavLM), on ASVspoof 2019 studio data only.

Image (3 channels, each 128 x 400, aligned in time and frequency):
  R  log-mel spectrogram 0-8 kHz            (what we hear)
  G  high-resolution log spectrogram 0-4 kHz  (fine low-frequency harmonics; survives compression)
  B  instantaneous-frequency deviation 0-4 kHz (phase timing; inaudible, often wrong in AI speech)
CNN: ResNet-18 (ImageNet weights if torchvision is available, else a compact ResNet from scratch).
Train: 2019 LA train (attacks A01-A06), all 2,580 real + 2,580 random fake, clean or MP3 32k per step.
Calibration: one Platt calibrator per bitrate on the same 200 validation clips used by EXP-004d/005.
Test: the SAME 1,000 unseen-attack files and 9 conditions as EXP-005. For one comparison table, the official
RawNet2 (Model A) and the Delta-Net (mu-law difference) probe are re-scored here on the same files with the same
per-bitrate calibration; WavLM rows are read from results/exp005.json (identical files/conditions).
Writes models/exp009a_image_cnn.pt, results/exp009a.json
"""
import csv
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_phase2_pilot import sample_split  # noqa: E402
from run_ssl_probe import fit_probe  # noqa: E402
from src.data.audio_processing import N_SAMPLES, SR, load_audio, peak_normalize, preprocess  # noqa: E402
from src.data.offline_cache import cache_path, degrade_file  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402
from src.models.backbones import load_pretrained_rawnet2  # noqa: E402
from src.models.degradation_engine import apply_mulaw_quantization  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BRS = ["64k", "32k", "16k", "8k"]
EPOCHS, BATCH, LR = 15, 32, 3e-4
LOG = ROOT / "results" / "exp009a_log.txt"
random.seed(9); np.random.seed(9); torch.manual_seed(9)


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


# ---------------------------------------------------------------- image front end (GPU, batched)
def mel_fb(n_fft=512, n_mels=128, fmax=8000):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / SR).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return torch.tensor(fb)


class FingerprintImage(nn.Module):
    def __init__(self, H=128, W=400):
        super().__init__()
        self.H, self.W = H, W
        self.register_buffer("fb", mel_fb())
        self.register_buffer("w512", torch.hann_window(512)); self.register_buffer("w1024", torch.hann_window(1024))

    def _resize(self, x):
        return Fn.interpolate(x.unsqueeze(1), size=(self.H, self.W), mode="bilinear", align_corners=False).squeeze(1)

    @torch.no_grad()
    def forward(self, x):                                   # x [B, L] -> [B, 3, H, W]
        s512 = torch.stft(x, 512, 160, window=self.w512, return_complex=True)     # [B, 257, T]
        mag = s512.abs() ** 2
        R = torch.log(torch.einsum("mf,bft->bmt", self.fb, mag) + 1e-6)
        s1024 = torch.stft(x, 1024, 160, window=self.w1024, return_complex=True)
        G = torch.log(s1024.abs()[:, :257] ** 2 + 1e-6)                           # 0-4 kHz, 15.6 Hz bins
        ph = torch.angle(s512[:, :129])                                          # 0-4 kHz
        k = torch.arange(129, device=x.device).view(1, -1, 1)
        d = ph[:, :, 1:] - ph[:, :, :-1] - 2 * math.pi * k * 160 / 512
        d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
        B = Fn.pad(d, (1, 0)) * (torch.log(s512.abs()[:, :129] + 1e-6) > -6).float()   # ignore near-silent bins
        img = torch.stack([self._resize(c.flip(1)) for c in (R, G, B)], 1)        # low frequencies at the bottom
        mu = img.mean((2, 3), keepdim=True); sd = img.std((2, 3), keepdim=True) + 1e-5
        return (img - mu) / sd


def build_cnn():
    try:
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, 2)
        return m, "ResNet-18 (ImageNet-pretrained, torchvision)"
    except Exception as e:  # noqa: BLE001
        log(f"torchvision unavailable ({e.__class__.__name__}); using compact ResNet from scratch")

        def block(ci, co, s):
            return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                                 nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True))
        m = nn.Sequential(block(3, 32, 2), block(32, 64, 2), block(64, 128, 2), block(128, 256, 2),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 2))
        return m, "compact ResNet (scratch)"


def spec_augment(img):
    b, _, h, w = img.shape
    for i in range(b):
        f0 = random.randint(0, h - 16); t0 = random.randint(0, w - 40)
        img[i, :, f0:f0 + random.randint(0, 15)] = 0
        img[i, :, :, t0:t0 + random.randint(0, 39)] = 0
    return img


def crop(x, train):
    if len(x) < N_SAMPLES:
        return preprocess(x)
    s = random.randint(0, len(x) - N_SAMPLES) if train else (len(x) - N_SAMPLES) // 2
    return peak_normalize(x[s:s + N_SAMPLES].astype(np.float32))


def main():
    LOG.write_text("")
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    cr = str(ROOT / cfg["cache_root"])
    P = lambda r, br: r["path"] if br is None else str(cache_path(cr, "mp3", br, r["file_id"]))
    D = lambda r, br: str(cache_path(cr, "mp3x2from64k", br, r["file_id"]))
    tr_all = list(csv.DictReader(open(ROOT / "data/manifests/asvspoof2019_train.csv")))
    pv = sample_split(tr_all, {"n_train": 300, "n_val": 100, "n_test": 0}, cfg["seed"])   # same as EXP-004d/005
    val_ids = {r["file_id"] for r in pv["val"]}
    pool = [r for r in tr_all if r["file_id"] not in val_ids]
    real = [r for r in pool if r["label"] == "1"]; fake = random.sample([r for r in pool if r["label"] == "0"], len(real))
    train = real + fake
    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    test = sample_split(ev, {"n_train": 0, "n_val": 0, "n_test": 500}, 7)["test"]          # same as EXP-004d/005/006
    yva = np.array([int(r["label"]) for r in pv["val"]]); yte = np.array([int(r["label"]) for r in test])
    log(f"train {len(train)} clips ({len(real)} real / {len(fake)} fake), val {len(pv['val'])}, test {len(test)}")

    jobs = [(r, b) for r in train for b in ("32k",) if not Path(P(r, b)).exists()]
    log(f"encoding {len(jobs)} MP3 32k training copies")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: degrade_file(j[0]["path"], Path(P(*j)), "mp3", j[1]), jobs))

    front = FingerprintImage().to(DEV)
    cnn, arch = build_cnn(); cnn = cnn.to(DEV)
    log(f"CNN: {arch}")
    opt = torch.optim.AdamW(cnn.parameters(), LR, weight_decay=1e-4)
    steps = EPOCHS * (len(train) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss()
    wav = lambda rows, fn, br, train_=False: torch.from_numpy(np.stack([crop(load_audio(fn(r, br)), train_) for r in rows])).to(DEV)

    @torch.no_grad()
    def score(rows, fn, br):
        cnn.eval(); out = []
        for i in range(0, len(rows), 64):
            lg = cnn(front(wav(rows[i:i + 64], fn, br))).float()
            out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
        return np.concatenate(out)

    best = 1e9
    for ep in range(1, EPOCHS + 1):
        cnn.train(); random.shuffle(train); tot = 0.0
        for i in range(0, len(train) - BATCH + 1, BATCH):
            rows = train[i:i + BATCH]
            x = torch.from_numpy(np.stack([crop(load_audio(P(r, random.choice([None, "32k"]))), True) for r in rows])).to(DEV)
            y = torch.tensor([int(r["label"]) for r in rows], device=DEV)
            loss = lossf(cnn(spec_augment(front(x))), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        sv = np.concatenate([score(pv["val"], P, None), score(pv["val"], P, "32k")])
        e = eer(np.tile(yva, 2), sv)[0]
        log(f"epoch {ep:2d}: loss {tot / (len(train) // BATCH):.4f}  val EER {e * 100:.2f}%")
        if e <= best:
            best = e; torch.save(cnn.state_dict(), ROOT / "models/exp009a_image_cnn.pt")
    cnn.load_state_dict(torch.load(ROOT / "models/exp009a_image_cnn.pt", map_location=DEV))

    # ---------------- baselines re-scored on the SAME files with the SAME per-bitrate calibration
    rawnet = load_pretrained_rawnet2(str(ROOT / cfg["pretrained"])).to(DEV).eval()

    @torch.no_grad()
    def rawnet_scores(rows, fn, br):
        A, Dz = [], []
        for r in rows:
            x = torch.from_numpy(preprocess(load_audio(fn(r, br)))).unsqueeze(0).to(DEV)
            lg = rawnet(x); A.append((lg[0, 1] - lg[0, 0]).item())
            z = rawnet.frame_features(x).transpose(1, 2); zp = rawnet.frame_features(apply_mulaw_quantization(x)).transpose(1, 2)
            dz = z - zp; cos = 1 - torch.cosine_similarity(z.mean(1), zp.mean(1), dim=-1)
            Dz.append(torch.cat([dz.mean(1), dz.std(1, unbiased=False), dz.norm(dim=-1).mean(1, keepdim=True), cos[:, None]], -1)[0].cpu().numpy())
        return np.array(A), np.stack(Dz)

    log("fitting baselines (RawNet2, Delta-Net probe) on the EXP-004d/005 train/val clips")
    ytr = np.array([int(r["label"]) for r in pv["train"]])
    _, dz_tr = rawnet_scores(pv["train"], P, "32k")
    vals = {b: rawnet_scores(pv["val"], P, b) for b in [None] + BRS}
    _, dz_probe = fit_probe(dz_tr, ytr, vals["32k"][1], yva)
    cal = {"RawNet2 (official)": {b: PlattCalibrator().fit(vals[b][0], yva) for b in vals},
           "Delta-Net (mu-law difference)": {b: PlattCalibrator().fit(dz_probe.decision_function(vals[b][1]), yva) for b in vals},
           "Image CNN (EXP-009a)": {b: PlattCalibrator().fit(score(pv["val"], P, b), yva) for b in vals}}
    cal_one = PlattCalibrator().fit(score(pv["val"], P, "32k"), yva)

    conds = [("clean", P, None)] + [(f"single_{b}", P, b) for b in BRS] + [(f"double_{b}", D, b) for b in BRS]
    rng = np.random.default_rng(0); boot = [rng.integers(0, len(yte), len(yte)) for _ in range(1000)]
    out = []

    def add(method, c, p, s):
        ok = (p >= .5) == yte; bs = [ok[i].mean() for i in boot]
        out.append({"method": method, "condition": c, "acc": float(ok.mean()), "acc_lo": float(np.quantile(bs, .025)),
                    "acc_hi": float(np.quantile(bs, .975)), "auc": roc_auc(yte, s), "eer": eer(yte, s)[0],
                    "false_alarm": float((p[yte == 1] < .5).mean()), "miss": float((p[yte == 0] >= .5).mean())})
        r = out[-1]
        log(f"  {c:10} {method:32} acc {r['acc']*100:5.1f}%  AUC {r['auc']:.4f}  EER {r['eer']*100:5.1f}%  "
            f"FA {r['false_alarm']*100:4.1f}%  miss {r['miss']*100:4.1f}%")

    log("scoring 1,000 test files x 9 conditions")
    for c, fn, b in conds:
        si = score(test, fn, b); sa, sd = rawnet_scores(test, fn, b); sdz = dz_probe.decision_function(sd)
        add("Image CNN (EXP-009a)", c, cal["Image CNN (EXP-009a)"][b].predict_proba(si), si)
        add("Image CNN, one calibration", c, cal_one.predict_proba(si), si)
        add("RawNet2 (official)", c, cal["RawNet2 (official)"][b].predict_proba(sa), sa)
        add("Delta-Net (mu-law difference)", c, cal["Delta-Net (mu-law difference)"][b].predict_proba(sdz), sdz)
    for r in json.load(open(ROOT / "results/exp005.json")):
        if r["variant"] in ("A", "F"):
            out.append({"method": {"A": "WavLM probe + per-bitrate cal (A)", "F": "WavLM + RawBoost + per-bitrate cal (F)"}[r["variant"]],
                        "condition": r["condition"], **{k: r[k] for k in ("acc", "acc_lo", "acc_hi", "auc", "eer")},
                        "false_alarm": r["false_alarm_rate"], "miss": r["miss_rate"]})
    json.dump({"arch": arch, "train_size": len(train), "rows": out}, open(ROOT / "results/exp009a.json", "w"), indent=2)
    log("EXP009A_DONE")


if __name__ == "__main__":
    main()
