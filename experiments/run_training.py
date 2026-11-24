"""Phases 3-5 training (dev split only; locked set never touched).

  python experiments/run_training.py --model baseline_b  --config config/baseline_rawnet2.yaml   # Phase 3
  python experiments/run_training.py --model baseline_c  --config config/baseline_rawnet2.yaml   # 50% clean
  python experiments/run_training.py --model deltanet    --config config/delta_net_mvp.yaml     # Phase 4
  ... --set perturbation=gaussian   (override any config key, e.g. Baseline F / ablations)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import DegradedAudioDataset  # noqa: E402
from src.evaluation.calibrator import PlattCalibrator  # noqa: E402
from src.evaluation.metrics import bootstrap_ci, eer, roc_auc  # noqa: E402
from src.models.backbones import RawNet2Encoder, load_pretrained_rawnet2  # noqa: E402
from src.models.delta_net import DeltaNet  # noqa: E402
from src.training.losses import weighted_ce  # noqa: E402
from src.training.trainer import score_loader, train  # noqa: E402


def parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def build_model(kind, cfg):
    rawnet = load_pretrained_rawnet2(str(ROOT / cfg["pretrained"]))
    if kind in ("baseline_b", "baseline_c"):
        return rawnet
    enc = RawNet2Encoder(rawnet, proj_dim=cfg.get("proj_dim"))
    return DeltaNet(enc, perturbation=cfg.get("perturbation", "mulaw8"),
                    hidden=cfg.get("hidden", 256), dropout=cfg.get("dropout", 0.3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["baseline_b", "baseline_c", "deltanet"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", help="key=value overrides")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0, help="debug: limit train files")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)) | parse_overrides(a.set)
    seed = cfg.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kw = dict(codec=cfg["codec"], bitrate=cfg["bitrate"], cache_root=str(ROOT / cfg["cache_root"]))
    tr = DegradedAudioDataset(str(ROOT / cfg["train_manifest"]), train=True, seed=seed,
                              clean_prob=0.5 if a.model == "baseline_c" else 0.0, **kw)
    if a.max_train:
        tr.rows = random.sample(tr.rows, a.max_train)
    dv = DegradedAudioDataset(str(ROOT / cfg["dev_manifest"]), train=False, **kw)
    # Dev is split in half: A -> checkpoint selection, B -> Platt calibration (disjoint).
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dv.rows))
    dev_sel = DegradedAudioDataset("", train=False, rows=[dv.rows[i] for i in idx[: len(idx) // 2]], **kw)
    dev_cal = DegradedAudioDataset("", train=False, rows=[dv.rows[i] for i in idx[len(idx) // 2:]], **kw)

    dl = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh, num_workers=a.workers,
                                   pin_memory=True, persistent_workers=a.workers > 0, drop_last=sh)
    model = build_model(a.model, cfg).to(device)
    crit = weighted_ce(tr.class_weights().to(device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    name = a.model + (f"_{a.tag}" if a.tag else "")
    out = ROOT / "results/runs" / f"{name}_{time.strftime('%Y%m%d_%H%M')}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"device={device} train={len(tr)} dev_sel={len(dev_sel)} dev_cal={len(dev_cal)} -> {out}")

    best = train(model, dl(tr, True), dl(dev_sel, False), crit, opt, device, out,
                 epochs=cfg["epochs"], patience=cfg["patience"])

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    y, s = score_loader(model, dl(dev_cal, False), device)
    cal = PlattCalibrator().fit(s, y)
    cal.save(out / "platt.json")
    e, lo, hi = bootstrap_ci(y, s, eer)
    a_, alo, ahi = bootstrap_ci(y, s, roc_auc)
    summ = {"model": name, "best_dev_sel_eer": best, "dev_cal_eer": [e, lo, hi], "dev_cal_auc": [a_, alo, ahi]}
    json.dump(summ, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
