"""EXP-003: bitrate sweep. Model A (official RawNet2 classifier output) vs frozen-probe features (base, Δz)
across MP3 bitrates on the same 600-file subset (pilot_unseen_attacks.yaml split).

Usage: python experiments/run_bitrate_sweep.py [--config config/pilot_unseen_attacks.yaml]
"""
import argparse
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
from run_phase2_pilot import probe, sample_split  # noqa: E402
from src.data.audio_processing import load_audio, preprocess  # noqa: E402
from src.data.offline_cache import cache_path, degrade_file  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402
from src.models.backbones import load_pretrained_rawnet2  # noqa: E402
from src.models.degradation_engine import apply_mulaw_quantization  # noqa: E402

BITRATES = [None, "320k", "128k", "64k", "32k", "16k", "8k"]  # None = clean FLAC


def src_path(r, cfg, br):
    return r["path"] if br is None else str(cache_path(cfg["cache_root"], "mp3", br, r["file_id"]))


@torch.no_grad()
def extract(rows, model, cfg, br, device):
    A, base, delta, y_lab = [], [], [], []
    for r in rows:
        y = torch.from_numpy(preprocess(load_audio(src_path(r, cfg, br)))).unsqueeze(0).to(device)
        lg = model(y)
        A.append((lg[0, 1] - lg[0, 0]).item())  # official score: higher = bonafide
        z = model.frame_features(y).transpose(1, 2)
        zp = model.frame_features(apply_mulaw_quantization(y)).transpose(1, 2)
        dz = z - zp
        cos = 1 - torch.cosine_similarity(z.mean(1), zp.mean(1), dim=-1)
        base.append(z.mean(1)[0].cpu().numpy())
        delta.append(torch.cat([dz.mean(1), dz.std(1, unbiased=False), dz.norm(dim=-1).mean(1, keepdim=True),
                                cos.unsqueeze(-1)], -1)[0].cpu().numpy())
        y_lab.append(int(r["label"]))
    return np.array(A), np.stack(base), np.stack(delta), np.array(y_lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/pilot_unseen_attacks.yaml"))
    cfg = yaml.safe_load(open(ap.parse_args().config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"], newline=""))
            if r["attack"] not in set(cfg.get("exclude_attacks", []))]
    split = sample_split(rows, cfg, cfg["seed"])
    allrows = split["train"] + split["val"] + split["test"]

    jobs = [(r, br) for br in BITRATES if br for r in allrows
            if not cache_path(cfg["cache_root"], "mp3", br, r["file_id"]).exists()]
    print(f"encoding {len(jobs)} files...", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: degrade_file(j[0]["path"], cache_path(cfg["cache_root"], "mp3", j[1], j[0]["file_id"]),
                                           "mp3", j[1]), jobs))

    model = load_pretrained_rawnet2(str(ROOT / cfg["pretrained"])).to(device).eval()
    table = []
    for br in BITRATES:
        d = {k: extract(v, model, cfg, br, device) for k, v in split.items()}
        A, _, _, yte = d["test"]
        row = {"condition": "clean" if br is None else f"mp3_{br}",
               "modelA_auc": roc_auc(yte, A), "modelA_eer": eer(yte, A)[0],
               "modelA_acc@0": float(((A >= 0).astype(int) == yte).mean())}
        for name, i in [("base", 1), ("delta", 2)]:
            r, s = probe(d["train"][i], d["train"][3], d["val"][i], d["val"][3], d["test"][i], yte)
            row[f"{name}_probe_auc"], row[f"{name}_probe_eer"] = r["test_auc"], eer(yte, s)[0]
        table.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    json.dump(table, open(ROOT / "results/bitrate_sweep.json", "w"), indent=2)
    print("SWEEP_DONE")


if __name__ == "__main__":
    main()
