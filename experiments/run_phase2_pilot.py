"""Phase 2: frozen-backbone GO/NO-GO pilot on ASVspoof 2019 LA DEV only (never the locked set).

Usage:  python experiments/run_phase2_pilot.py --config config/pilot_config.yaml
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audio_processing import load_audio, preprocess  # noqa: E402
from src.data.offline_cache import cache_path, degrade_file  # noqa: E402
from src.evaluation.metrics import roc_auc  # noqa: E402
from src.models.backbones import RawNet2Encoder, load_pretrained_rawnet2  # noqa: E402
from src.models.degradation_engine import get_perturbation  # noqa: E402
from src.utils.benchmark_guard import assert_not_locked  # noqa: E402


def sample_split(rows, cfg, seed):
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for lab in (1, 0):
        pool = [r for r in rows if int(r["label"]) == lab]
        idx = rng.permutation(len(pool))
        n_tr, n_va, n_te = cfg["n_train"], cfg["n_val"], cfg["n_test"]
        need = n_tr + n_va + n_te
        if len(pool) < need:
            raise SystemExit(f"Not enough files for label {lab}: {len(pool)} < {need}")
        sel = [pool[i] for i in idx[:need]]
        out["train"] += sel[:n_tr]; out["val"] += sel[n_tr:n_tr + n_va]; out["test"] += sel[n_tr + n_va:]
    return out


@torch.no_grad()
def extract(rows, encoder, cfg, device):
    """Per file: pooled stats for base / perturbed under each perturbation, plus confound controls."""
    feats = {p: {"base": [], "pert": [], "delta": []} for p in cfg["perturbations"]}
    rms, dur, labels = [], [], []
    for r in rows:
        dst = cache_path(cfg["cache_root"], cfg["codec"], cfg["bitrate"], r["file_id"])
        if not dst.exists():
            degrade_file(r["path"], dst, cfg["codec"], cfg["bitrate"])
        raw = load_audio(str(dst))
        dur.append(len(raw) / 16000); rms.append(float(np.sqrt(np.mean(raw ** 2))))
        y = torch.from_numpy(preprocess(raw)).unsqueeze(0).to(device)
        z = encoder(y)
        for p in cfg["perturbations"]:
            zp = encoder(get_perturbation(p)(y))
            dz = z - zp
            mz, mzp = z.mean(1), zp.mean(1)
            cos = 1 - torch.cosine_similarity(mz, mzp, dim=-1)
            delta = torch.cat([dz.mean(1), dz.std(1, unbiased=False), dz.norm(dim=-1).mean(1, keepdim=True),
                               cos.unsqueeze(-1)], -1)
            feats[p]["base"].append(mz.cpu().numpy()[0]); feats[p]["pert"].append(mzp.cpu().numpy()[0])
            feats[p]["delta"].append(delta.cpu().numpy()[0])
        labels.append(int(r["label"]))
    np_feats = {p: {k: np.stack(v) for k, v in d.items()} for p, d in feats.items()}
    return np_feats, np.array(rms), np.array(dur), np.array(labels)


def probe(Xtr, ytr, Xva, yva, Xte, yte):
    best, best_c = -1, None
    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(Xtr, ytr)
        a = roc_auc(yva, m.decision_function(Xva))
        if a > best:
            best, best_c = a, c
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=5000)).fit(Xtr, ytr)
    s = m.decision_function(Xte)
    return {"C": best_c, "val_auc": best, "test_auc": roc_auc(yte, s)}, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/pilot_config.yaml"))
    cfg = yaml.safe_load(open(ap.parse_args().config))
    assert_not_locked(cfg["dev_manifest"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    rows = list(csv.DictReader(open(cfg["dev_manifest"], newline="")))
    excl = set(cfg.get("exclude_attacks", []))
    rows = [r for r in rows if r["attack"] not in excl]
    split = sample_split(rows, cfg, cfg["seed"])
    encoder = RawNet2Encoder(load_pretrained_rawnet2(cfg["pretrained"]), proj_dim=None).to(device).eval()

    data = {k: extract(v, encoder, cfg, device) for k, v in split.items()}
    res = {"device": device, "n": {k: len(v) for k, v in split.items()}, "probes": {}}
    for p in cfg["perturbations"]:
        res["probes"][p] = {}
        for fs in ["delta", "base", "pert", "full"]:
            def X(k):
                f = data[k][0][p]
                return np.hstack([f["base"], f["pert"], f["delta"]]) if fs == "full" else f[fs]
            r, s = probe(X("train"), data["train"][3], X("val"), data["val"][3], X("test"), data["test"][3])
            if fs == "delta":  # confound check on held-out scores
                r["corr_rms"] = float(abs(np.corrcoef(s, data["test"][1])[0, 1]))
                r["corr_duration"] = float(abs(np.corrcoef(s, data["test"][2])[0, 1]))
            res["probes"][p][fs] = r
    yte = data["test"][3]
    res["controls"] = {"rms_auc": roc_auc(yte, data["test"][1]), "duration_auc": roc_auc(yte, data["test"][2])}

    g = cfg["gate"]
    d = res["probes"]["mulaw8"]["delta"]
    noise = res["probes"]["gaussian"]["delta"]["test_auc"]
    base = res["probes"]["mulaw8"]["base"]["test_auc"]
    confound_ok = max(d["corr_rms"], d["corr_duration"]) < g["max_confound_r"]
    if d["test_auc"] < g["nogo_auc"] or d["test_auc"] < base:
        verdict = "NO-GO"
    elif d["test_auc"] >= g["go_auc"] and d["test_auc"] - noise >= g["min_margin_over_noise"] and confound_ok:
        verdict = "GO"
    elif d["test_auc"] >= g["conditional_auc"]:
        verdict = "CONDITIONAL GO"
    else:
        verdict = "INCONCLUSIVE (0.60-0.65 or failed noise/confound check) - review with user"
    res["verdict"] = verdict
    out = ROOT / "results" / f"{cfg.get('result_name', 'phase2_pilot')}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nVERDICT: {verdict}   (saved -> {out})\nSTOP: report to user before Phase 3.")


if __name__ == "__main__":
    main()
