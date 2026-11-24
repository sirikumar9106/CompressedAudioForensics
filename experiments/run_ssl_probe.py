"""EXP-004: frozen SSL front ends (WavLM) vs RawNet2 / Model A, fair cross-attack protocol.

Probe TRAIN/VAL = ASVspoof 2019 LA train (attacks A01-A06, same as Model A's training data).
TEST = the EXP-002/003 200-file unseen-attack set (2019 LA eval, A07-A18 w/o A16/A19).
Conditions: clean, MP3 32k, MP3 8k (probe trained and tested on the same condition).

Usage: python experiments/run_ssl_probe.py [--models microsoft/wavlm-base-plus microsoft/wavlm-large]
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from run_phase2_pilot import sample_split  # noqa: E402
from src.data.audio_processing import load_audio, preprocess  # noqa: E402
from src.data.offline_cache import cache_path, degrade_file  # noqa: E402
from src.evaluation.metrics import eer, roc_auc  # noqa: E402
from src.models.backbones import load_pretrained_rawnet2  # noqa: E402

LOGF = ROOT / "results/ssl_probe_log.txt"


def log(msg, **kw):
    print(msg)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


CONDS = [None, "32k", "8k"]
N_TRAIN, N_VAL = 300, 100  # per class, from LA train


def path(r, br, cache_root):
    return r["path"] if br is None else str(cache_path(cache_root, "mp3", br, r["file_id"]))


def fit_probe(Xtr, ytr, Xva, yva):
    best = (-1, None)
    for c in [0.001, 0.01, 0.1, 1.0]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(Xtr, ytr)
        a = roc_auc(yva, m.decision_function(Xva))
        if a > best[0]:
            best = (a, m)
    return best


@torch.no_grad()
def wavlm_feats(model, norm, wavs, device):
    """Returns [N, L, 2D] mean+std pooled per hidden layer."""
    out = []
    for w in wavs:
        x = torch.from_numpy(w).unsqueeze(0).to(device)
        if norm:
            x = (x - x.mean()) / (x.std() + 1e-7)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            hs = model(x, output_hidden_states=True).hidden_states
        h = torch.stack(hs, 1)[0].float()  # [L, T, D]
        out.append(torch.cat([h.mean(1), h.std(1)], -1).cpu().numpy())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["microsoft/wavlm-base-plus", "microsoft/wavlm-large"])
    a = ap.parse_args()
    LOGF.write_text("")
    from transformers import AutoFeatureExtractor, WavLMModel
    cfg = yaml.safe_load(open(ROOT / "config/pilot_unseen_attacks.yaml"))
    cr, device = str(ROOT / cfg["cache_root"]), ("cuda" if torch.cuda.is_available() else "cpu")

    ev = [r for r in csv.DictReader(open(ROOT / cfg["dev_manifest"])) if r["attack"] not in {"A16", "A19"}]
    test = sample_split(ev, cfg, cfg["seed"])["test"]
    tr_all = list(csv.DictReader(open(ROOT / "data/manifests/asvspoof2019_train.csv")))
    tcfg = {"n_train": N_TRAIN, "n_val": N_VAL, "n_test": 0}
    s = sample_split(tr_all, tcfg, cfg["seed"])
    train, val = s["train"], s["val"]
    rows = train + val + test

    jobs = [(r, br) for br in CONDS if br for r in rows if not cache_path(cr, "mp3", br, r["file_id"]).exists()]
    log(f"encoding {len(jobs)} files")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: degrade_file(j[0]["path"], cache_path(cr, "mp3", j[1], j[0]["file_id"]), "mp3", j[1]), jobs))

    rawnet = load_pretrained_rawnet2(str(ROOT / cfg["pretrained"])).to(device).eval()
    ssl = {}
    for name in a.models:
        fe = AutoFeatureExtractor.from_pretrained(name)
        ssl[name] = (WavLMModel.from_pretrained(name).to(device).eval(), bool(getattr(fe, "do_normalize", False)))
        log(f"loaded {name}")

    y = {k: np.array([int(r["label"]) for r in v]) for k, v in [("tr", train), ("va", val), ("te", test)]}
    results = []
    for br in CONDS:
        wav = {k: [preprocess(load_audio(path(r, br, cr))) for r in v] for k, v in [("tr", train), ("va", val), ("te", test)]}
        row = {"condition": "clean" if br is None else f"mp3_{br}"}
        with torch.no_grad():  # Model A (official classifier) and RawNet2 pooled embedding
            A, R = {}, {}
            for k, ws in wav.items():
                sc, em = [], []
                for w in ws:
                    x = torch.from_numpy(w).unsqueeze(0).to(device)
                    lg = rawnet(x); sc.append((lg[0, 1] - lg[0, 0]).item())
                    em.append(rawnet.frame_features(x).mean(-1)[0].cpu().numpy())
                A[k], R[k] = np.array(sc), np.stack(em)
        row["modelA"] = (roc_auc(y["te"], A["te"]), eer(y["te"], A["te"])[0])
        _, m = fit_probe(R["tr"], y["tr"], R["va"], y["va"])
        st = m.decision_function(R["te"]); row["rawnet2_probe"] = (roc_auc(y["te"], st), eer(y["te"], st)[0])
        for name, (mdl, norm) in ssl.items():
            F = {k: wavlm_feats(mdl, norm, ws, device) for k, ws in wav.items()}
            short = name.split("/")[-1]
            # variant 1: last layer; variant 2: all-layer average; variant 3: best single layer chosen on VAL
            cands = {"last": lambda f: f[:, -1], "layeravg": lambda f: f.mean(1)}
            for vn, fn in cands.items():
                _, m = fit_probe(fn(F["tr"]), y["tr"], fn(F["va"]), y["va"])
                st = m.decision_function(fn(F["te"])); row[f"{short}_{vn}"] = (roc_auc(y["te"], st), eer(y["te"], st)[0])
            best = max(range(F["tr"].shape[1]), key=lambda l: fit_probe(F["tr"][:, l], y["tr"], F["va"][:, l], y["va"])[0])
            _, m = fit_probe(F["tr"][:, best], y["tr"], F["va"][:, best], y["va"])
            st = m.decision_function(F["te"][:, best])
            row[f"{short}_bestlayer(L{best})"] = (roc_auc(y["te"], st), eer(y["te"], st)[0])
        results.append(row)
        log(json.dumps({k: [round(x, 4) for x in v] if isinstance(v, tuple) else v for k, v in row.items()}))
    json.dump(results, open(ROOT / "results/ssl_probe.json", "w"), indent=2)
    log("SSL_DONE")


if __name__ == "__main__":
    main()
