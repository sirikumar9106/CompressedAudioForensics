"""EXP-007a: why does a real phone recording get classified as AI?

Uses the EXACT app pipeline (src/app/server.py windows + WavLM + method F) on:
  1. 50 real + 50 fake ASVspoof 2019 LA eval files (unseen attacks), as-is vs leading/trailing silence trimmed
  2. every file in data/real_world/: per-window P(real) under every calibration profile,
     as-is and with 1 s of silence padded at both ends
Writes results/exp007a.json
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.app import server  # noqa: E402

SR = 16000


def trim_silence(x, db=-35.0, frame=320):
    """Cut leading/trailing frames whose RMS is > |db| below the loudest frame."""
    n = len(x) // frame
    if n < 3:
        return x
    rms = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, 1)) + 1e-9
    keep = np.where(20 * np.log10(rms / rms.max()) > db)[0]
    return x if len(keep) == 0 else x[keep[0] * frame:(keep[-1] + 1) * frame]


def decode(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def score(x, calib):
    art = server.STATE["art"]
    probe = art["probe_U"] if calib == "unknown" else art["probe_F"]
    a, b = art["cal"][calib]
    W = server.windows(x)
    s = probe.decision_function(server.embed([w for _, _, w in W]))
    p = 1 / (1 + np.exp(-(a * s + b)))
    return p, s


def main():
    server._load()
    out = {"asvspoof": [], "real_world": []}
    rows = [r for r in csv.DictReader(open(ROOT / "data/manifests/asvspoof2019_eval.csv")) if r["attack"] not in {"A16", "A19"}]
    rng = np.random.default_rng(11)
    pick = [r for lab in ("1", "0") for r in rng.choice([r for r in rows if r["label"] == lab], 50, replace=False)]
    print("[1/2] ASVspoof eval files through the app pipeline (clean profile)", flush=True)
    for r in pick:
        x = decode(r["path"])
        xt = trim_silence(x)
        p0, _ = score(x, "clean"); p1, _ = score(xt, "clean")
        out["asvspoof"].append({"file": r["file_id"], "label": int(r["label"]), "sec": len(x) / SR, "sec_trimmed": len(xt) / SR,
                                "p_real": float(np.median(p0)), "p_real_trimmed": float(np.median(p1))})
    A = out["asvspoof"]
    for lab, nm in ((1, "REAL"), (0, "FAKE")):
        g = [a for a in A if a["label"] == lab]
        ok = lambda k: np.mean([(a[k] >= 0.5) == (lab == 1) for a in g]) * 100
        print(f"  {nm}: correct as-is {ok('p_real'):.0f}%   correct after trimming silence {ok('p_real_trimmed'):.0f}%   "
              f"mean silence removed {np.mean([a['sec'] - a['sec_trimmed'] for a in g]):.2f}s", flush=True)

    print("[2/2] real-world recordings in data/real_world/", flush=True)
    files = sorted(p for p in (ROOT / "data/real_world").glob("*") if p.is_file() and p.suffix.lower() != ".gitkeep")
    for f in files:
        x = decode(f)
        meta = server.probe_file(str(f))
        pad = np.concatenate([np.zeros(SR, np.float32), x, np.zeros(SR, np.float32)])
        res = {"file": f.name, "meta": meta, "sec": len(x) / SR, "profiles": {}}
        for c in ["clean", "64k", "32k", "16k", "8k", "unknown"]:
            p, s = score(x, c); pp, _ = score(pad, c)
            res["profiles"][c] = {"median_p_real": float(np.median(p)), "windows": [round(float(v), 3) for v in p],
                                  "median_p_real_padded": float(np.median(pp)), "median_raw_score": float(np.median(s))}
        out["real_world"].append(res)
        print(f"  {f.name}  ({res['sec']:.1f}s, {meta.get('codec')} {meta.get('bitrate_kbps')} kbps)", flush=True)
        for c, v in res["profiles"].items():
            print(f"     {c:8} P(real) median {v['median_p_real']*100:5.1f}%   with 1s silence pads {v['median_p_real_padded']*100:5.1f}%   "
                  f"windows {v['windows'][:8]}", flush=True)
    if not files:
        print("  (no files found — put recordings in data\\real_world\\)", flush=True)
    json.dump(out, open(ROOT / "results/exp007a.json", "w"), indent=2)
    print("EXP007A_DONE", flush=True)


if __name__ == "__main__":
    main()
