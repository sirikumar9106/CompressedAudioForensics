"""EXP-008 data: merge ASVspoof 5 (train part aa) + Fake-or-Real (norm + rerec) + LibriSeVoc into one
homogeneous, balanced manifest, and pre-encode one random codec copy per clip.

Rules learned from EXP-001..007:
  * real/fake balanced 50/50 INSIDE every dataset and split (no "this dataset = fake" shortcut)
  * per-dataset cap so no single corpus dominates
  * speaker-/file-disjoint train vs val split
  * silence left untouched; 4 s random crops happen at training time (kills FoR's length bias)
  * test-only sets (In-the-Wild, PartialSpoof, WaveFake, ASVspoof 2019) are never read here
Output: data/manifests/exp008_{train,val}.csv  (path,label,dataset,codec_path)
"""
import csv
import hashlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.offline_cache import degrade_file  # noqa: E402

RAW = ROOT / "data" / "raw"
CAP_TRAIN, CAP_VAL = 4000, 400          # per dataset per class
CODECS = [("mp3", b) for b in ("64k", "32k", "16k", "8k")] + [("opus", b) for b in ("64k", "32k", "16k", "8k")] + \
         [("aac", b) for b in ("64k", "32k", "16k")]
AUDIO = {".wav", ".flac", ".mp3", ".ogg"}
rng = random.Random(8)


def bucket(key, val_frac=0.1):
    return "val" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 < val_frac * 1000 else "train"


def asvspoof5():
    d = RAW / "asvspoof5"
    have = {p.stem: p for p in (d / "flac_T").glob("*.flac")}
    rows = []
    for line in open(d / "ASVspoof5.train.tsv"):
        f = line.split()
        if f[1] in have:
            rows.append({"path": str(have[f[1]]), "label": int(f[8] == "bonafide"), "split": bucket("a5" + f[0])})
    return rows


def fake_or_real():
    rows = []
    for ver in ("for-norm", "for-rerec"):
        for p in (RAW / "fake_or_real").rglob("*"):
            if p.suffix.lower() not in AUDIO or ver not in str(p).lower():
                continue
            parts = [s.lower() for s in p.parts]
            lab = 1 if "real" in parts else 0 if "fake" in parts else None
            if lab is None:
                continue
            split = "val" if ("validation" in parts or "testing" in parts) else "train"
            rows.append({"path": str(p), "label": lab, "split": split})
    return rows


def librisevoc():
    rows = []
    for p in (RAW / "librisevoc").rglob("*"):
        if p.suffix.lower() not in AUDIO:
            continue
        parts = [s.lower() for s in p.parts]
        lab = 1 if ("gt" in parts or "real" in parts or "original" in parts) else 0
        spk = p.stem.split("_")[0]
        rows.append({"path": str(p), "label": lab, "split": bucket("lsv" + spk)})
    return rows


def balance(rows, split, cap):
    r = [x for x in rows if x["split"] == split]
    real = [x for x in r if x["label"] == 1]; fake = [x for x in r if x["label"] == 0]
    n = min(len(real), len(fake), cap)
    return rng.sample(real, n) + rng.sample(fake, n)


def main():
    sources = {"asvspoof5": asvspoof5, "fake_or_real": fake_or_real, "librisevoc": librisevoc}
    out = {"train": [], "val": []}
    for name, fn in sources.items():
        try:
            rows = fn()
        except FileNotFoundError:
            rows = []
        n_real = sum(r["label"] for r in rows)
        print(f"{name:13} found {len(rows):6d} clips ({n_real} real / {len(rows) - n_real} fake)", flush=True)
        if not rows:
            print(f"   WARNING: {name} missing - skipped", flush=True)
            continue
        for split, cap in (("train", CAP_TRAIN), ("val", CAP_VAL)):
            b = balance(rows, split, cap)
            for r in b:
                r["dataset"] = name
            out[split] += b
            print(f"   {split:5}: {len(b)} clips (balanced 50/50)", flush=True)

    cache = ROOT / "data" / "cache" / "exp008"
    jobs = []
    for split in out:
        for r in out[split]:
            c, b = rng.choice(CODECS)
            dst = cache / f"{c}_{b}" / (hashlib.md5(r["path"].encode()).hexdigest()[:16] + ".wav")
            r["codec_path"] = str(dst)
            if not dst.exists():
                jobs.append((r["path"], dst, c, b))
    print(f"encoding {len(jobs)} codec copies (mp3/opus/aac, 8-64 kbps; same distribution for real and fake)", flush=True)
    done = [0]

    def job(j):
        try:
            degrade_file(j[0], j[1], j[2], j[3])
        except Exception:
            pass
        done[0] += 1
        if done[0] % 2000 == 0:
            print(f"   {done[0]}/{len(jobs)}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(job, jobs))
    for split, rows in out.items():
        rows = [r for r in rows if Path(r["codec_path"]).exists()]
        rng.shuffle(rows)
        p = ROOT / "data" / "manifests" / f"exp008_{split}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "dataset", "codec_path"], extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"wrote {p.name}: {len(rows)} clips", flush=True)
    print("EXP008_DATA_DONE", flush=True)


if __name__ == "__main__":
    main()
