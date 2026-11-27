"""Permanently delete dataset parts and caches that no kept experiment needs (approved by the user, 2026-11-26).

Deletes:
  * data/raw/wavefake, data/raw/partialspoof         (downloaded, never used)
  * everything inside data/cache                     (regenerable compressed copies; .gitkeep kept)
  * audio in ASVspoof 5 / Fake-or-Real / LibriSeVoc that no experiment manifest references
  * ASVspoof 2019 dev / eval audio except the files the experiments sampled
    (dev pilot 600, eval pilot 600, eval 1,000 test set). ASVspoof 2019 train is kept whole.
Never touches: models, results, code, MLAAD, In-the-Wild, SpoofCeleb, protocols / metadata files.
Only paths inside <project>/data are ever deleted. Sizes are measured first, then deleted; all logged to
results/cleanup_log.txt.
"""
import csv, os, shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, RAW = ROOT / "data", ROOT / "data" / "raw"
LOG = ROOT / "results" / "cleanup_log.txt"
AUDIO = {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".opus"}


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


def key(p):
    return os.path.normcase(os.path.abspath(str(p)))


def walk(d):
    for dp, _, fs in os.walk(d):
        for f in fs:
            yield Path(dp) / f


def size(paths):
    n = b = 0
    for p in paths:
        try:
            b += p.stat().st_size; n += 1
        except OSError:
            pass
    return n, b


def sample_split(rows, n_tr, n_va, n_te, seed):          # identical to experiments/run_phase2_pilot.py
    rng = np.random.default_rng(seed); out = []
    for lab in (1, 0):
        pool = [r for r in rows if int(r["label"]) == lab]
        idx = rng.permutation(len(pool))
        out += [pool[i] for i in idx[:n_tr + n_va + n_te]]
    return out


def keep_set():
    keep = set()
    for m in (DATA / "manifests").glob("exp0*.csv"):
        for r in csv.DictReader(open(m, encoding="utf-8")):
            keep.add(key(r["path"]))
    dev = list(csv.DictReader(open(DATA / "manifests/asvspoof2019_dev.csv")))
    ev = [r for r in csv.DictReader(open(DATA / "manifests/asvspoof2019_eval.csv")) if r["attack"] not in {"A16", "A19"}]
    for r in sample_split(dev, 150, 50, 100, 42) + sample_split(ev, 150, 50, 100, 42) + sample_split(ev, 0, 0, 500, 7):
        keep.add(key(r["path"]))
    return keep


def main():
    LOG.write_text("")
    before = shutil.disk_usage(ROOT).free
    log(f"free disk before: {before / 1e9:.1f} GB")
    keep = keep_set()
    log(f"files referenced by experiments (kept): {len(keep)}")

    whole = [RAW / "wavefake", RAW / "partialspoof"]
    prune_dirs = [RAW / "asvspoof5" / "flac_T", RAW / "fake_or_real", RAW / "librisevoc",
                  RAW / "LA" / "ASVspoof2019_LA_dev", RAW / "LA" / "ASVspoof2019_LA_eval"]
    plan = []                                           # (label, [files])
    for d in whole:
        if d.exists():
            plan.append((str(d.relative_to(ROOT)), list(walk(d))))
    cache = [p for p in walk(DATA / "cache") if p.name != ".gitkeep"]
    plan.append(("data/cache (all compressed copies)", cache))
    for d in prune_dirs:
        if d.exists():
            files = [p for p in walk(d) if p.suffix.lower() in AUDIO and key(p) not in keep]
            plan.append((f"{d.relative_to(ROOT)} (unused clips only)", files))

    log("\n=== measured before deleting ===")
    tot = 0
    for lab, files in plan:
        n, b = size(files); tot += b
        log(f"  {lab:55} {n:>8} files  {b / 1e9:7.2f} GB")
    log(f"  {'TOTAL':55} {'':>8}        {tot / 1e9:7.2f} GB")

    log("\n=== deleting ===")
    for lab, files in plan:
        bad = 0
        for p in files:
            assert key(p).startswith(key(DATA)), p
            try:
                p.unlink()
            except OSError:
                bad += 1
        log(f"  {lab}: done{'' if not bad else f' ({bad} could not be deleted)'}")
    for d in whole:
        shutil.rmtree(d, ignore_errors=True)
    for d in [DATA / "cache"] + prune_dirs:                 # remove folders left empty
        for dp, dn, fn in sorted(os.walk(d), key=lambda t: -len(t[0])):
            if dp != str(DATA / "cache") and not os.listdir(dp):
                os.rmdir(dp)

    log("\n=== check ===")
    left = [p for p in walk(DATA / "cache") if p.name != ".gitkeep"]
    log(f"  files left in data/cache: {len(left)}")
    for d in whole:
        log(f"  {d.relative_to(ROOT)} exists: {d.exists()}")
    after = shutil.disk_usage(ROOT).free
    log(f"free disk after: {after / 1e9:.1f} GB  (freed {(after - before) / 1e9:.1f} GB)")
    log("CLEANUP_DONE")


if __name__ == "__main__":
    main()
