"""Download + extract the real-world datasets (resumable; safe to re-run).

  1. ASVspoof 5 protocols + training part flac_T_aa   (Zenodo 14498691, ODC-By)          -> data/raw/asvspoof5/
  2. In-the-Wild release_in_the_wild.zip              (HF mueller91/In-The-Wild, CC-BY-SA) -> data/raw/in_the_wild/  [TEST ONLY]
  3. SpoofCeleb first 3 of 27 split parts (15 GB)     (HF jungjee/spoofceleb)             -> data/raw/spoofceleb/
Deepfake-Eval-2024 is gated (accept terms on Hugging Face yourself; evaluation-only) and is not downloaded here.
"""
import gzip
import io
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DL = RAW / "_downloads"
Z5 = "https://zenodo.org/records/14498691/files/{}?download=1"
HF = "https://huggingface.co/datasets/{}/resolve/main/{}"
JOBS = [
    ("ASVspoof5 protocols", Z5.format("ASVspoof5_protocols.tar.gz"), "ASVspoof5_protocols.tar.gz"),
    ("In-the-Wild", HF.format("mueller91/In-The-Wild", "release_in_the_wild.zip"), "release_in_the_wild.zip"),
    ("ASVspoof5 train part aa", Z5.format("flac_T_aa.tar"), "flac_T_aa.tar"),
] + [(f"SpoofCeleb part {s}", HF.format("jungjee/spoofceleb", f"spoofceleb.tar.gz{s}"), f"spoofceleb.tar.gz{s}")
     for s in ("aa", "ab", "ac")]


def download(name, url, fn):
    dst = DL / fn
    done = DL / (fn + ".done")
    if done.exists():
        print(f"[skip] {name} already downloaded", flush=True)
        return dst
    pos = dst.stat().st_size if dst.exists() else 0
    for attempt in range(20):
        try:
            h = {"Range": f"bytes={pos}-"} if pos else {}
            with requests.get(url, headers=h, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                if pos and r.status_code == 200:  # server ignored Range -> restart
                    pos = 0
                total = pos + int(r.headers.get("Content-Length", 0))
                t0, got = time.time(), 0
                with open(dst, "ab" if pos else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk); got += len(chunk); pos += len(chunk)
                        if got % (64 << 20) < (1 << 20):
                            mbps = got / max(time.time() - t0, 1e-6) / 1e6
                            eta = (total - pos) / max(mbps * 1e6, 1) / 60
                            print(f"   {name}: {pos/1e9:6.2f} / {total/1e9:.2f} GB  {mbps:5.1f} MB/s  ETA {eta:5.0f} min", flush=True)
            break
        except Exception as e:
            print(f"   {name}: interrupted ({e.__class__.__name__}), retrying in 15 s [{attempt+1}/20]", flush=True)
            time.sleep(15)
            pos = dst.stat().st_size if dst.exists() else 0
    done.write_text("ok")
    print(f"[done] {name}  ({dst.stat().st_size/1e9:.2f} GB)", flush=True)
    return dst


def extract(path, out):
    out.mkdir(parents=True, exist_ok=True)
    flag = out / f".extracted_{path.name}"
    if flag.exists():
        return
    print(f"[extract] {path.name} -> {out}", flush=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
    else:
        with tarfile.open(path) as t:
            t.extractall(out)
    flag.write_text("ok")


class Chain(io.RawIOBase):
    """Read split archive parts as one continuous stream."""
    def __init__(self, parts):
        self.parts, self.f = list(parts), None
    def readable(self):
        return True
    def readinto(self, b):
        while True:
            if self.f is None:
                if not self.parts:
                    return 0
                self.f = open(self.parts.pop(0), "rb")
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close(); self.f = None


def extract_partial_spoofceleb(parts, out):
    """Stream-extract whatever complete files the first N parts contain (archive is truncated on purpose)."""
    out.mkdir(parents=True, exist_ok=True)
    print(f"[extract] SpoofCeleb from {len(parts)} parts (partial archive) -> {out}", flush=True)
    n = 0
    try:
        with tarfile.open(fileobj=io.BufferedReader(Chain(parts), 1 << 20), mode="r|gz") as t:
            for m in t:
                t.extract(m, out)
                n += m.isfile()
                if n % 5000 == 0 and m.isfile():
                    print(f"   {n} files extracted (latest: {m.name})", flush=True)
    except (EOFError, tarfile.ReadError, gzip.BadGzipFile, OSError) as e:
        print(f"   reached end of downloaded parts after {n} files ({e.__class__.__name__}) - expected for a subset", flush=True)
    print(f"[done] SpoofCeleb subset: {n} files", flush=True)


def main():
    DL.mkdir(parents=True, exist_ok=True)
    got = {fn: download(name, url, fn) for name, url, fn in JOBS}
    extract(got["ASVspoof5_protocols.tar.gz"], RAW / "asvspoof5")
    extract(got["release_in_the_wild.zip"], RAW / "in_the_wild")
    extract(got["flac_T_aa.tar"], RAW / "asvspoof5")
    extract_partial_spoofceleb([got[f"spoofceleb.tar.gz{s}"] for s in ("aa", "ab", "ac")], RAW / "spoofceleb")
    print("ALL_DOWNLOADS_DONE  (archives kept in data/raw/_downloads; delete them once you are happy)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
