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

import os

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DL = RAW / "_downloads"
Z5 = "https://zenodo.org/records/14498691/files/{}?download=1"
HF = "https://huggingface.co/datasets/{}/resolve/main/{}"
FOR = "https://bil.eecs.yorku.ca/share/{}"
ZPS = "https://zenodo.org/records/5766198/files/{}?download=1"
ZWF = "https://zenodo.org/records/5642694/files/{}?download=1"
LIBRISEVOC_GDRIVE_ID = "1NXF9w0YxzVjIAwGm_9Ku7wfLHVbsT7aG"
# EXP-008 additions: (name, url, filename, extract-to)
EXTRA = [
    ("FoR for-norm (train)", FOR.format("for-norm.tar.gz"), "for-norm.tar.gz", RAW / "fake_or_real"),
    ("FoR for-rerec (train)", FOR.format("for-rerec.tar.gz"), "for-rerec.tar.gz", RAW / "fake_or_real"),
    ("PartialSpoof protocols (test)", ZPS.format("database_protocols.tar.gz"), "ps_protocols.tar.gz", RAW / "partialspoof"),
    ("PartialSpoof segment labels (test)", ZPS.format("database_segment_labels_v1.2.tar.gz"), "ps_segment_labels.tar.gz", RAW / "partialspoof"),
    ("PartialSpoof eval audio (test)", ZPS.format("database_eval.tar.gz"), "ps_eval.tar.gz", RAW / "partialspoof"),
    ("WaveFake generated audio (test)", ZWF.format("generated_audio.zip"), "wavefake_generated_audio.zip", RAW / "wavefake"),
    ("LJSpeech real audio for WaveFake (test)", "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", "LJSpeech-1.1.tar.bz2", RAW / "wavefake"),
]
JOBS = [
    ("ASVspoof5 protocols", Z5.format("ASVspoof5_protocols.tar.gz"), "ASVspoof5_protocols.tar.gz"),
    ("In-the-Wild", HF.format("mueller91/In-The-Wild", "release_in_the_wild.zip"), "release_in_the_wild.zip"),
    ("ASVspoof5 train part aa", Z5.format("flac_T_aa.tar"), "flac_T_aa.tar"),
] + ([(f"SpoofCeleb part {s}", HF.format("jungjee/spoofceleb", f"spoofceleb.tar.gz{s}"), f"spoofceleb.tar.gz{s}")
       for s in ("aa", "ab", "ac")] if os.environ.get("HF_TOKEN") else [])  # gated: needs approved HF account + token


def download(name, url, fn):
    dst = DL / fn
    done = DL / (fn + ".done")
    if done.exists() and dst.exists():
        print(f"[skip] {name} already downloaded", flush=True)
        return dst
    pos = dst.stat().st_size if dst.exists() else 0
    for attempt in range(20):
        try:
            h = {"Range": f"bytes={pos}-"} if pos else {}
            if "huggingface.co" in url and os.environ.get("HF_TOKEN"):
                h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
            with requests.get(url, headers=h, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416:
                    break
                if r.status_code in (401, 403):
                    print(f"[gated] {name}: access denied (HTTP {r.status_code}) - accept the dataset terms and set HF_TOKEN", flush=True)
                    return None
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
    else:
        print(f"[failed] {name}: giving up after 20 attempts", flush=True)
        return None
    done.write_text("ok")
    print(f"[done] {name}  ({dst.stat().st_size/1e9:.2f} GB)", flush=True)
    return dst


def extract(path, out):
    out.mkdir(parents=True, exist_ok=True)
    flag = out / f".extracted_{path.name}"
    if flag.exists():
        return
    print(f"[extract] {path.name} -> {out}", flush=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
    else:
        with tarfile.open(path) as t:  # auto-detects gz / bz2
            t.extractall(out, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(out)
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


def download_librisevoc():
    dst = DL / "LibriSeVoc.zip"
    done = DL / "LibriSeVoc.zip.done"
    if not (done.exists() and dst.exists()):
        import gdown
        print("[download] LibriSeVoc from Google Drive (train)", flush=True)
        gdown.download(id=LIBRISEVOC_GDRIVE_ID, output=str(dst), quiet=False, resume=True)
        if not dst.exists() or dst.stat().st_size < 1e6:
            print("[failed] LibriSeVoc: Google Drive refused (quota/permission). Download it manually from the authors' "
                  "GitHub link and place it at data/raw/_downloads/LibriSeVoc.zip", flush=True)
            return
        done.write_text("ok")
    out = RAW / "librisevoc"
    flag = out / ".extracted"
    if not flag.exists():
        out.mkdir(parents=True, exist_ok=True)
        print("[extract] LibriSeVoc", flush=True)
        if zipfile.is_zipfile(dst):
            zipfile.ZipFile(dst).extractall(out)
        else:
            tarfile.open(dst).extractall(out)
        flag.write_text("ok")


def main():
    DL.mkdir(parents=True, exist_ok=True)
    got = {fn: download(name, url, fn) for name, url, fn in JOBS}
    for fn, out in [("ASVspoof5_protocols.tar.gz", RAW / "asvspoof5"), ("release_in_the_wild.zip", RAW / "in_the_wild"),
                    ("flac_T_aa.tar", RAW / "asvspoof5")]:
        if got.get(fn):
            extract(got[fn], out)
    sc = [got.get(f"spoofceleb.tar.gz{s}") for s in ("aa", "ab", "ac")]
    if all(sc):
        extract_partial_spoofceleb(sc, RAW / "spoofceleb")
    else:
        print("[skip] SpoofCeleb not available yet (gated; request access on Hugging Face)", flush=True)
    # ---- EXP-008 datasets: training data first, test-only data last ----
    order = EXTRA[:2]
    for name, url, fn, out in order:
        f = download(name, url, fn)
        if f:
            extract(f, out)
    download_librisevoc()
    for name, url, fn, out in EXTRA[2:]:
        f = download(name, url, fn)
        if f:
            extract(f, out)
    print("ALL_DOWNLOADS_DONE  (archives kept in data/raw/_downloads; delete them once you are happy)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
