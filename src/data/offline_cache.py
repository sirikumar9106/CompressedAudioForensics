"""Precompute simulated platform degradation y = D_platform(x) with FFmpeg -> 16 kHz mono WAV cache.

Usage:
  python -m src.data.offline_cache --manifest data/manifests/asvspoof2019_dev.csv --codec mp3 --bitrate 32k
"""
import argparse
import csv
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODECS = {  # name -> (ffmpeg encoder, container extension)
    "mp3": ("libmp3lame", "mp3"),
    "opus": ("libopus", "ogg"),
    "aac": ("aac", "m4a"),
}


def cache_path(cache_root: str, codec: str, bitrate: str, file_id: str) -> Path:
    return Path(cache_root) / f"{codec}_{bitrate}" / f"{file_id}.wav"


def degrade_file(src: str, dst: Path, codec: str, bitrate: str) -> None:
    enc, ext = CODECS[codec]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / f"enc.{ext}"
        sr = "48000" if codec == "opus" else "16000"  # Opus only supports 8/12/16/24/48k; 16k is fine too
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", sr,
                        "-c:a", enc, "-b:a", bitrate, str(tmp)], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(tmp), "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--codec", choices=list(CODECS), default="mp3")
    ap.add_argument("--bitrate", default="32k")
    ap.add_argument("--cache_root", default="data/cache")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if "evaluation_manifest" in a.manifest:
        raise SystemExit("Refusing: locked benchmark is cached only in Phase 6 (use benchmark_runner).")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    rows = list(csv.DictReader(open(a.manifest, newline="")))

    def job(r):
        dst = cache_path(a.cache_root, a.codec, a.bitrate, r["file_id"])
        if not dst.exists():
            degrade_file(r["path"], dst, a.codec, a.bitrate)

    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(job, rows), 1):
            if i % 500 == 0:
                print(f"{i}/{len(rows)}")
    print(f"Done: {len(rows)} files -> {Path(a.cache_root) / f'{a.codec}_{a.bitrate}'}")


if __name__ == "__main__":
    main()
