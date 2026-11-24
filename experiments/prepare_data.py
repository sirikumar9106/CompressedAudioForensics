"""Build ASVspoof 2019 LA manifests and (optionally) the full MP3 cache for training.

Usage:
  python experiments/prepare_data.py                    # manifests only (enough for the pilot)
  python experiments/prepare_data.py --cache train dev  # + 32 kbps MP3 cache for Phases 3-4 (~25-40 min)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.manifests import build_manifest  # noqa: E402

LA = ROOT / "data/raw/LA"
SPLITS = {"train": "trn", "dev": "trl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--la_root", default=str(LA))
    ap.add_argument("--cache", nargs="*", default=[], choices=list(SPLITS))
    ap.add_argument("--codec", default="mp3")
    ap.add_argument("--bitrate", default="32k")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    la = Path(a.la_root)
    for split, suf in SPLITS.items():
        proto = la / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.{suf}.txt"
        audio = la / f"ASVspoof2019_LA_{split}" / "flac"
        if not proto.exists() or not audio.exists():
            print(f"MISSING for {split}: {proto if not proto.exists() else audio}")
            continue
        build_manifest(str(proto), str(audio), str(ROOT / f"data/manifests/asvspoof2019_{split}.csv"))
    for split in a.cache:
        subprocess.run([sys.executable, "-m", "src.data.offline_cache", "--manifest",
                        f"data/manifests/asvspoof2019_{split}.csv", "--codec", a.codec, "--bitrate", a.bitrate,
                        "--workers", str(a.workers)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
