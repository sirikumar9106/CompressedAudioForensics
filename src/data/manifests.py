"""Build CSV manifests (file_id,path,label,attack) from ASVspoof protocol files. label: 1=bonafide, 0=spoof.

ASVspoof 2019 LA protocol line:  SPEAKER FILE_ID - ATTACK KEY   (KEY in {bonafide, spoof})
Usage:
  python -m src.data.manifests --protocol <ASVspoof2019.LA.cm.dev.trl.txt> --audio_dir <.../ASVspoof2019_LA_dev/flac> \
      --out data/manifests/asvspoof2019_dev.csv
"""
import argparse
import csv
from pathlib import Path


def build_manifest(protocol: str, audio_dir: str, out: str, ext: str = ".flac") -> int:
    rows, missing = [], 0
    for line in open(protocol):
        p = line.split()
        if len(p) < 5:
            continue
        fid, attack, key = p[1], p[3], p[4]
        path = Path(audio_dir) / f"{fid}{ext}"
        if not path.exists():
            missing += 1
            continue
        rows.append({"file_id": fid, "path": str(path), "label": int(key == "bonafide"), "attack": attack})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_id", "path", "label", "attack"])
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} files ({sum(r['label'] for r in rows)} bonafide), {missing} missing audio")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", default=".flac")
    a = ap.parse_args()
    build_manifest(a.protocol, a.audio_dir, a.out, a.ext)
