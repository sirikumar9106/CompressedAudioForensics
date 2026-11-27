"""SpoofCeleb (gated, HF: jungjee/spoofceleb) -> extract ONLY evaluation audio + protocol/metadata files.

The dataset ships as one 134 GB gzip tarball split into 27 parts (aa .. ba). This script streams the parts
in order straight into a tar reader, so the archive is never stored on disk: members whose path looks like
evaluation data or metadata are written to data/raw/spoofceleb/, everything else (training audio) is skipped.
Needs HF_TOKEN (read) with SpoofCeleb access. Not resumable mid-stream (a restart re-streams from part aa,
but already extracted files are skipped). Progress + a listing of the archive layout go to results/spoofceleb_stream_log.txt.
"""
import os, re, sys, tarfile, time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "spoofceleb"
LOG = ROOT / "results" / "spoofceleb_stream_log.txt"
TOK = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN not set")
PARTS = [f"spoofceleb.tar.gz{a}{b}" for a in "a" for b in "abcdefghijklmnopqrstuvwxyz"] + ["spoofceleb.tar.gzba"]
URL = "https://huggingface.co/datasets/jungjee/spoofceleb/resolve/main/{}"
KEEP = re.compile(r"(eval|protocol|metadata|\.csv$|\.txt$|\.tsv$|\.lst$|readme|license)", re.I)
ALL = "--all" in sys.argv          # keep EVERY file (full dataset, for training) instead of evaluation-only


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


class Chain:
    """File-like object that reads the 27 HTTP parts back to back."""
    def __init__(self):
        self.i, self.r, self.it, self.cur, self.pos, self.n, self.off, self.t0 = -1, None, None, b"", 0, 0, 0, time.time()
        self._next()

    def _open(self, offset):
        """(Re)open the current part from byte `offset`; keeps retrying for ~12 hours of network trouble."""
        h = {"Authorization": f"Bearer {TOK}"}
        if offset:
            h["Range"] = f"bytes={offset}-"
        for k in range(800):                       # ~12 h of patience: power cuts / Wi-Fi outages just pause the stream
            try:
                r = requests.get(URL.format(PARTS[self.i]), headers=h, stream=True, timeout=60)
                r.raise_for_status()
                if offset and r.status_code != 206:
                    raise IOError("server ignored Range request")
                self.r, self.it = r, r.iter_content(1 << 20)
                return
            except Exception as e:
                log(f"  network problem on {PARTS[self.i]} at {offset / 1e9:.2f} GB: {e!r}"[:200]); time.sleep(min(60, 5 * (k + 1)))
        raise SystemExit(f"cannot reach {PARTS[self.i]} - re-run train_voxtrace_wild.bat to start over (extracted files are kept)")

    def _next(self):
        self.i += 1
        if self.i >= len(PARTS):
            self.it = None; return
        self.off = 0
        self._open(0)
        log(f"part {self.i + 1}/{len(PARTS)} {PARTS[self.i]}  ({self.n / 1e9:.1f} GB streamed, "
            f"{(time.time() - self.t0) / 60:.0f} min)")

    def _chunk(self):
        """Next chunk of the current part; on a dropped connection resume the same part at the same byte."""
        while True:
            try:
                c = next(self.it)
                self.off += len(c)
                return c
            except StopIteration:
                raise
            except Exception as e:
                log(f"  connection dropped in {PARTS[self.i]} at {self.off / 1e9:.2f} GB ({type(e).__name__}) - resuming")
                time.sleep(5)
                self._open(self.off)

    def read(self, size=-1):
        # keep the current 1 MB chunk and an offset (no repeated copying of the whole buffer)
        out = []
        need = size if size >= 0 else float("inf")
        while need > 0:
            if self.pos >= len(self.cur):
                if self.it is None:
                    break
                try:
                    self.cur, self.pos = self._chunk(), 0
                except StopIteration:
                    self._next(); continue
            take = self.cur[self.pos:self.pos + int(min(need, len(self.cur) - self.pos))]
            self.pos += len(take); need -= len(take); out.append(take)
        data = b"".join(out)
        self.n += len(data)
        return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== new run {time.ctime()} (all={ALL}) =====\n")
    dirs, kept, seen = Counter(), 0, 0
    src = Chain()
    with tarfile.open(fileobj=src, mode="r|gz") as tar:
        for m in tar:
            seen += 1
            top = "/".join(m.name.split("/")[:3])
            dirs[top] += 1
            if seen <= 20 or seen % 20000 == 0:
                log(f"  [{seen}] {m.name}   ({src.n / 1e9:.1f} GB, {(time.time() - src.t0) / 60:.0f} min)")
            if m.isfile() and (ALL or KEEP.search(m.name)):
                dst = OUT / m.name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(dst.name + ".part")          # never leave a half-written file behind
                    with tar.extractfile(m) as fsrc, open(tmp, "wb") as f:
                        f.write(fsrc.read())
                    os.replace(tmp, dst)
                kept += 1
    log(f"archive members seen: {seen}, extracted: {kept}")
    log("folders (first 3 path levels, member counts):")
    for d, n in sorted(dirs.items()):
        log(f"  {d}: {n}")
    (OUT / (".full_extracted" if ALL else ".eval_extracted")).write_text(str(kept))
    log("SPOOFCELEB_FULL_DONE" if ALL else "SPOOFCELEB_EVAL_DONE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        log("CRASHED:\n" + traceback.format_exc())
        raise
