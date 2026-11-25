"""EXP-010 data: English slice of MLAAD (modern TTS / voice cloning fakes), N clips per TTS system.

Needs HF_TOKEN (read) and the MLAAD conditions accepted on huggingface.co/datasets/mueller91/MLAAD.
Resumable: files already on disk are skipped. Output: data/raw/mlaad/fake/en/<model>/...
"""
import os, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO, N_PER_MODEL = "mueller91/MLAAD", int(os.environ.get("MLAAD_N", 200))
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "mlaad"
tok = os.environ.get("HF_TOKEN")
if not tok:
    sys.exit("HF_TOKEN is not set. Run: setx HF_TOKEN <your read token>  and open a NEW terminal.")

api = HfApi(token=tok)
print("listing English files ...", flush=True)
files = [f.path for f in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="fake/en", recursive=True)
         if hasattr(f, "size")]
by_model = defaultdict(list)
for p in files:
    by_model[p.split("/")[2]].append(p)
want = []
for m, fs in sorted(by_model.items()):
    wav = sorted(p for p in fs if p.lower().endswith((".wav", ".flac", ".mp3")))
    want += [p for p in fs if p.endswith(".csv")] + wav[:N_PER_MODEL]
todo = [p for p in want if not (OUT / p).exists()]
print(f"{len(by_model)} TTS systems, {len(want)} files wanted, {len(todo)} still to download", flush=True)


def get(p):
    for k in range(4):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=OUT, token=tok)
            return True
        except Exception as e:
            err = e; time.sleep(3 * (k + 1))
    print("  failed:", p, repr(err)[:120], flush=True)
    return False


t0, ok = time.time(), 0
with ThreadPoolExecutor(8) as ex:
    for i, r in enumerate(as_completed([ex.submit(get, p) for p in todo]), 1):
        ok += r.result()
        if i % 200 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  ({ok} ok, {(time.time()-t0)/60:.1f} min)", flush=True)
(OUT / ".done").write_text(f"{len(want)} files\n") if ok == len(todo) else None
print("MLAAD_DONE" if ok == len(todo) else "MLAAD_INCOMPLETE - re-run to resume", flush=True)
