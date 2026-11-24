"""CompressedAudioForensics — local web app (single page + JSON API).

Run:  python -m uvicorn src.app.server:app --host 127.0.0.1 --port 7860
Uploaded audio is decoded in memory/temp, analysed, and deleted immediately.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.app import model_store  # noqa: E402
from src.data.audio_processing import N_SAMPLES, SR, peak_normalize  # noqa: E402

HOP = N_SAMPLES // 2          # 50% overlap
MAX_SECONDS = 600             # analyse at most 10 minutes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Typical default audio handling per source (estimates; a lossy file header overrides them).
PLATFORMS = {
    "whatsapp":  ("WhatsApp",            "16k",     "Voice notes: Opus, about 16 kbps"),
    "telegram":  ("Telegram",            "32k",     "Voice messages: Opus, about 32 kbps"),
    "signal":    ("Signal",              "32k",     "Voice notes: AAC, about 32 kbps"),
    "messenger": ("Facebook Messenger",  "32k",     "Audio clips: AAC, about 32-48 kbps"),
    "instagram": ("Instagram",           "64k",     "Voice messages: AAC, about 64 kbps"),
    "viber":     ("Viber",               "16k",     "Voice messages: Opus, about 16-24 kbps"),
    "wechat":    ("WeChat",              "8k",      "Voice messages: SILK/AMR narrowband"),
    "discord":   ("Discord",             "64k",     "Voice: Opus, about 64 kbps"),
    "snapchat":  ("Snapchat",            "64k",     "Audio: AAC, about 64 kbps"),
    "reddit":    ("Reddit",              "64k",     "Video audio: AAC, 64-128 kbps"),
    "gmail":     ("Gmail / e-mail",      "clean",   "Attachments are usually not re-encoded"),
    "phone":     ("Phone call / voicemail", "8k",   "Narrowband telephony (G.711/AMR)"),
    "other":     ("Other messenger",     "32k",     "Assumed mid-rate voice codec"),
    "unknown":   ("I don't know",        "unknown", "General model, no bitrate assumption"),
}

app = FastAPI(title="CompressedAudioForensics")
STATE = {}


@app.on_event("startup")
def _load():
    from transformers import WavLMModel
    STATE["wavlm"] = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE).eval()
    STATE["art"] = model_store.load()
    print(f"[ready] device={DEVICE}  model={STATE['art']['version']}", flush=True)


def probe_file(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                              "stream=codec_name,bit_rate,sample_rate,channels:format=bit_rate,duration,format_name",
                              "-of", "json", path], capture_output=True, text=True, timeout=30).stdout
        j = json.loads(out or "{}")
    except Exception:
        return {}
    st = (j.get("streams") or [{}])[0]
    fm = j.get("format", {})
    br = st.get("bit_rate") or fm.get("bit_rate")
    return {"codec": st.get("codec_name"), "bitrate_kbps": round(int(br) / 1000) if br and br.isdigit() else None,
            "sample_rate": int(st["sample_rate"]) if st.get("sample_rate") else None,
            "channels": st.get("channels"), "container": fm.get("format_name"),
            "duration": float(fm["duration"]) if fm.get("duration") else None}


def decode(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-t", str(MAX_SECONDS), "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, timeout=180)
    if raw.returncode != 0 or not raw.stdout:
        raise HTTPException(400, "Could not decode this audio file.")
    return np.frombuffer(raw.stdout, dtype=np.float32).copy()


LOSSLESS = {"pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_u8", "flac", "alac", "wavpack"}


def bitrate_class(kbps):
    if kbps is None:
        return None
    return "8k" if kbps <= 12 else "16k" if kbps <= 24 else "32k" if kbps <= 48 else "64k" if kbps <= 128 else "clean"


def choose_calibration(platform, meta):
    name, default, note = PLATFORMS.get(platform, PLATFORMS["unknown"])
    codec = (meta.get("codec") or "").lower()
    if codec and codec not in LOSSLESS and meta.get("bitrate_kbps"):
        c = bitrate_class(meta["bitrate_kbps"])
        why = f"Read from the file itself: {codec.upper()} at {meta['bitrate_kbps']} kbps"
        if default not in ("unknown", c):
            why += f" (overrides the {name} default of {default})"
        return c, why, True
    if default == "unknown":
        return "unknown", "No bitrate information; general model used", False
    return default, f"{name} default: {note}", False


def windows(x):
    if len(x) <= N_SAMPLES:
        n = len(x)
        w = np.pad(x, (0, N_SAMPLES - n), mode="reflect" if n > 1 else "constant") if n < N_SAMPLES else x
        return [(0.0, n / SR, w)]
    starts = list(range(0, len(x) - N_SAMPLES + 1, HOP))
    if starts[-1] + N_SAMPLES < len(x):
        starts.append(len(x) - N_SAMPLES)
    return [(s / SR, (s + N_SAMPLES) / SR, x[s:s + N_SAMPLES]) for s in starts]


@torch.no_grad()
def embed(ws):
    wl, feats = STATE["wavlm"], []
    for w in ws:
        t = torch.from_numpy(peak_normalize(w.astype(np.float32))).unsqueeze(0).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            hs = wl(t, output_hidden_states=True).hidden_states
        h = torch.stack(hs, 1)[0].float()
        feats.append(torch.cat([h.mean(1), h.std(1)], -1).mean(0).cpu().numpy())
    return np.stack(feats)


def density(vals, grid, bw):
    v = np.asarray(vals)[:, None]
    d = np.exp(-0.5 * ((grid[None, :] - v) / bw) ** 2).sum(0)
    return (d / (len(vals) * bw * math.sqrt(2 * math.pi))).tolist()


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), platform: str = Form("unknown")):
    if "art" not in STATE:
        raise HTTPException(503, "Model is still loading, try again in a moment.")
    tmpdir = tempfile.mkdtemp(prefix="caf_")
    try:
        path = os.path.join(tmpdir, "upload" + Path(file.filename or "a").suffix[:8])
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        meta = probe_file(path)
        x = decode(path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)  # privacy: never keep uploads
    if len(x) < SR * 0.5:
        raise HTTPException(400, "Audio is shorter than 0.5 s.")

    art = STATE["art"]
    calib, why, from_header = choose_calibration(platform, meta)
    probe = art["probe_U"] if calib == "unknown" else art["probe_F"]
    a, b = art["cal"][calib]
    W = windows(x)
    scores = probe.decision_function(embed([w for _, _, w in W]))
    probs = 1 / (1 + np.exp(-(a * scores + b)))

    rms = np.array([float(np.sqrt(np.mean(w ** 2))) for _, _, w in W])
    speech = rms >= 0.1 * rms.max() if rms.max() > 0 else np.ones(len(W), bool)
    use = speech if speech.any() else np.ones(len(W), bool)
    p_med = float(np.median(probs[use]))
    s_med = float(np.median(scores[use]))

    run = best = 0
    run_start = best_span = None
    for i, (p, sp) in enumerate(zip(probs, speech)):
        if sp and p < 0.25:
            run += 1
            run_start = i if run == 1 else run_start
            if run > best:
                best, best_span = run, (W[run_start][0], W[i][1])
        else:
            run = 0
    spread = float(np.std(probs[use])) if use.sum() > 1 else 0.0

    if p_med >= 0.5 and best >= 2:
        verdict, level = "Suspicious segment detected", "warn"
    elif 0.35 < p_med < 0.65:
        verdict, level = "Uncertain", "warn"
    elif p_med >= 0.5:
        verdict, level = "Likely genuine human voice", "real"
    else:
        verdict, level = "Likely AI-generated voice", "fake"

    notes = []
    if calib == "8k":
        notes.append("Very low bitrate (about 8 kbps): measured accuracy about 95%, lower than at higher bitrates.")
    if calib == "unknown":
        notes.append("General model without a bitrate hint: measured accuracy 95-97% down to 16 kbps but about 78% at 8 kbps. "
                     "Choose the source app if you know it.")
    if len(W) > 1 and spread > 0.3:
        notes.append("Confidence varies strongly over time; inspect the timeline in View analytics.")

    ref = art["ref"][calib]
    allv = np.concatenate([ref["real"], ref["fake"], [s_med]])
    lo, hi = float(np.percentile(allv, 1)) - 1, float(np.percentile(allv, 99)) + 1
    grid = np.linspace(lo, hi, 160)
    bw = max((hi - lo) / 40, 1e-3)
    advice = ["Automated detection is probabilistic. Treat this as an advisory, not proof."]
    if level != "real":
        advice = ["Do not send money, codes or personal details based on this audio.",
                  "Call the person back on a number you already know.",
                  "Ask something only the real person would know."] + advice

    return JSONResponse({
        "verdict": verdict, "level": level,
        "p_real": round(p_med, 4), "p_fake": round(1 - p_med, 4),
        "margin": round(abs(p_med - 0.5) * 2, 3),
        "calibration": calib, "calibration_reason": why, "from_header": from_header,
        "platform": PLATFORMS.get(platform, PLATFORMS["unknown"])[0],
        "file": {**meta, "decoded_seconds": round(len(x) / SR, 2)},
        "windows": [{"t0": round(t0, 2), "t1": round(t1, 2), "p_real": round(float(p), 4), "speech": bool(sp)}
                    for (t0, t1, _), p, sp in zip(W, probs, speech)],
        "suspicious_span": [round(best_span[0], 2), round(best_span[1], 2)] if best >= 2 else None,
        "reference": {"grid": grid.round(3).tolist(), "real": density(ref["real"], grid, bw),
                      "fake": density(ref["fake"], grid, bw), "file_score": round(s_med, 3),
                      "threshold": round(-b / a, 3) if a else 0.0},
        "notes": notes, "advice": advice, "model": art["version"],
    })


@app.get("/api/platforms")
def platforms():
    return [{"id": k, "name": v[0], "assumed": v[1], "note": v[2]} for k, v in PLATFORMS.items()]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
