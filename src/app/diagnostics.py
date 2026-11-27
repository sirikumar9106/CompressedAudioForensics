"""Per-file diagnostics for the fine-tuned WavLM detector (EXP-008b onward).

Six interactive charts (data only), in the model's own score units (log-odds of "real"; +ve = sounds human):
  1 spectrogram + per-window score        2 time x frequency-band occlusion map
  3 frequency-band importance (mean)      4 WavLM layer importance (+ learned layer weights)
  5 where the model listens (attention)   6 long-term spectrum + simple acoustic measurements
Occlusion = remove one 1 kHz band (or one layer) and measure how much the score drops.
A positive bar means "this part makes the file look HUMAN to the model".
The arrays go to the web page as JSON for drawing; nothing is written to disk.
"""
import numpy as np
import torch

SR = 16000
BANDS = [(i * 1000, (i + 1) * 1000) for i in range(8)]
MAX_WIN = 30                       # diagnose at most 30 windows (~60 s)


def _norm(w):
    m = np.abs(w).max()
    return (w / m).astype(np.float32) if m > 0 else w.astype(np.float32)


@torch.no_grad()
def _states(m, xs, dev):
    t = torch.from_numpy(np.stack([_norm(x) for x in xs])).to(dev)
    with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
        hs = m.wavlm(t, output_hidden_states=True).hidden_states
    return torch.stack(hs, 0).float()                     # [L, B, T, D]


@torch.no_grad()
def _head(m, hs, lw, low=None):
    h = (torch.softmax(lw, 0)[:, None, None, None] * hs).sum(0)
    att = torch.softmax(m.pool.att(h), 1)[..., 0]           # [B, T]  (main branch attention)
    lg = m.from_states(hs, layer_w=lw, low_w=low) if hasattr(m, "from_states") else m.head(m.pool(h))
    return (lg[:, 1] - lg[:, 0]).cpu().numpy(), att.cpu().numpy()


def _band_stop(x, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f >= lo) & (f < hi)] = 0
    return np.fft.irfft(X, len(x))


def _quiet(att, env):
    n = min(len(att), len(env)); a = np.nan_to_num(att[:n]); e = env[:n]
    return float(100 * (a * (e < .1 * e.max())).sum() / max(a.sum(), 1e-9))


def acoustic(x):
    fr = x[: len(x) // 320 * 320].reshape(-1, 320)                     # 20 ms frames
    rms = np.sqrt((fr ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)
    voiced = db > db.max() - 30
    S = np.abs(np.fft.rfft(fr * np.hanning(320), axis=1)) ** 2 + 1e-12
    f = np.fft.rfftfreq(320, 1 / SR)
    Sv = S[voiced] if voiced.any() else S
    flat = np.exp(np.log(Sv).mean(1)) / Sv.mean(1)
    cen = (Sv * f).sum(1) / Sv.sum(1)
    return {"pause_ratio": float(1 - voiced.mean()), "loudness_range_db": float(np.percentile(db[voiced], 95) - np.percentile(db[voiced], 5)) if voiced.sum() > 2 else 0.0,
            "energy_above_4k_pct": float(100 * Sv[:, f >= 4000].sum() / Sv.sum()),
            "energy_below_300hz_pct": float(100 * Sv[:, f < 300].sum() / Sv.sum()),
            "spectral_flatness": float(flat.mean()), "spectral_centroid_hz": float(cen.mean()),
            "centroid_variability_hz": float(cen.std())}


def _thin(a, n):
    """Keep at most n evenly spaced points."""
    a = np.asarray(a)
    return a if len(a) <= n else a[np.linspace(0, len(a) - 1, n).astype(int)]


def _r(a, d=3):
    return np.round(np.nan_to_num(np.asarray(a, float)), d).tolist()


def run(model, x, W, speech, scores, dev):
    W, speech, scores = W[:MAX_WIN], np.asarray(speech[:MAX_WIN]), np.asarray(scores[:MAX_WIN])
    wavs = [w for _, _, w in W]
    lw = model.layer_w.detach()
    n_layers = lw.numel()
    band_d = np.zeros((len(W), len(BANDS))); layer_d = np.zeros((len(W), n_layers))
    T_att = np.zeros(int(W[-1][1] * 50) + 2); C_att = np.zeros_like(T_att)
    for i in range(0, len(W), 4):
        chunk = wavs[i:i + 4]
        hs = _states(model, chunk, dev)
        base, att = _head(model, hs, lw)
        for k in range(n_layers):                                   # layer occlusion (cheap: backbone run once)
            l2 = lw.clone(); l2[k] = -1e4
            lo = None
            if getattr(model, "n_low", 0) and k < model.n_low:      # remove the layer from the texture branch too
                lo = model.low_w.detach().clone(); lo[k] = -1e4
            layer_d[i:i + len(chunk), k] = base - _head(model, hs, l2, lo)[0]
        for j, a in enumerate(att):                                 # attention -> 20 ms timeline
            s0 = int(W[i + j][0] * 50)
            n = min(len(a), len(T_att) - s0)
            T_att[s0:s0 + n] += a[:n] * len(a); C_att[s0:s0 + n] += 1
        for bi, (lo, hi) in enumerate(BANDS):                       # frequency-band occlusion
            sc, _ = _head(model, _states(model, [_band_stop(w, lo, hi) for w in chunk], dev), lw)
            band_d[i:i + len(chunk), bi] = base - sc
    att_t = np.where(C_att > 0, T_att / np.maximum(C_att, 1), np.nan)
    wts = torch.softmax(lw, 0).cpu().numpy()
    return pack(x, W, speech, scores, band_d, layer_d, wts, att_t)


def pack(x, W, speech, scores, band_d, layer_d, wts, att_t):
    """Turn the measurements into compact JSON for the charts."""
    dur = W[-1][1]; xs = x[: int(dur * SR)]
    centres = [(a + b) / 2 for a, b, _ in W]
    use = speech if speech.any() else np.ones(len(W), bool)

    fr = xs[: len(xs) // 256 * 256].reshape(-1, 256)              # spectrogram: 16 ms frames, 64 bands to 8 kHz
    S = 10 * np.log10(np.abs(np.fft.rfft(fr * np.hanning(256), axis=1))[:, :128] ** 2 + 1e-10)
    S = S.reshape(len(S), 64, 2).mean(2)
    cols = min(len(S), 180)
    S = np.array([c.mean(0) for c in np.array_split(S, cols)])
    hi = np.percentile(S, 99); S = np.clip((S - (hi - 70)) / 70, 0, 1)

    env = np.sqrt(np.convolve(xs ** 2, np.ones(320) / 320, "same"))[::320]
    env = env / (env.max() + 1e-9); att = att_t / (np.nanmax(att_t) or 1)
    n = min(len(env), len(att)); idx = np.linspace(0, n - 1, min(n, 300)).astype(int)

    blk = xs[: len(xs) // 512 * 512].reshape(-1, 512)
    lt = 10 * np.log10((np.abs(np.fft.rfft(blk * np.hanning(512), axis=1)) ** 2).mean(0) + 1e-12)
    return {
        "centres": _r(centres, 2), "scores": _r(scores), "speech": [bool(s) for s in speech],
        "spec": {"dur": round(float(dur), 2), "rows": 64, "cols": cols, "v": (S.T * 99).round().astype(int).ravel().tolist()},
        "bands": [f"{lo // 1000}–{hi // 1000} kHz" for lo, hi in BANDS],
        "band_map": _r(band_d), "band_mean": _r(band_d[use].mean(0)),
        "layer_drop": _r(layer_d[use].mean(0)), "layer_weight": _r(wts, 4),
        "attention": {"t": _r(idx / 50, 2), "att": _r(att[idx]), "env": _r(env[idx])},
        "spectrum": {"f": _r(_thin(np.fft.rfftfreq(512, 1 / SR), 129), 0), "db": _r(_thin(lt, 129), 1)},
        "acoustic": acoustic(xs),
    }
