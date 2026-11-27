"""Per-file diagnostics for the fine-tuned WavLM detector (EXP-008b onward).

Six pictures, all in the model's own score units (log-odds of "real"; +ve = sounds human):
  1 spectrogram + per-window score        2 time x frequency-band occlusion map
  3 frequency-band importance (mean)      4 WavLM layer importance (+ learned layer weights)
  5 where the model listens (attention)   6 long-term spectrum + simple acoustic measurements
Occlusion = remove one 1 kHz band (or one layer) and measure how much the score drops.
A positive bar means "this part makes the file look HUMAN to the model".
Images are returned to the web page only (base64); nothing is written to disk.
"""
import base64, io

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000
BANDS = [(i * 1000, (i + 1) * 1000) for i in range(8)]
MAX_WIN = 30                       # diagnose at most 30 windows (~60 s)
BG, FG, MUT, GRID = "#0e131c", "#dbe3ef", "#8793a6", "#232b38"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": GRID,
                     "axes.grid": True, "grid.color": GRID, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


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


def _png(fig):
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


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

    dur = W[-1][1]; xs = x[: int(dur * SR)]
    centres = np.array([(a + b) / 2 for a, b, _ in W])
    imgs = []
    use = speech if speech.any() else np.ones(len(W), bool)

    # 1 spectrogram + score
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.specgram(xs + 1e-7, NFFT=512, Fs=SR, noverlap=352, cmap="magma", vmin=-130)
    ax.set_ylim(0, 8000); ax.set_ylabel("frequency (Hz)"); ax.set_xlabel("time (s)"); ax.grid(False)
    a2 = ax.twinx(); a2.plot(centres, scores, color="#7fd1ff", lw=2, marker="o", ms=4)
    a2.axhline(0, color="#7fd1ff", ls="--", lw=1, alpha=.6); a2.set_ylabel("score (+ human / − AI)", color="#7fd1ff"); a2.grid(False)
    ax.set_title("1 · Spectrogram with the model's score per 4 s window")
    imgs.append(("Spectrogram + score", "What the audio contains over time, and how human each window sounded to the model (above the dashed line = human).", _png(fig)))

    # 2 time x band occlusion
    lim = max(np.abs(band_d).max(), 1e-3)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(band_d.T, aspect="auto", origin="lower", cmap="RdBu", vmin=-lim, vmax=lim,
                   extent=[0, len(W), 0, 8])
    ax.set_yticks(np.arange(8) + .5, [f"{lo//1000}–{hi//1000} kHz" for lo, hi in BANDS], fontsize=8)
    ax.set_xticks(np.arange(len(W)) + .5, [f"{c:.0f}" for c in centres], fontsize=7); ax.set_xlabel("window centre (s)"); ax.grid(False)
    fig.colorbar(im, ax=ax, label="score drop when band removed")
    ax.set_title("2 · Which frequency band, at which moment, drives the verdict (blue = looks human)")
    imgs.append(("Time × frequency map", "Blue cells: removing that 1 kHz band at that moment makes the file look less human, so the model relies on it as evidence of a real voice. Red cells: that band pushes the verdict towards AI.", _png(fig)))

    # 3 band importance
    bm = band_d[use].mean(0)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(range(8), bm, color=["#4dabf7" if v >= 0 else "#ff6b6b" for v in bm], width=.7)
    ax.set_xticks(range(8), [f"{lo//1000}–{hi//1000} kHz" for lo, hi in BANDS]); ax.axhline(0, color=MUT, lw=1)
    ax.set_ylabel("mean score drop"); ax.set_title("3 · Frequency-band importance (blue = evidence for human, red = evidence for AI)")
    imgs.append(("Frequency bands", "Average over speech windows of picture 2. The tallest bars are the parts of the spectrum the verdict depends on most.", _png(fig)))

    # 4 layer importance
    lm = layer_d[use].mean(0); wts = torch.softmax(lw, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(range(n_layers), lm, color=["#4dabf7" if v >= 0 else "#ff6b6b" for v in lm], width=.7, label="score drop when layer removed")
    a2 = ax.twinx(); a2.plot(range(n_layers), wts, color="#ffd43b", marker="o", lw=1.5, label="learned layer weight"); a2.grid(False)
    a2.set_ylabel("learned weight", color="#ffd43b"); ax.axhline(0, color=MUT, lw=1)
    ax.set_xticks(range(n_layers), ["CNN"] + [str(i) for i in range(1, n_layers)]); ax.set_xlabel("WavLM layer (low = acoustic detail, high = speech content)")
    ax.set_ylabel("mean score drop"); ax.set_title("4 · Which WavLM layers carry the verdict")
    imgs.append(("WavLM layers", "Bars: how much the verdict depends on each layer of WavLM for this file. Low layers describe raw sound texture, high layers describe phonetic content. Yellow: the fixed weight the model learned for each layer.", _png(fig)))

    # 5 attention
    tt = np.arange(len(att_t)) / 50
    fig, ax = plt.subplots(figsize=(9, 3.2))
    env = np.sqrt(np.convolve(xs ** 2, np.ones(320) / 320, "same"))[::320]
    ax.fill_between(np.arange(len(env)) / 50, 0, env / (env.max() + 1e-9), color="#3b4658", label="loudness")
    ax.plot(tt, att_t / np.nanmax(att_t), color="#ffd43b", lw=1.4, label="model attention")
    ax.set_xlim(0, dur); ax.set_xlabel("time (s)"); ax.set_yticks([]); ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_title("5 · Where the model listens (attention over 20 ms frames)")
    imgs.append(("Attention", "Yellow peaks are the moments the model weighted most when forming its decision (grey = loudness). Peaks on silence or breaths mean the model is using non-speech cues.", _png(fig)))

    # 6 spectrum + acoustics
    ac = acoustic(xs)
    fr = xs[: len(xs) // 512 * 512].reshape(-1, 512)
    lt = 10 * np.log10((np.abs(np.fft.rfft(fr * np.hanning(512), axis=1)) ** 2).mean(0) + 1e-12)
    fig, (ax, at) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax.plot(np.fft.rfftfreq(512, 1 / SR), lt, color="#4dabf7", lw=1.5); ax.set_xlim(0, 8000)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("dB"); ax.set_title("6 · Long-term spectrum")
    at.axis("off")
    rows = [("Pauses", f"{100*ac['pause_ratio']:.0f}%"), ("Loudness range", f"{ac['loudness_range_db']:.1f} dB"),
            ("Energy > 4 kHz", f"{ac['energy_above_4k_pct']:.2f}%"), ("Energy < 300 Hz", f"{ac['energy_below_300hz_pct']:.1f}%"),
            ("Spectral flatness", f"{ac['spectral_flatness']:.3f}"), ("Brightness (centroid)", f"{ac['spectral_centroid_hz']:.0f} Hz"),
            ("Brightness variation", f"{ac['centroid_variability_hz']:.0f} Hz")]
    for r, (k, v) in enumerate(rows):
        at.text(0, 1 - r * .14, k, color=MUT, fontsize=9, va="top"); at.text(1, 1 - r * .14, v, color=FG, fontsize=9, va="top", ha="right", family="monospace")
    imgs.append(("Acoustic profile", "The overall tone colour of the file and simple measurements. Compare these numbers between a real recording and an AI one of the same speaker.", _png(fig)))

    return {"images": [{"title": t, "caption": c, "png": p} for t, c, p in imgs]}
