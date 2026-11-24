"""Audio standardization: 16 kHz mono, peak-normalize to 0.95, pad/crop to 64,600 samples."""
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

SR = 16000
N_SAMPLES = 64600


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)  # mono downmix
    if fs != sr:
        g = gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g).astype(np.float32)
    return x


def peak_normalize(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x if m < 1e-8 else (x / m * peak).astype(np.float32)


def fix_length(x: np.ndarray, n: int = N_SAMPLES, mode: str = "center", pad: str = "reflect",
               rng: np.random.Generator | None = None):
    """Returns (window[n], valid_mask[n]). mode: 'center' (inference) | 'random' (training).

    No np.tile repeat-padding (introduces artificial periodicity).
    """
    L = len(x)
    if L >= n:
        start = (L - n) // 2 if mode == "center" else int((rng or np.random.default_rng()).integers(0, L - n + 1))
        return x[start:start + n].astype(np.float32), np.ones(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    mask[:L] = True
    if pad == "reflect" and L > 1:
        out = x
        while len(out) < n:  # reflect repeatedly for very short clips
            out = np.pad(out, (0, min(n - len(out), len(out) - 1)), mode="reflect")
        return out[:n].astype(np.float32), mask
    return np.pad(x, (0, n - L)).astype(np.float32), mask


def preprocess(x: np.ndarray, mode: str = "center", rng=None) -> np.ndarray:
    w, _ = fix_length(x, mode=mode, rng=rng)
    return peak_normalize(w)


def sliding_windows(x: np.ndarray, n: int = N_SAMPLES, hop: int = N_SAMPLES // 2) -> list[np.ndarray]:
    """Extension stage: 50%-overlap windows."""
    if len(x) <= n:
        return [preprocess(x)]
    return [peak_normalize(x[s:s + n]) for s in range(0, len(x) - n + 1, hop)]
