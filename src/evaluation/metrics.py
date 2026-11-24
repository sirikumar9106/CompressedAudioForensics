"""Pure-NumPy metrics. Convention: score = higher means MORE bonafide; label 1 = bonafide, 0 = spoof."""
import numpy as np


def roc_auc(labels, scores) -> float:
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def eer(labels, scores) -> tuple[float, float]:
    """Returns (EER, threshold). FAR = spoof accepted as bonafide; FRR = bonafide rejected."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    thr = np.unique(s)
    thr = np.concatenate([thr, [thr[-1] + 1e-6]])
    pos, neg = s[y == 1], s[y == 0]
    far = np.array([(neg >= t).mean() for t in thr])
    frr = np.array([(pos < t).mean() for t in thr])
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(thr[k])


def threshold_rates(labels, scores, threshold: float) -> dict:
    """Confusion-matrix rates at a fixed threshold (predict bonafide if score >= threshold).
    Positive class here = SPOOF (detection task): FPR = bonafide flagged as spoof, FNR = spoof missed."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    pred_bona = s >= threshold
    tp = int(((y == 0) & ~pred_bona).sum()); fn = int(((y == 0) & pred_bona).sum())
    tn = int(((y == 1) & pred_bona).sum()); fp = int(((y == 1) & ~pred_bona).sum())
    return {"TP_spoof": tp, "FN_spoof": fn, "TN_bona": tn, "FP_bona": fp,
            "FPR": fp / max(fp + tn, 1), "FNR": fn / max(fn + tp, 1), "accuracy": (tp + tn) / max(len(y), 1)}


def brier(labels, probs_bonafide) -> float:
    return float(np.mean((np.asarray(probs_bonafide, float) - np.asarray(labels, float)) ** 2))


def bootstrap_ci(labels, scores, metric, n_iter: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Returns (point, lo, hi). Stratified resampling keeps both classes present."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_iter):
        idx = np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))])
        v = metric(y[idx], s[idx])
        vals.append(v[0] if isinstance(v, tuple) else v)
    point = metric(y, s)
    point = point[0] if isinstance(point, tuple) else point
    return float(point), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
