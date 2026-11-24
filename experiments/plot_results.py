"""Render result graphs for each experiment into mapped_results/EXP-xxx/.

Usage: python experiments/plot_results.py [--results results] [--out mapped_results]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    plt.close(fig)


def bars(ax, labels, series, names, ylim=None, fmt="{:.3f}"):
    n = len(series)
    w = 0.8 / n
    for i, (vals, nm) in enumerate(zip(series, names)):
        xs = [x + (i - (n - 1) / 2) * w for x in range(len(labels))]
        b = ax.bar(xs, vals, w * 0.92, color=C[i], label=nm)
        ax.bar_label(b, [fmt.format(v) for v in vals], fontsize=7, color=INK2, padding=2)
    ax.set_xticks(range(len(labels)), labels)
    if ylim:
        ax.set_ylim(*ylim)
    if n > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(n, 4), fontsize=8)


def pilot(res, out, title):
    fs = ["delta", "base", "pert", "full"]
    lab = ["Δz (diff)", "Base emb.", "Perturbed emb.", "All combined"]
    p = res["probes"]
    # 1 feature-set comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    bars(ax, lab, [[p["mulaw8"][f]["test_auc"] for f in fs], [p["gaussian"][f]["test_auc"] for f in fs]],
         ["μ-law perturbation", "Gaussian-noise control"], ylim=(0.5, 1.0))
    ax.set_ylabel("Held-out ROC-AUC"); ax.set_title(f"{title}: AUC by feature set")
    save(fig, out, "1_auc_by_feature_set.png")
    # 2 gate check
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [p["mulaw8"]["delta"]["test_auc"], p["mulaw8"]["base"]["test_auc"], p["gaussian"]["delta"]["test_auc"]]
    b = ax.bar(["Δz (μ-law)", "Base embedding", "Δz (noise)"], vals, 0.55, color=[C[0], C[2], C[1]])
    ax.bar_label(b, [f"{v:.3f}" for v in vals], fontsize=8, color=INK2, padding=2)
    for y, t in [(0.70, "GO threshold 0.70"), (0.60, "NO-GO below 0.60")]:
        ax.axhline(y, color=INK2, lw=1, ls="--"); ax.text(2.45, y + 0.006, t, ha="right", fontsize=8, color=INK2)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Held-out ROC-AUC")
    ax.set_title(f"{title}: gate check ({res['verdict']}) — Δz must beat base embedding")
    save(fig, out, "2_gate_check.png")
    # 3 confounds
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    cv = [res["controls"]["rms_auc"], res["controls"]["duration_auc"]]
    b = a1.bar(["Loudness (RMS)", "Duration"], cv, 0.5, color=C[3]); a1.bar_label(b, [f"{v:.2f}" for v in cv], fontsize=8)
    a1.axhline(0.5, color=INK2, ls="--", lw=1); a1.set_ylim(0, 1); a1.set_ylabel("AUC alone (0.5 = chance)")
    a1.set_title("Nuisance-only classifiers")
    d = p["mulaw8"]["delta"]; rv = [d["corr_rms"], d["corr_duration"]]
    b = a2.bar(["|r| Δz score vs RMS", "|r| Δz score vs duration"], rv, 0.5, color=C[4])
    a2.bar_label(b, [f"{v:.2f}" for v in rv], fontsize=8)
    a2.axhline(0.20, color=INK2, ls="--", lw=1); a2.text(1.4, 0.21, "limit 0.20", ha="right", fontsize=8, color=INK2)
    a2.set_ylim(0, 0.7); a2.set_title("Confound correlation")
    fig.suptitle(f"{title}: shortcut / confound checks", fontsize=11, y=1.02)
    save(fig, out, "3_confound_checks.png")


def sweep(rows, out):
    x = [r["condition"].replace("mp3_", "") for r in rows]
    for i, (key, ylab, name, fn) in enumerate([
        ("auc", "Test ROC-AUC", "1_auc_vs_bitrate.png", None),
        ("eer", "Test EER (%)", "2_eer_vs_bitrate.png", 100)]):
        fig, ax = plt.subplots(figsize=(7, 4))
        for j, (k, lab) in enumerate([("modelA", "Model A (official RawNet2)"), ("base_probe", "Base-embedding probe"),
                                      ("delta_probe", "Δz probe")]):
            col = f"{k}_{key}"
            ys = [r[col] * (fn or 1) for r in rows]
            ax.plot(x, ys, color=C[j], lw=2, marker="o", ms=5, label=lab)
        ax.set_xlabel("Condition (MP3 bitrate)"); ax.set_ylabel(ylab); ax.legend()
        ax.set_title(f"EXP-003: {ylab.split(' (')[0]} vs compression (n=200, unseen attacks)")
        save(fig, out, name)
    fig, ax = plt.subplots(figsize=(7, 4))
    ys = [r["modelA_acc@0"] * 100 for r in rows]
    b = ax.bar(x, ys, 0.6, color=C[0]); ax.bar_label(b, [f"{v:.1f}%" for v in ys], fontsize=8, color=INK2)
    ax.set_ylim(50, 100); ax.set_ylabel("Accuracy at default threshold (%)"); ax.set_xlabel("Condition (MP3 bitrate)")
    ax.set_title("EXP-003: Model A fixed-threshold accuracy drifts under heavy compression")
    save(fig, out, "3_modelA_threshold_accuracy.png")


def ssl(rows, out):
    conds = [r["condition"].replace("mp3_", "MP3 ") for r in rows]
    def get(r, prefix):
        k = next(k for k in r if k.startswith(prefix)); return r[k]
    main = [("modelA", "Model A (official)"), ("rawnet2_probe", "RawNet2 probe"),
            ("wavlm-base-plus_layeravg", "WavLM-Base+ layer-avg"), ("wavlm-large_layeravg", "WavLM-Large layer-avg")]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars(ax, conds, [[get(r, k)[0] for r in rows] for k, _ in main], [n for _, n in main], ylim=(0.9, 1.005))
    ax.set_ylabel("Test ROC-AUC (unseen attacks)"); ax.set_title("EXP-004: front-end comparison under compression")
    save(fig, out, "1_auc_by_frontend.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, (k, n) in enumerate(main):
        ax.plot(conds, [get(r, k)[1] * 100 for r in rows], color=C[j], lw=2, marker="o", ms=5, label=n)
    ax.set_ylabel("Test EER (%)"); ax.set_xlabel("Condition"); ax.legend(fontsize=8)
    ax.set_title("EXP-004: EER vs compression (lower is better)")
    save(fig, out, "2_eer_vs_compression.png")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, m, nm in zip(axs, ["wavlm-base-plus", "wavlm-large"], ["WavLM-Base+", "WavLM-Large"]):
        bars(ax, conds, [[get(r, f"{m}_{v}")[0] for r in rows] for v in ["last", "layeravg", "bestlayer"]],
             ["Last layer", "All-layer average", "Best layer (val)"], ylim=(0.97, 1.002), fmt="{:.4f}")
        ax.set_title(nm)
    axs[0].set_ylabel("Test ROC-AUC")
    fig.suptitle("EXP-004: which WavLM layers to use", fontsize=11, y=1.02)
    save(fig, out, "3_layer_aggregation.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="mapped_results")
    a = ap.parse_args()
    R, O = Path(a.results), Path(a.out)
    if (R / "phase2_pilot.json").exists():
        pilot(json.load(open(R / "phase2_pilot.json")), O / "EXP-001_pilot_seen_attacks", "EXP-001")
    if (R / "phase2_pilot_unseen.json").exists():
        pilot(json.load(open(R / "phase2_pilot_unseen.json")), O / "EXP-002_pilot_unseen_attacks", "EXP-002")
    if (R / "bitrate_sweep.json").exists():
        sweep(json.load(open(R / "bitrate_sweep.json")), O / "EXP-003_bitrate_sweep")
    if (R / "ssl_probe.json").exists():
        ssl(json.load(open(R / "ssl_probe.json")), O / "EXP-004_wavlm_probe")
    print("plots written to", O)


if __name__ == "__main__":
    main()
