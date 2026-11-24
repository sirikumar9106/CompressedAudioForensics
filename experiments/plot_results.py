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


def stability(summ, rows, out):
    import csv as _csv
    dets = list(dict.fromkeys(r["detector"] for r in summ))
    conds = list(dict.fromkeys(r["condition"] for r in summ))
    S = {(r["detector"], r["condition"]): r for r in summ}
    fig, axs = plt.subplots(1, len(dets), figsize=(13, 3.8), sharey=True)
    for ax, d in zip(axs, dets):
        ax.plot(conds, [S[d, c]["mean_p_bonafide_on_bonafide"] for c in conds], color=C[2], lw=2, marker="o", ms=5, label="Real voices")
        ax.plot(conds, [S[d, c]["mean_p_bonafide_on_spoof"] for c in conds], color=C[1], lw=2, marker="o", ms=5, label="AI voices")
        ax.axhline(0.5, color=INK2, ls="--", lw=1); ax.set_title(d, fontsize=9); ax.set_xlabel("MP3 bitrate")
    axs[0].set_ylabel("Mean P(real)"); axs[0].legend(fontsize=8)
    fig.suptitle("EXP-004b: average confidence per class as compression increases (same 200 files)", fontsize=11, y=1.02)
    save(fig, out, "1_mean_confidence_per_class.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, d in enumerate(dets):
        ax.plot(conds, [S[d, c]["decision_flip_rate_vs_clean"] * 100 for c in conds], color=C[j], lw=2, marker="o", ms=5, label=d)
    ax.set_ylabel("Files whose verdict flipped vs clean (%)"); ax.set_xlabel("MP3 bitrate"); ax.legend(fontsize=8)
    ax.set_title("EXP-004b: decision flips caused by compression alone")
    save(fig, out, "2_decision_flip_rate.png")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for j, d in enumerate(dets):
        a1.plot(conds, [S[d, c]["auc"] for c in conds], color=C[j], lw=2, marker="o", ms=5, label=d)
        a2.plot(conds, [S[d, c]["acc@0.5"] * 100 for c in conds], color=C[j], lw=2, marker="o", ms=5, label=d)
    a1.set_title("Ranking (AUC): stays high"); a1.set_ylabel("ROC-AUC"); a2.set_title("Accuracy at 0.5: collapses for WavLM at 8k")
    a2.set_ylabel("Accuracy (%)"); a2.axhline(50, color=INK2, ls="--", lw=1)
    for a in (a1, a2): a.set_xlabel("MP3 bitrate")
    a2.legend(fontsize=7, loc="lower left")
    fig.suptitle("EXP-004b: separability survives, calibrated confidence does not", fontsize=11, y=1.02)
    save(fig, out, "3_ranking_vs_accuracy.png")
    data = {}
    for r in rows:
        if r["label"] == "1":
            data.setdefault((r["detector"], r["condition"]), []).append(float(r["p_bonafide"]))
    fig, axs = plt.subplots(1, len(dets), figsize=(13, 3.8), sharey=True)
    for ax, d in zip(axs, dets):
        bp = ax.boxplot([data[d, c] for c in conds], tick_labels=conds, widths=0.5, patch_artist=True,
                        medianprops={"color": INK}, flierprops={"markersize": 3})
        for b in bp["boxes"]:
            b.set_facecolor(C[2]); b.set_alpha(0.6)
        ax.axhline(0.5, color=INK2, ls="--", lw=1); ax.set_title(d, fontsize=9); ax.set_xlabel("MP3 bitrate")
    axs[0].set_ylabel("P(real) for genuine human voices")
    fig.suptitle("EXP-004b: spread of confidence on REAL voices (false-alarm risk below 0.5)", fontsize=11, y=1.02)
    save(fig, out, "4_real_voice_confidence_spread.png")


def double_comp(rows, out):
    dets = list(dict.fromkeys(r["detector"] for r in rows))
    brs = list(dict.fromkeys(r["bitrate"] for r in rows))
    S = {(r["detector"], r["mode"], r["bitrate"]): r for r in rows}
    def single_vs_double(key, ylab, title, name, scale=1, ref=None):
        fig, axs = plt.subplots(1, len(dets), figsize=(13, 3.8), sharey=True)
        for ax, d in zip(axs, dets):
            for j, (m, lab) in enumerate([("single", "Single compression (X)"), ("double", "Re-compressed (64k → X)")]):
                ax.plot(brs, [S[d, m, b][key] * scale for b in brs], color=C[j], lw=2, marker="o", ms=5, label=lab,
                        ls="-" if m == "single" else "--")
            if ref is not None:
                ax.axhline(ref, color=INK2, ls=":", lw=1)
            ax.set_title(d, fontsize=9); ax.set_xlabel("Final MP3 bitrate X")
        axs[0].set_ylabel(ylab); axs[0].legend(fontsize=8)
        fig.suptitle(title, fontsize=11, y=1.02)
        save(fig, out, name)
    single_vs_double("confidence_gap", "Mean P(real) on real − on fake",
                     "EXP-004c: confidence gap between real and AI voices, single vs re-compressed", "1_confidence_gap.png")
    single_vs_double("mean_p_real_on_real", "Mean P(real) on REAL voices",
                     "EXP-004c: confidence on genuine voices (below 0.5 = false alarm)", "2_real_voice_confidence.png", ref=0.5)
    single_vs_double("acc@0.5", "Accuracy at 0.5 (%)", "EXP-004c: accuracy, single vs re-compressed",
                     "3_accuracy.png", scale=100, ref=50)
    single_vs_double("auc", "ROC-AUC", "EXP-004c: ranking ability, single vs re-compressed", "4_auc.png")


def exp004d(rows, out):
    V = ["BASE", "A", "B", "C"]
    VN = {"BASE": "BASE (EXP-004c)", "A": "A: per-bitrate calibration", "B": "B: mixed-bitrate probe",
          "C": "C: + 4 kHz low-pass branch"}
    S = {(r["variant"], r["condition"]): r for r in rows}
    brs = ["64k", "32k", "16k", "8k"]
    for k, (mode, ttl) in enumerate([("single", "single compression"), ("double", "re-compressed (64k → X)")], 1):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for j, v in enumerate(V):
            ys = [S[v, f"{mode}_{b}"]["acc"] * 100 for b in brs]
            lo = [y - S[v, f"{mode}_{b}"]["acc_lo"] * 100 for y, b in zip(ys, brs)]
            hi = [S[v, f"{mode}_{b}"]["acc_hi"] * 100 - y for y, b in zip(ys, brs)]
            ax.errorbar(brs, ys, yerr=[lo, hi], color=C[j], lw=2, marker="o", ms=5, capsize=3, label=VN[v])
        ax.axhline(95, color=INK2, ls="--", lw=1); ax.text(3, 95.6, "target 95%", ha="right", fontsize=8, color=INK2)
        ax.set_ylim(45, 101); ax.set_ylabel("Accuracy (%) with 95% CI"); ax.set_xlabel("Final MP3 bitrate")
        ax.set_title(f"EXP-004d: accuracy vs bitrate, {ttl} (n=1,000)"); ax.legend(fontsize=8, loc="lower left")
        save(fig, out, f"{k}_accuracy_{mode}.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    cs = ["single_16k", "double_16k"]
    n = len(V); w = 0.8 / n
    for j, v in enumerate(V):
        ys = [S[v, c]["acc"] * 100 for c in cs]
        err = [[y - S[v, c]["acc_lo"] * 100 for y, c in zip(ys, cs)], [S[v, c]["acc_hi"] * 100 - y for y, c in zip(ys, cs)]]
        xs = [x + (j - (n - 1) / 2) * w for x in range(len(cs))]
        b = ax.bar(xs, ys, w * 0.92, color=C[j], label=VN[v], yerr=err, capsize=3, ecolor=INK2)
        ax.bar_label(b, [f"{y:.1f}" for y in ys], fontsize=8, color=INK2, padding=8)
    ax.axhline(95, color=INK2, ls="--", lw=1)
    ax.set_xticks(range(len(cs)), ["16 kbps (single)", "16 kbps (re-compressed)"]); ax.set_ylim(85, 100)
    ax.set_ylabel("Accuracy (%) with 95% CI"); ax.set_title("EXP-004d: the 16 kbps target (≥ 95%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    save(fig, out, "3_target_16k.png")
    conds = ["clean"] + [f"single_{b}" for b in brs] + [f"double_{b}" for b in brs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
    for j, v in enumerate(V):
        a1.plot(range(len(conds)), [S[v, c]["false_alarm_rate"] * 100 for c in conds], color=C[j], lw=2, marker="o", ms=5, label=VN[v])
        a2.plot(range(len(conds)), [S[v, c]["miss_rate"] * 100 for c in conds], color=C[j], lw=2, marker="o", ms=5, label=VN[v])
    lbl = [c.replace("single_", "").replace("double_", "2×") for c in conds]
    for a, t in [(a1, "False alarms: real voices called AI (%)"), (a2, "Misses: AI voices called real (%)")]:
        a.set_xticks(range(len(conds)), lbl); a.set_title(t, fontsize=10); a.set_xlabel("Condition (2× = re-compressed)")
    a1.set_yscale("symlog", linthresh=5); a2.legend(fontsize=8)
    fig.suptitle("EXP-004d: where the errors come from", fontsize=11, y=1.02)
    save(fig, out, "4_error_breakdown.png")


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
    if (R / "confidence_stability.json").exists():
        import csv as _csv
        stability(json.load(open(R / "confidence_stability.json")),
                  list(_csv.DictReader(open(R / "confidence_stability.csv"))), O / "EXP-004b_confidence_stability")
    if (R / "double_compression.json").exists():
        double_comp(json.load(open(R / "double_compression.json")), O / "EXP-004c_recompression")
    if (R / "exp004d.json").exists():
        exp004d(json.load(open(R / "exp004d.json")), O / "EXP-004d_tweaks")
    print("plots written to", O)


if __name__ == "__main__":
    main()
