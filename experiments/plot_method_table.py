"""Final method-comparison table (one row per METHOD, not per experiment) -> mapped_results/final_method_comparison.png

Usage: python experiments/plot_method_table.py [--results results] [--out mapped_results]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID, BG, HEAD = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb", "#eeede9"
VERDICT = {"Rejected": "#f6d5d0", "Baseline": "#e8e7e3", "Partial": "#fbe9c6", "Adopted": "#cfe9dc", "Final": "#9fd7bb"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="mapped_results")
    a = ap.parse_args()
    R = Path(a.results)
    cs = {(r["detector"], r["condition"]): r for r in json.load(open(R / "confidence_stability.json"))}
    dc = {(r["detector"], r["mode"], r["bitrate"]): r for r in json.load(open(R / "double_compression.json"))}
    bs = {r["condition"]: r for r in json.load(open(R / "bitrate_sweep.json"))}
    d4 = {(r["variant"], r["condition"]): r for r in json.load(open(R / "exp004d.json"))}
    d5 = {(r["variant"], r["condition"]): r for r in json.load(open(R / "exp005.json"))}
    d6 = {(r["codec"], r["bitrate"], r["mode"]): r for r in json.load(open(R / "exp006.json"))["codecs"]}

    pct = lambda v: f"{v * 100:.1f}%"
    auc = lambda v: f"AUC {v:.3f}"
    MA = "Model A (official)"
    rows = [
        ["Official ASVspoof RawNet2\n(Model A)", "Pretrained raw-waveform CNN+GRU\nscore; Platt-calibrated on clean", "003, 004, 004b, 004c",
         pct(cs[MA, "clean"]["acc@0.5"]), pct(cs[MA, "64k"]["acc@0.5"]), pct(cs[MA, "32k"]["acc@0.5"]),
         pct(cs[MA, "16k"]["acc@0.5"]), pct(cs[MA, "8k"]["acc@0.5"]), pct(dc[MA, "double", "16k"]["acc@0.5"]), "—", "Baseline"],
        ["Δ-Net differential probe\n(μ-law Δz)", "Compare file with its own 8-bit μ-law\nre-degraded copy; classify the change Δz", "001, 002, 003",
         auc(bs["clean"]["delta_probe_auc"]), auc(bs["mp3_64k"]["delta_probe_auc"]), auc(bs["mp3_32k"]["delta_probe_auc"]),
         auc(bs["mp3_16k"]["delta_probe_auc"]), auc(bs["mp3_8k"]["delta_probe_auc"]), "—", "—", "Rejected"],
        ["RawNet2 embedding probe", "Linear classifier on pooled RawNet2\nframe features (static, no Δz)", "001, 002, 003, 004",
         auc(bs["clean"]["base_probe_auc"]), auc(bs["mp3_64k"]["base_probe_auc"]), auc(bs["mp3_32k"]["base_probe_auc"]),
         auc(bs["mp3_16k"]["base_probe_auc"]), auc(bs["mp3_8k"]["base_probe_auc"]), "—", "—", "Rejected"],
    ]

    def wl(src, v, label, desc, exps, verdict, opus=None):
        g = lambda c: pct(src[v, c]["acc"])
        return [label, desc, exps, g("clean"), g("single_64k"), g("single_32k"), g("single_16k"), g("single_8k"),
                g("double_16k"), opus or "—", verdict]

    op = lambda mode: f"{pct(d6['opus', '16k', mode]['acc'])} / {pct(d6['opus', '8k', mode]['acc'])}"
    rows += [
        wl(d4, "BASE", "WavLM probe, one calibration", "Frozen WavLM-Base+, all-layer mean+std,\nlinear probe on 32k, one threshold",
           "004, 004b, 004c, 004d", "Partial"),
        wl(d4, "C", "WavLM + 4 kHz low-pass branch", "Adds a parallel band-limited WavLM\npass; features concatenated", "004d", "Rejected"),
        wl(d4, "B", "WavLM, mixed-bitrate training", "Probe trained on clean+64+32+16k;\none calibration", "004d", "Partial"),
        wl(d5, "E", "WavLM + MLP head\n+ per-bitrate calibration", "Non-linear 1-hidden-layer classifier;\ncalibrator picked by bitrate", "005", "Rejected"),
        wl(d5, "U", "\"Not sure\" mode", "Mixed-bitrate probe + one pooled\ncalibrator (bitrate unknown)", "005, 006", "Adopted",
           op("Not sure")),
        wl(d5, "A", "WavLM + per-bitrate calibration", "32k probe; 2-parameter calibrator\nchosen by the user's bitrate dropdown", "004d, 005, 006", "Adopted",
           op("A + matching dropdown")),
        wl(d5, "F", "WavLM + RawBoost augmentation\n+ per-bitrate calibration", "As above, trained also on noise/channel-\naugmented copies (RawBoost-style)", "005, 006",
           "Final", op("F + matching dropdown")),
    ]
    cols = ["Method", "What it does", "Experiments", "Clean", "MP3 64k", "MP3 32k", "MP3 16k", "MP3 8k",
            "Re-compressed\n64k→16k", "Opus 16k / 8k\n(unseen codec)", "Verdict"]

    fig, ax = plt.subplots(figsize=(24, 7.2))
    fig.patch.set_facecolor(BG); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
                 colWidths=[0.135, 0.16, 0.09, 0.065, 0.065, 0.065, 0.065, 0.065, 0.075, 0.09, 0.06])
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 3.1)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor(HEAD); cell.set_text_props(weight="bold", color=INK)
            continue
        txt = cell.get_text().get_text()
        cell.set_facecolor(BG)
        if c in (0, 1):
            cell.set_text_props(ha="left", color=INK if c == 0 else INK2, weight="bold" if c == 0 else "normal")
            cell._loc = "left"; cell.PAD = 0.03
        if c == 10:
            cell.set_facecolor(VERDICT[txt]); cell.set_text_props(weight="bold")
        if 3 <= c <= 9 and txt.endswith("%") and "/" not in txt:
            v = float(txt[:-1])
            cell.set_text_props(color="#007d4f" if v >= 95 else ("#b4321d" if v < 85 else INK), weight="bold" if v >= 95 else "normal")
        if rows[r - 1][10] == "Final":
            cell.set_linewidth(1.6); cell.set_edgecolor("#1baf7a")
    ax.set_title("Method comparison: detecting AI voices in compressed audio (unseen attacks)", fontsize=15, color=INK, pad=14, weight="bold")
    fig.text(0.5, 0.02,
             "Values = accuracy at the deployed threshold on 1,000 unseen-attack files (2019 LA eval, A07–A18), except Model A (200 files) "
             "and the two RawNet2 probe rows, which report ROC-AUC (only ranking was measured).\n"
             "Green bold ≥ 95%   ·   red < 85%   ·   Re-compressed = sent at 64 kbps, forwarded at 16 kbps   ·   "
             "Opus was never used for training or calibration   ·   Full details: EXPERIMENT_LOG.md",
             ha="center", fontsize=9, color=INK2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "final_method_comparison.png", dpi=170, bbox_inches="tight", facecolor=BG)
    print("saved", out / "final_method_comparison.png")


if __name__ == "__main__":
    main()
