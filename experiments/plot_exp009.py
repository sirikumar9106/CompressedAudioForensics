"""Graphs for EXP-009a (image CNN vs all methods) and EXP-009b (feature audit + fusion)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"; M = "mapped_results"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "axes.axisbelow": True})
COL = {"WavLM F (reference)": "#2a6fdb", "WavLM F": "#2a6fdb", "Image CNN (EXP-009a)": "#d9480f",
       "RawNet2 (official)": "#2b8a3e", "Delta-Net (mu-law difference)": "#868e96",
       "Image CNN, selected features": "#e8590c", "Combined: average": "#9c36b5",
       "Combined: learned weights": "#c2255c"}

def label(c):
    return c.replace("single_", "").replace("double_", "2×").replace("clean", "clean")

def table(rows):
    t = {}
    for r in rows: t.setdefault(r["method"], {})[r["condition"]] = r
    return t

a = json.load(open(f"{R}/exp009a.json")); b = json.load(open(f"{R}/exp009b.json"))
ta, tb = table(a["rows"]), table(b["rows"])
ta["WavLM F"] = tb["WavLM F (reference)"]
conds = [c for c in ta["Image CNN (EXP-009a)"] if c in ta["WavLM F"]]
x = np.arange(len(conds))
A_METHODS = ["WavLM F", "Image CNN (EXP-009a)", "RawNet2 (official)", "Delta-Net (mu-law difference)"]

def save(fig, d, name):
    os.makedirs(f"{M}/{d}", exist_ok=True); fig.tight_layout()
    fig.savefig(f"{M}/{d}/{name}", dpi=150); plt.close(fig)

# ---------- EXP-009a ----------
D = "EXP-009a_image_cnn"
fig, ax = plt.subplots(figsize=(9, 4.2))
for m in A_METHODS:
    y = [ta[m][c]["acc"] * 100 for c in conds]
    ax.plot(x, y, marker="o", lw=2, ms=6, color=COL[m], label=m)
ax.set_xticks(x, [label(c) for c in conds]); ax.set_ylabel("accuracy (%)"); ax.set_ylim(70, 100)
ax.set_title("EXP-009a: accuracy vs compression (same 1,000 ASVspoof 2019 files)")
ax.legend(frameon=False, fontsize=8, loc="lower left"); save(fig, D, "1_accuracy_vs_bitrate.png")

fig, ax = plt.subplots(figsize=(9, 4.2)); w = 0.2
for i, m in enumerate(A_METHODS):
    ax.bar(x + (i - 1.5) * w, [ta[m][c]["eer"] * 100 for c in conds], w - 0.02, color=COL[m], label=m)
ax.set_xticks(x, [label(c) for c in conds]); ax.set_ylabel("EER (%) — lower is better")
ax.set_title("EXP-009a: equal error rate per method"); ax.legend(frameon=False, fontsize=8)
save(fig, D, "2_eer_per_method.png")

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, c in zip(axs, ["single_16k", "single_8k"]):
    fa = [ta[m][c]["false_alarm"] * 100 for m in A_METHODS]; mi = [ta[m][c]["miss"] * 100 for m in A_METHODS]
    xi = np.arange(len(A_METHODS))
    ax.bar(xi - 0.2, fa, 0.38, color="#f08c00", label="false alarm (real→AI)")
    ax.bar(xi + 0.2, mi, 0.38, color="#495057", label="miss (AI→real)")
    ax.set_xticks(xi, ["WavLM F", "Image CNN", "RawNet2", "Δ-Net"]); ax.set_title(f"{label(c)}")
axs[0].set_ylabel("% of files"); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("EXP-009a: error types at heavy compression"); save(fig, D, "3_errors_16k_8k.png")

# ---------- EXP-009b ----------
D = "EXP-009b_selected_features_fusion"
views = ["mel (0–8 kHz)", "fine spectrum (0–4 kHz)", "phase IF (0–4 kHz)"]
fig, axs = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
for ax, key, t in zip(axs, ["stability", "separation", "weight"],
                      ["stability under compression (min corr. clean vs 16k/8k)",
                       "real/fake separation at 16k (|AUC−0.5|×2)", "final weight (0 = dropped)"]):
    im = ax.imshow(np.array(b["bands"][key]), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(3), views, fontsize=8); ax.set_title(t, fontsize=9); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025)
axs[-1].set_xlabel("frequency slice (low → high)")
fig.suptitle("EXP-009b: which picture features survive compression"); save(fig, D, "1_feature_audit.png")

B_METHODS = ["WavLM F (reference)", "Image CNN, selected features", "Combined: average", "Combined: learned weights"]
cb = [c for c in conds if c in tb["Image CNN, selected features"]]; xb = np.arange(len(cb))
fig, ax = plt.subplots(figsize=(9, 4.2))
for m in B_METHODS:
    ax.plot(xb, [tb[m][c]["acc"] * 100 for c in cb], marker="o", lw=2, ms=6, color=COL[m], label=m)
ax.set_xticks(xb, [label(c) for c in cb]); ax.set_ylabel("accuracy (%)"); ax.set_ylim(75, 100)
ax.set_title("EXP-009b: adding the image model to WavLM makes it worse")
ax.legend(frameon=False, fontsize=8, loc="lower left"); save(fig, D, "2_accuracy_fusion_vs_F.png")

lw = tb["Combined: learned weights"]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(xb - 0.2, [lw[c].get("F_only", 0) for c in cb], 0.38, color=COL["WavLM F"], label="only WavLM F correct")
ax.bar(xb + 0.2, [lw[c].get("fused_only", 0) for c in cb], 0.38, color=COL["Combined: learned weights"], label="only combined correct")
ax.set_xticks(xb, [label(c) for c in cb]); ax.set_ylabel("files (of 1,000)")
ax.set_title("EXP-009b: McNemar disagreements (p < 0.001 in every condition)")
ax.legend(frameon=False, fontsize=8); save(fig, D, "3_mcnemar_disagreements.png")

for m in A_METHODS + ["Image CNN, selected features", "Combined: average", "Combined: learned weights"]:
    t = ta.get(m) or tb[m]
    print(f"{m:34s}", " ".join(f"{label(c)}:{t[c]['acc']*100:.1f}/{t[c]['eer']*100:.1f}" for c in conds if c in t))
for c in cb: print(c, "p=", lw[c].get("p_vs_F"), "F_only", lw[c].get("F_only"), "fused_only", lw[c].get("fused_only"))
