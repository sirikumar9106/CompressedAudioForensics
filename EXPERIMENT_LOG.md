# Experiment Log

## EXP-001 · 2026-11-24 · Δ-Net Phase 2 pilot (pre-registered GO/NO-GO) — **SHUT DOWN (NO-GO)**
- **Goal:** test whether Δz = φ(y) − φ(μ-law(y)) separates bona fide vs spoof after lossy compression.
- **Model:** frozen ASVspoof 2021 DF RawNet2 baseline (Model A) as encoder φ, frame features [29×128]; logistic-regression probe (C tuned on val).
- **Data:** ASVspoof 2019 LA **dev**, 600 files (150/50/100 per class train/val/test), seed 42; attacks A01–A06 (seen by Model A in training).
- **Degradation:** D_platform = MP3 32 kbps (FFmpeg/libmp3lame) → 16 kHz mono, 64,600-sample center crop, peak-norm 0.95.
- **Perturbation:** 8-bit μ-law (μ=255); control: Gaussian noise σ=0.01.
- **Expected:** Δz AUC ≥ 0.70, ≥ 0.08 above noise control, and > base embedding.
- **Observed (held-out AUC):** Δz 0.913 · base 0.971 · perturbed 0.967 · full 0.952 · Δz(noise) 0.956 · RMS 0.43 · duration 0.56. Confound r < 0.11.
- **Why it failed:** Δz < base embedding (pre-registered NO-GO rule); random noise ≥ μ-law → no structured-quantization benefit; Δz redundant (full < base).
- **Caveat:** in-domain attacks; base already near ceiling (0.97) → little headroom for Δz.
- **Decision:** Δ-Net not promoted. Exploratory follow-up on unseen attacks → EXP-002.

## EXP-002 · 2026-11-24 · Δ-Net pilot on unseen attacks (exploratory) — **SHUT DOWN (NO-GO)**
- **Goal:** re-test EXP-001 where Model A has no prior knowledge of the attacks (more headroom for Δz).
- **Model / degradation / perturbation / probe:** identical to EXP-001.
- **Data:** ASVspoof 2019 LA **eval**, attacks A07–A18 (A16, A19 excluded = copies of seen A04/A06); 600 files, same 150/50/100 split, seed 42.
- **Expected:** Δz AUC ≥ 0.70, ≥ 0.08 above noise, > base embedding.
- **Observed (held-out AUC):** Δz 0.947 · base 0.979 · perturbed 0.980 · full 0.980 · Δz(noise) 0.969 · RMS 0.23 (i.e. 0.77 inverted) · duration 0.68.
- **Confounds:** Δz score correlates with RMS (r = 0.52) and duration (r = 0.25) → partly a loudness/length cue, not codec response.
- **Why it failed:** Δz < base again; noise ≥ μ-law again; adding Δz gives no gain over base (0.980 vs 0.979); confound limit (r < 0.20) violated.
- **Caveat:** probe is trained on the same unseen-attack pool, so "unseen" holds for the encoder only.
- **Decision:** Δz with frozen RawNet2 + μ-law offers no information beyond the static embedding. Δ-Net (as specified) closed.
