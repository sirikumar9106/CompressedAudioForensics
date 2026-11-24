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

## EXP-003 · 2026-11-24 · MP3 bitrate sweep: Model A vs Δz probe (exploratory) — **Δz REJECTED at all bitrates**
- **Goal:** map detection vs compression strength; compare official Model A output with base/Δz probes.
- **Data:** EXP-002 subset (600 files, 2019 LA eval A07–A18, test n=200). Conditions: clean, MP3 320/128/64/32/16/8 kbps.
- **Model A** = official RawNet2 classifier score (no training). Probes = logistic regression trained per condition (300 train / 100 val).
- **Observed (test AUC / EER):**
  | cond | Model A | base probe | Δz probe | Model A acc@default thr |
  |---|---|---|---|---|
  | clean | 0.962 / 11% | 0.979 / 5% | 0.956 / 13% | 89.5% |
  | 32k | 0.964 / 9% | 0.979 / 8% | 0.947 / 10% | 89.0% |
  | 16k | 0.954 / 10% | 0.973 / 9% | 0.953 / 13% | 83.0% |
  | 8k | 0.944 / 7% | 0.962 / 11% | 0.934 / 16% | **75.0%** |
- **Findings:** Δz is the worst of the three at every bitrate. Ranking (AUC) degrades only mildly with compression, but Model A's fixed-threshold accuracy collapses 90% → 75% at 8 kbps → operating-point drift, not lost separability.
- **Caveat:** n=200 → EER resolution 1%, ±~4% CI; small EER differences are noise.
- **Decision:** Δz closed. Threshold drift under compression supports the calibration fallback (spec §27/§35).

## EXP-004 · 2026-11-24 · Frozen WavLM front end + linear probe (fair cross-attack) — **PROMISING (promote)**
- **Goal:** test whether a self-supervised front end beats Model A on unseen attacks under compression.
- **Protocol (new, fair):** probe train/val = 2019 LA **train** (A01–A06; 300+100 per class, same attacks Model A saw); test = EXP-002/003 200-file unseen set (A07–A18). Probe trained and tested per condition.
- **Models:** Model A (official RawNet2 score); RawNet2 pooled-embedding probe; WavLM-Base+ (95M) and WavLM-Large (317M), frozen, fp16, mean+std pooling per layer → variants: last layer / all-layer average / best layer chosen on val.
- **Observed (test AUC / EER):**
  | cond | Model A | RawNet2 probe | WavLM-B+ last | **WavLM-B+ layer-avg** | WavLM-L layer-avg |
  |---|---|---|---|---|---|
  | clean | 0.962 / 11% | 0.963 / 11% | 0.985 / 6% | **0.9997 / 1%** | 0.9999 / 1% |
  | MP3 32k | 0.964 / 9% | 0.969 / 10% | 0.985 / 5% | **0.9995 / 2%** | 0.9993 / 1% |
  | MP3 8k | 0.944 / 7% | 0.955 / 11% | 0.988 / 4% | **0.9986 / 1%** | 0.9977 / 2% |
- **Findings:** all-layer averaging ≫ last layer; Base+ ≈ Large (Base+ is 3× cheaper → preferred on 8 GB GPU). At 8 kbps, EER 7% → ~1%. Best-layer-on-val unreliable (val saturated).
- **Caveats:** n=200 test (EER ±~3%); same corpus (VCTK speakers) for train/test → possible dataset shortcuts (silence/loudness, cf. EXP-002); needs cross-dataset check (2021 DF, Phase 6) and larger test.
- **Decision:** promote WavLM-Base+ layer-avg as new static backbone. Next: verify on larger unseen set + shortcut controls before training anything.
