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

## EXP-004b · 2026-11-24 · Confidence stability across bitrates (same files) — **WavLM UNSTABLE below 32 kbps**
- **Goal:** check if a file's calibrated confidence holds as the *same* audio is compressed harder (deployment: codec unknown).
- **Setup:** EXP-004 test set (200 unseen-attack files) scored at clean, MP3 320/128/64/32/16/8k. Detectors Platt-calibrated on their own 2019 LA train-split val: Model A (clean cal), WavLM-B+ layer-avg probe trained on clean, and on MP3 32k. One fixed model per detector (no per-bitrate retraining).
- **Observed (8 kbps vs clean):**
  | detector | AUC | acc@0.5 | verdict flips | mean P(real) on real voices |
  |---|---|---|---|---|
  | Model A | 0.944 | 87.0% | 12.5% | 0.94 → 0.78 |
  | WavLM probe (train clean) | 0.978 | **50.5%** | **49%** | 0.98 → **0.01** |
  | WavLM probe (train 32k) | 0.976 | **51.0%** | **50.5%** | 0.99 → **0.05** |
- **Findings:** stable down to 32k (flips ≤ 5%). At 16k and 8k, WavLM pushes nearly every REAL voice to "AI" (massive false alarms) though ranking (AUC) stays ≥ 0.976. Model A is less accurate but far more confidence-stable.
- **Why:** probe only saw ≥ 32k audio; 8k MP3 (band-limited ~3–4 kHz) is out-of-distribution, so the score shifts as a whole. EXP-004's 8k result used an 8k-trained probe and hid this.
- **Decision:** WavLM front end kept, but must be trained on a multi-bitrate mix (incl. 8–16k) and/or given per-condition calibration + an "uncertain" output. → next experiment.

## EXP-004c · 2026-11-24 · Re-compression / forwarding (64k → X) — **final bitrate dominates; re-compression adds little**
- **Goal:** check whether a voice note compressed twice (sent at 64k, forwarded at X) keeps the same confidence as single compression.
- **Setup:** EXP-004b test set (200 unseen-attack files) and detectors/calibration unchanged. Double = MP3 64k → decode → MP3 X; X ∈ {320,128,64,32,16,8}k. Compared with single MP3 X.
- **Observed (WavLM-B+ probe trained on 32k; single → double):**
  | X | acc@0.5 | P(real) on real | confidence gap real−fake |
  |---|---|---|---|
  | 64k | 98.5% → 98.5% | 0.991 → 0.986 | 0.970 → 0.965 |
  | 32k | 98.5% → 98.0% | 0.969 → 0.962 | 0.966 → 0.958 |
  | 16k | 93.0% → 90.5% | 0.835 → 0.797 | 0.833 → 0.794 |
  | 8k | 51.0% → 51.5% | 0.045 → 0.038 | 0.045 → 0.038 |
- **Model A:** AUC drops ~0.01 with re-compression (0.971 → 0.961 at 64k); accuracy ≈ unchanged (87–91%).
- **Findings:** a second encode costs ≤ 2.5 pts accuracy and ≤ 0.04 confidence; the *last* (lowest) bitrate decides behaviour. The 8–16k collapse from EXP-004b is unchanged.
- **Decision:** re-compression is not a separate problem; fixing low-bitrate calibration (EXP-005: multi-bitrate training + bitrate-aware calibration) covers it.

## EXP-004d · 2026-11-24 · Small tweaks for 16 kbps (A/B/C), n = 1,000 — **A PASSES (winner); B partial; C fails**
- **Goal:** raise 16 kbps accuracy (single + re-compressed) to ≥ 95% without losing > 0.5 pts at 32/64k.
- **Test:** 1,000 new unseen-attack files (500/500, 2019 LA eval A07–A18), clean + MP3 64/32/16/8k single + re-compressed (64k → X). Train/val: 2019 LA train 300+100/class. Frozen WavLM-B+ layer-avg.
- **Variants:** BASE = 32k probe, one calibrator (EXP-004c) · **A** = BASE + per-bitrate calibrator picked by final bitrate (app dropdown) · **B** = probe trained on clean+64+32+16k · **C** = BASE + parallel 4 kHz low-pass WavLM branch.
- **Observed (accuracy, 95% CI):**
  | cond | BASE | A | B | C |
  |---|---|---|---|---|
  | 64k / 2×64k | 98.2 / 98.4 | 98.4 / 97.9 | 98.0 / 98.0 | 99.1 / 98.8 |
  | 32k / 2×32k | 98.1 / 97.8 | 98.1 / 97.8 | 97.7 / **97.0** | 98.1 / 97.7 |
  | **16k / 2×16k** | 91.9 / 91.1 | **97.1 [96.1–98.1] / 96.7 [95.6–97.8]** | **97.5 / 96.9** | 92.7 / 92.2 |
  | 8k / 2×8k | 51.2 / 51.9 | **94.2 / 93.8** | 61.1 / 62.3 | 50.6 / 50.8 |
- **Errors at 16k:** false alarms (real → AI) BASE 16.0% → A 5.4%, B 4.2%; misses stay ≤ 1.4%.
- **Verdict:** A meets every criterion (16k ≥ 95%; 32/64k within 0.5 pts) and also lifts **8k from 51% to 94%**. B hits 16k but loses 0.8 pts at 2×32k (rule fail) and does little at 8k. C does not help (16k 92.7%).
- **Why A works:** AUC stays ≥ 0.987 at every bitrate → ranking survives; only the score offset drifts, which a 2-parameter calibrator per bitrate removes.
- **Caveat:** A relies on the user's bitrate choice being roughly right; a wrong choice is untested → next: EXP-004e (mismatched dropdown).
- **Decision:** adopt A (bitrate-aware calibration) as the deployment design.

## EXP-005 · 2026-11-24 · Challengers vs EXP-004d winner (A), n = 1,000 — **F (RawBoost-aug + per-bitrate cal) BEST**
- **Goal:** check whether A can be beaten before adopting it. Same 1,000 test files, 9 conditions, frozen WavLM-B+ layer-avg.
- **Variants (all with per-bitrate calibration except U):** A = 32k linear probe (reference) · D = mixed-bitrate (clean/64/32/16k) probe · E = 1-hidden-layer MLP (256) · F = 32k probe trained on originals + RawBoost-style copies (random band-pass coloration + 10–35 dB noise) · U = "Not sure" mode (D probe, one pooled calibrator).
- **Observed (accuracy %, 95% CI):**
  | cond | A | D | E | **F** | U |
  |---|---|---|---|---|---|
  | clean | 97.9 | 98.0 | 96.3 | **98.9** | 96.5 |
  | 64k / 2×64k | 98.4 / 97.9 | 97.9 / 97.9 | 97.2 / 96.2 | **98.9 / 99.1** | 96.2 / 96.8 |
  | 32k / 2×32k | 98.1 / 97.8 | 97.8 / 97.5 | 96.8 / 96.2 | **98.9 / 98.4** | 97.3 / 97.0 |
  | 16k / 2×16k | 97.1 / 96.7 | 97.1 / 96.4 | 96.2 / 95.2 | **98.5 [97.7–99.2] / 98.3 [97.5–99.1]** | 95.4 / 95.8 |
  | 8k / 2×8k | 94.2 / 93.8 | 94.7 / 93.7 | 94.2 / 93.2 | 94.7 / 93.9 | 77.9 / 77.0 |
  | avg / worst | 96.9 / 93.8 | 96.8 / 93.7 | 95.7 / 93.2 | **97.7 / 93.9** | 92.2 / 77.0 |
- **Findings:** F beats A in all 9 conditions (+0.1 to +1.6 pts; largest at 16k). D ≈ A (mixed-bitrate training adds nothing once calibration is per-bitrate). E (MLP) is worse than linear → no benefit from a non-linear head on 600 training files. U ("Not sure") holds ≥ 95% down to 16k but drops to ~77% at 8k.
- **Caveat:** individual CIs overlap slightly at 16k; the case for F rests on consistent wins across all conditions. Paired significance test to be added.
- **Decision:** deployment design = **F** (RawBoost-augmented WavLM probe + per-bitrate calibration chosen by the dropdown); "Not sure" uses U and should warn the user when the audio looks like ≤ 8 kbps.
