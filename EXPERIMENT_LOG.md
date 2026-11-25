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

## EXP-006 · 2026-11-25 · Real-world robustness of design F (n = 1,000) — **PASSES; one dropdown risk found**
- **Goal:** (1) confirm F > A with a paired test; (2) cost of a wrong dropdown choice; (3) codecs never used in training/calibration (Opus = WhatsApp/Telegram, AAC).
- **Setup:** EXP-005 features/probes reused (same 1,000 unseen-attack files). Opus 64/32/16/8k and AAC 64/32/16k newly encoded; dropdown uses the MP3-fitted calibrator of the same nominal bitrate.
- **(1) McNemar F vs A:** F significantly better at 16k (p = 0.001), 2×16k (p = 0.003), 2×64k (p = 0.004), clean (p = 0.03); not significant at 64k/32k/2×32k and at 8k (p ≥ 0.08). F never significantly worse.
- **(2) Wrong dropdown (F accuracy %, true → picked):** one step off costs ≤ 4.6 pts down to 16k (16k → 32k: 94.0). Picking a **higher** quality than true → false alarms (8k → 16k: 61.9, FA 76%). Picking a **lower** quality than true → missed fakes (64k → 8k: 67.6, miss 65%). "Not sure": 96.2 / 97.3 / 95.4 / 77.9 at 64/32/16/8k.
- **(3) Unseen codecs (F + matching dropdown):** Opus 99.0 / 99.1 / 98.9 / **95.3** (64/32/16/8k); AAC 98.8 / 98.7 / 98.3 (64/32/16k). All ≥ 95%, beats A everywhere; "Not sure" ≥ 96% except Opus 8k (87.7, FA 25%).
- **Findings:** F generalises to WhatsApp-type Opus without retraining. The only serious risk is a large dropdown error, worst when the user picks 8k for good-quality audio (fakes pass as real).
- **Decision:** keep F. App: default dropdown = "Not sure"; warn on 8k choice; add automatic bandwidth check to flag dropdown choices inconsistent with the audio (next step).

## EXP-007 · 2026-11-25 · Real-world baseline: method F on In-the-Wild — **FAILS (83.8% false alarms)**
- **Goal:** measure the deployed model (exact app pipeline) on real-world audio it never saw, after a live recording of a real voice was classified as AI.
- **Data:** In-the-Wild (Müller et al. 2022; real + deepfake speech of 58 public figures from online videos), random balanced 1,000 real + 1,000 fake; clips 0.6–24.7 s (median 3.4 s). Test only.
- **Observed:**
  | variant | accuracy | false alarms (real→AI) | misses (AI→real) | AUC | EER |
  |---|---|---|---|---|---|
  | clean profile (as deployed) | 58.1% | **83.8%** | 0.0% | 0.900 | 16.6% |
  | "I don't know" mode | 66.5% | 62.6% | 4.4% | 0.733 | 32.0% |
  | clean profile, silence trimmed | 54.3% | 91.4% | 0.0% | 0.895 | 16.8% |
- **Findings:** model flags almost everything as AI (mean P(real) on real voices 0.18). Ranking partly survives (AUC 0.90) but the decision boundary is useless out of domain. Trimming silence makes it worse (89 real clips flip to AI) → silence is one learned cue, but not the main one. False alarms are high at every clip length (78–94%).
- **Why:** "real" was learned from one studio corpus (ASVspoof 2019 / VCTK, 600 clips). Any other microphone, room or speaking style falls on the "not real" side. ASVspoof accuracy (98.9%) does not transfer.
- **Decision:** current design is not deployable. Next: retrain with diverse real-world bona fide data (ASVspoof 5 crowdsourced; SpoofCeleb when access is granted), silence-stripping + loudness normalisation, shared corruptions; In-the-Wild stays test-only.

## EXP-008 · 2026-11-25 · Multi-dataset fine-tuning (ASVspoof 5 + Fake-or-Real) — **FALSE-ALARM PROBLEM SOLVED; threshold too strict**
- **Goal:** fix EXP-007 (83.8% of real voices called AI) by training on diverse real speech, avoiding EXP-001..007 pitfalls.
- **Data (train/val, balanced 50/50 inside each dataset, speaker/file-disjoint):** ASVspoof 5 train part aa (crowdsourced, 400 speakers) 6,872 / 730; Fake-or-Real for-norm + for-rerec 8,000 / 800. One random codec copy per clip (MP3/Opus/AAC 8–64 kbps, same for both classes). Silence untouched; 4 s random crops. LibriSeVoc still downloading → EXP-008b.
- **Model:** WavLM-Base+, top 4 transformer layers fine-tuned + learned weights over all 13 layers + attentive statistics pooling + MLP. 4 epochs, AdamW (2e-5 / 1e-3), fp16, RTX 5050 (0.44 s/step, ~30 min). Threshold set on validation for ≤ 5% false alarms (P(real) ≥ 0.876).
- **Observed (never-trained-on test sets):**
  | test set | accuracy | false alarms (real→AI) | misses (AI→real) | AUC | EER |
  |---|---|---|---|---|---|
  | In-the-Wild, same 2,000 clips as EXP-007 | 75.0% | **0.7%** (was 83.8%) | 49.3% (was 0.0%) | **0.971** (was 0.900) | **8.6%** (was 16.6%) |
  | ASVspoof 2019 eval clean (1,000) | 87.3% | 0.2% | 25.2% | 0.982 | 6.5% |
  | ASVspoof 2019 eval MP3 16 kbps | 88.5% | 2.4% | 20.6% | 0.963 | 10.7% |
- **Findings:** real-world false alarms collapse 83.8% → 0.7% and ranking improves (EER halves 16.6% → 8.6%), so diverse real speech fixed the core failure. But the threshold fitted on in-domain validation is now too strict out of domain: half of In-the-Wild fakes pass as real. Lab performance on ASVspoof 2019 drops vs method F (EER 1% → 6.5%), expected since 2019 is no longer in training.
- **Decision:** keep the EXP-008 model as the new base. Next: EXP-008b adds LibriSeVoc (+ ASVspoof 2019 LA train) and fixes the operating point (calibrate on a mixed held-out set incl. compressed real-world-like speech, report EER-threshold results), then re-test on In-the-Wild + the user's own recordings.

## EXP-009a · 2026-11-25 · Spectrogram "fingerprint" image + CNN — **VIABLE BUT WORSE THAN WavLM**
- **Goal:** test the image idea: turn audio into a 3-channel picture and classify it with a CNN; compare with every earlier method on the same 1,000 unseen-attack ASVspoof 2019 files.
- **Model:** picture = R mel (0–8 kHz), G fine spectrum (0–4 kHz, n_fft 1024), B phase instantaneous-frequency deviation (0–4 kHz), each 128×400; ResNet-18 (ImageNet weights), 4,960 balanced 2019 LA train clips, 15 epochs, per-bitrate calibration (same protocol as method F).
- **Observed (accuracy % / EER %):**
  | method | clean | 64k | 32k | 16k | 8k | 2×16k | 2×8k |
  |---|---|---|---|---|---|---|---|
  | WavLM F (EXP-005) | **98.9 / 1.0** | **98.9 / 0.8** | **98.9 / 1.2** | **98.5 / 1.8** | **94.7 / 4.8** | **98.3 / 1.4** | **93.9 / 5.8** |
  | Image CNN | 91.1 / 4.4 | 93.5 / 5.2 | 92.2 / 5.8 | 93.4 / 6.4 | 91.5 / 8.2 | 92.9 / 7.0 | 89.8 / 8.2 |
  | RawNet2 (official) | 92.6 / 6.4 | 92.8 / 7.0 | 93.0 / 6.6 | 92.6 / 6.2 | 91.6 / 8.2 | 92.9 / 6.8 | 90.8 / 9.4 |
  | Δ-Net | 84.5 / 15.2 | 83.4 / 15.4 | 82.3 / 17.6 | 83.8 / 15.6 | 77.5 / 21.8 | 83.5 / 16.0 | 76.7 / 21.6 |
- **Findings:** image CNN ≈ RawNet2 (better EER on clean/64k/32k, equal at 8k), far above Δ-Net, but 5–7 points below WavLM F everywhere. With a single calibration it collapses at 16k/8k (≈ 80%, misses 40%) → like every method, needs bitrate-aware calibration.
- **Decision:** image route is viable but not competitive alone. Test in EXP-009b whether compression-stable picture features add anything when combined with WavLM.
- **Graphs:** `mapped_results/EXP-009a_image_cnn/`.

## EXP-009b · 2026-11-25 · Compression-stable picture features + combination with WavLM — **FAILS; image route closed**
- **Goal:** keep only picture features that (a) stay the same after compression and (b) separate real from fake, weight them, retrain the CNN, and combine it with WavLM F.
- **Feature audit (16 frequency slices per view):** stability = min correlation clean vs 16k/8k; separation = |AUC−0.5|×2 at 16k; weight = stability × separation, kept if ≥ 25% of max. Kept: mel slices 1–9, fine-spectrum 0–8, phase slice 0 only. **Phase barely survives compression** (mean stability 0.08); high frequencies are destroyed by every codec.
- **Model:** ResNet-18 on the weighted, selected picture; trained with clean + 32/16/8k copies. Combination: (i) average of log-odds, (ii) per-bitrate learned weights; McNemar test vs F.
- **Observed (accuracy % / EER %):**
  | method | clean | 64k | 32k | 16k | 8k | 2×8k |
  |---|---|---|---|---|---|---|
  | WavLM F alone | **98.9 / 1.0** | **98.9 / 0.8** | **98.9 / 1.2** | **98.5 / 1.8** | **94.7 / 4.8** | **93.9 / 5.8** |
  | selected-feature CNN | 84.8 / 11.2 | 87.3 / 10.3 | 87.4 / 11.0 | 86.6 / 12.2 | 86.7 / 13.4 | 87.0 / 12.4 |
  | F + CNN (average) | 87.6 / 4.0 | 88.8 / 3.0 | 91.1 / 4.6 | 92.2 / 4.0 | 90.8 / 6.8 | 91.0 / 7.6 |
  | F + CNN (learned weights) | 85.8 / 4.2 | 88.0 / 3.0 | 88.9 / 5.6 | 89.5 / 5.6 | 88.6 / 8.4 | 88.8 / 8.6 |
- **Significance:** combination worse than F in all 9 conditions (McNemar p < 0.001; files only F got right 93–139 vs only combined 6–42).
- **Why:** removing unstable features removed most of the useful signal (CNN drops 91 → 85–87%); the CNN's errors are not complementary to WavLM, so mixing in its scores only adds noise and pulls misses up (miss 21–25% on clean).
- **Decision:** image/spectrogram route closed with evidence. Keep WavLM; continue real-world work with EXP-008b.
- **Graphs:** `mapped_results/EXP-009b_selected_features_fusion/`.

## EXP-008b · 2026-11-25 · EXP-008 + LibriSeVoc + ASVspoof 2019 — **BETTER, BUT FAILS ON MODERN VOICE CLONING**
- **Goal:** cut EXP-008's misses by adding more fake families (LibriSeVoc: 6 neural vocoders on LibriTTS; ASVspoof 2019 LA train) and fixing the operating point.
- **Data:** 27,472 train / 2,890 val clips, 4 datasets, balanced 50/50 inside each, one random codec copy per clip. Same model and recipe as EXP-008 (4 epochs).
- **Observed:** val EER 2.4%. In-the-Wild (same 2,000 clips) **84.5%** acc, false alarms 2.1%, misses 28.8%, EER 8.1% (EXP-008: 75.0 / 0.7 / 49.3 / 8.6). ASVspoof 2019 clean 97.5% (EER 2.4%), MP3 16k 93.7% (EER 3.6%), up from 87.3 / 88.5%.
- **Live test (user, app with new per-file diagnostics):** own real voice now correctly "human" (fixes EXP-007), but **all 3 AI clips from a modern online voice-cloning tool scored "100% human"**.
- **Diagnostics (occlusion of frequency bands and WavLM layers):** AI and real scores fully overlapped (median log-odds 11–13 for both). The verdict was driven by WavLM's top layer (speech content/prosody, +4 to +6); low "texture" layers voted AI for both classes and were outvoted. No frequency band separated the classes.
- **Why:** every fake in training came from older TTS/vocoders (2019–2023); modern zero-shot cloning sounds natural at the content level, which the model had learned to equate with "human".
- **Decision:** add modern TTS/cloning fakes (MLAAD) and give the low layers their own vote → EXP-010.
- **Graphs:** `mapped_results/EXP-008b_more_data/`.

## EXP-010 · 2026-11-25 · Modern TTS (MLAAD) + 8 fine-tuned layers + "texture" branch — **SUCCESS (deployed)**
- **Goal:** detect modern voice cloning without losing real-world false-alarm control.
- **Data:** EXP-008b mix + MLAAD v9 English (143 TTS/cloning systems, 200 clips each). 31 systems held out entirely (incl. ElevenLabs ×2, OpenAI TTS-1 HD, MiniMax 2.6 HD, Qwen3-TTS, VibeVoice); MLAAD fakes paired with unused LibriSeVoc real audiobook speech so "audiobook" is not a shortcut. 39,472 train / 4,090 val clips, codec copies as before.
- **Model:** WavLM-Base+, top 8 layers fine-tuned (was 4); main branch = learned weights over all 13 layers + attentive pooling + MLP; new texture branch = own weights over layers 0–6 + attentive pooling + MLP; logits summed. 3 epochs, batch 12, 0.13 s/step (~25 min). Balanced (EER) threshold deployed.
- **Observed (never trained on):**
  | test | accuracy | real→AI | AI→real | EER |
  |---|---|---|---|---|
  | In-the-Wild 2,000 (EXP-008b → EXP-010) | 84.5 → **94.6%** | 2.1 → 7.1% | 28.8 → **3.7%** | 8.1 → **5.1%** |
  | MLAAD, 31 unseen systems | **98.5%** | 0.1% | 3.0% | 1.2% |
  | ASVspoof 2019 clean / MP3 16k | 97.9 / 97.0% | 0.4 / 1.2% | 3.8 / 4.8% | 1.8 / 2.8% |
- **Per unseen system:** 26 of 31 detected ≥ 90% (most 100%); hardest: ElevenLabs Turbo 2.5 87%, ElevenLabs v2 84%, MiniMax 83%, MatchaTTS 75%.
- **Live test (user-reported, WhatsApp-shared):** 4 AI voices + 4 real voices from different speakers, all judged correctly — first model to pass the live test.
- **Caveats:** MLAAD held-out reals are audiobooks (same family as training reals) → In-the-Wild is the more honest real-world number. False alarms rose 2.1 → 7.1%.
- **Decision:** EXP-010 is the deployed model (web app default). Backup for extreme re-compression evaluated separately (EXP-010b, not deployed).
- **Graphs:** `mapped_results/EXP-010_modern_tts/`.
