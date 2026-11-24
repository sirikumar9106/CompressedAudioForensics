# CompressedAudioForensics — Δ-Net

**Detecting AI-cloned voices in compressed, real-world audio (WhatsApp, Telegram, phone calls).**

## Why this project exists
Online voice scams are growing fast. With a few seconds of someone's voice, free AI tools can clone it, and
fraudsters use these clones to call or send voice notes pretending to be a family member, a boss, a bank officer
or the police, usually asking for money urgently.

The protections people rely on today are not dependable:
- **Human judgement** fails: modern clones sound natural, especially over a noisy call.
- **Existing AI detectors** are trained on clean studio recordings. Real scam audio arrives through WhatsApp,
  Telegram or phone networks that compress it heavily, wiping out the tiny clues those detectors look for.
  The result is **false alarms** (a real relative flagged as fake) and **missed fakes** (a clone marked as real).
- **Comparing with the original** is impossible: the victim only ever has the one file they received.

## What this project does
CompressedAudioForensics builds a detector designed for the audio people *actually* receive. Instead of looking
for clues that compression has already destroyed, it deliberately compresses the received file once more in a
controlled way and measures **how the voice reacts**. The idea being tested: real human speech and AI-generated
speech respond differently to further degradation. The tool returns a probability, how confident it is, and
practical safety advice (call back on a known number, never send money based on a voice message alone).

The project is run as a proper research study: a pre-registered go/no-go pilot, a locked test set that is never
seen during development, comparison against standard baselines, and confidence intervals on every result.

## The approach (technical)
Δ-Net works on **the single compressed file you received**. It applies a controlled, deterministic secondary
degradation to that same file (8-bit μ-law companding), passes both versions through one shared encoder, and
studies **how the internal representation shifts** (Δz = φ(y) − φ(y′)). The hypothesis under test: real and
synthetic speech react differently to further degradation, even after heavy compression.
The output is a calibrated probability, a decision margin, and plain-language safety advice
(“call back on a known number; don’t send money”).

> Research status: the original Δ-Net hypothesis was rejected by its pre-registered pilot (kept as a documented
> negative result). The project now builds on a WavLM front end; see Results.
> Results are probabilistic and are not legal evidence.

## Results so far
All tests use only the compressed file (no clean original), on attacks the detector never saw in training.
Full details in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md); graphs in [`mapped_results/`](mapped_results/).

| Exp | What was tested | Outcome |
|---|---|---|
| EXP-001 | Δ-Net idea: compare file vs its own μ-law re-degraded copy (seen attacks) | NO-GO: Δz AUC 0.913 < plain embedding 0.971 |
| EXP-002 | Same, on unseen attacks | NO-GO: Δz 0.947 < 0.979; partly a loudness cue |
| EXP-003 | Official ASVspoof RawNet2 across MP3 320→8 kbps | Ranking holds (AUC 0.96→0.94) but default-threshold accuracy drops 90%→75% |
| EXP-004 | Frozen **WavLM** front end vs official model (fair cross-attack test) | WavLM wins: EER ~1–2% vs 7–11% at every bitrate |
| EXP-004b | Same files scored at every bitrate with ONE model (unknown-codec use) | Stable to 32 kbps; at 8–16 kbps WavLM flags most real voices as fake → needs bitrate-aware calibration |
| EXP-004c | Re-compression (forwarded voice notes: 64 kbps → X) | Second encode costs ≤ 2.5 pts; the final bitrate decides behaviour |

**Current direction:** WavLM-based detector trained on mixed bitrates, with the user's bitrate selection
choosing the calibration, plus an "uncertain" verdict when confidence is low.

## Pipeline
```
uploaded file y ──► shared RawNet2 encoder ──► z_y  ─┐
       └─ μ-law(y) = y′ ──► same encoder ──► z_y′ ───┼─► [mean z_y | mean z_y′ | mean Δz | std Δz | cos-dist] (1,025-d)
                                                     └─► MLP ──► Platt calibration ──► probability + advice
```

## Setup
```bash
# Windows: double-click setup.bat  (creates .venv, picks the right CUDA build, installs FFmpeg)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pytest tests/
```
Datasets and pretrained weights are **not** included — see the bottom of `requirements.txt` for links and paths.

## Run
| Phase | Command |
|---|---|
| 0 Environment / audit | `python experiments/run_phase0_audit.py` |
| 1 Manifests (+ MP3 cache) | `python experiments/prepare_data.py [--cache train dev]` |
| 2 GO/NO-GO pilot | `python experiments/run_phase2_pilot.py` (or `run_pilot.bat`) |
| 3 Static baseline B | `python experiments/run_training.py --model baseline_b --config config/baseline_rawnet2.yaml` |
| 4 Δ-Net MVP | `python experiments/run_training.py --model deltanet --config config/delta_net_mvp.yaml` |
| 5 Ablations | `... --model deltanet --set perturbation=gaussian` (etc.) |
| EXP-003 | `python experiments/run_bitrate_sweep.py` |
| EXP-004 | `python experiments/run_ssl_probe.py` |
| EXP-004b | `python experiments/run_confidence_stability.py` |
| EXP-004c | `python experiments/run_double_compression.py` |
| Graphs | `python experiments/plot_results.py` → `mapped_results/` |

## Repository layout
```
config/        YAML configs (pilot, baseline, Δ-Net)
src/data/      audio standardisation, FFmpeg codec cache, manifests, dataset
src/models/    μ-law engine, RawNet2 backbone, 1,025-d differential head, Δ-Net
src/training/  weighted CE, training loop
src/evaluation EER / AUC / bootstrap CI (pure NumPy), Platt calibration
src/app/       single-file inference engine (no clean reference allowed)
src/utils/     locked-benchmark guard
experiments/   phase + experiment scripts, plot_results.py
mapped_results/ graphs per experiment (EXP-001 … EXP-004c)
tests/         shape, weight-sharing, quantisation, metric and isolation tests
```

## Notes
- Labels: 1 = bona fide, 0 = spoof. Scores: higher = more bona fide.
- Native RawNet2 frame features are [B, 29, 128]; Δ-Net adds a learnable 128→256 projection to keep the
  1,025-d contract (`proj_dim: null` gives a 513-d variant). The frozen pilot uses native 128-d features.
- The 2021 DF evaluation subset is locked: code refuses to load it before Phase 6.

## Data & licences
ASVspoof 2019 / 2021 data are distributed by their organisers under their own licences and are not redistributed here.
