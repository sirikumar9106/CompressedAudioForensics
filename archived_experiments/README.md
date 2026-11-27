# Archived experiment launchers

Windows launchers for experiments that were superseded or only tested something. Each runs from the project
folder; results and write-ups are in `EXPERIMENT_LOG.md` and `mapped_results/`.

| Launcher | Experiment | Outcome |
|---|---|---|
| `run_exp006.bat` | EXP-006 robustness of method F (Opus/AAC, wrong dropdown) | passed in the lab |
| `run_exp007.bat`, `run_exp007a.bat` | EXP-007 In-the-Wild reality check | failed: 83.8% of real voices called AI |
| `run_exp008.bat`, `run_exp008b.bat` | EXP-008/008b fine-tuning on diverse real speech | missed modern voice clones |
| `run_exp009a.bat`, `run_exp009b.bat` | EXP-009 spectrogram-image CNN and fusion | rejected |
| `run_exp010b.bat` | EXP-010b standby for re-compressed audio | kept aside, not deployed |
| `run_exp011.bat` | EXP-011 SpoofCeleb test of VoxTrace-Base | led to VoxTrace-Wild |

The two successful models are trained with `train_voxtrace_base.bat` and `train_voxtrace_wild.bat` in the
project folder.
