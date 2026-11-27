# Tools

Setup, dataset download and housekeeping scripts (run from anywhere; they switch to the project folder).

- `setup_env.ps1` - called by `setup.bat` in the project folder
- `download_datasets.bat`, `download_mlaad.bat`, `download_spoofceleb_eval.bat` - datasets (Hugging Face datasets need `HF_TOKEN`)
- `cleanup_unused_data.bat` - deletes dataset parts and caches no experiment needs
