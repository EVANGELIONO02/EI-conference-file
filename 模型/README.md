# Wind power reconstruction experiment

Run from this directory with `conda run -n diffusion312 python src/run_experiment.py --config config.json`.

The main experiment uses mixed masks only, compares weather-conditioned and weather-free diffusion, and evaluates all four power fields. Result rows report `abs_error_max`, `abs_error_min`, and `MAE` for absolute-error comparison. Intermediate files and checkpoints are written under `work_v2/`; the four final CSV files are written under `results/`.

Run with the `diffusion312` environment. Do not launch the script with the system Python if it does not contain NumPy and PyTorch:

```powershell
conda run -n diffusion312 python src/run_experiment.py --config config.json
```
