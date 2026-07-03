# xrd_processing

Python tool for fitting `I_vs_2th_*.txt` scans frame-by-frame and exporting:

- per-frame fit PNGs
- one per-scan CSV
- `sin2psi_plot.png`
- `sin2psi_fit_params.json`

## Usage

```powershell
python -m nxs_XRD.xrd_processing.cli --data-dir "C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\export" --scan 1515 --plot-frames --force
```

Example exclusions:

```powershell
python -m nxs_XRD.xrd_processing.cli --data-dir "C:\path\to\export" --scan 1515 --exclude "0,3,5"
```

Example with an initial peak guess and frame-to-frame tracking:

```powershell
python -m nxs_XRD.xrd_processing.cli --data-dir "C:\path\to\export" --scan 1515 --peak-center 43.2 --track-window 0.4
```

## No-argument launcher

Run `run_workflow.py` directly to use the saved defaults:

```powershell
python C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\nxs_XRD\xrd_processing\run_workflow.py
```

Edit the constants at the top of that file to change the data folder, scan number, exclusions, backup mode, or peak tracking settings.

## Outputs

For scan `N` the tool writes to:

`<data-dir>\sin2psi_export\scan_N\`

Files:

- `scan_N_fits.csv`
- `sin2psi_plot.png`
- `sin2psi_fit_params.json`
- `frame_000_fit.png` etc.

The per-scan CSV includes fit-window metadata (`window_mode`, `seed_center`, `background_lower`, `peak_lower`, `peak_upper`, `background_upper`) so tracked and automatically selected fits can be audited later.

## Gradient Summary

Processed scans can be summarized and plotted across scan number or metadata fields:

```python
from nxs_XRD.xrd_processing import sin2psi_processor as proc

proc.collect_sin2psi_summaries(r"C:\path\to\export")
proc.plot_sin2psi_gradients(r"C:\path\to\export", x="scan_number")
proc.plot_sin2psi_gradients(r"C:\path\to\export", x="temperature")
```

The plot uses `slope` as the gradient and `slope_err` for error bars. It writes timestamped files such as `sin2psi_export\sin2psi_scan_summary_<timestamp>.csv` and `sin2psi_export\sin2psi_gradient_vs_<x>_<timestamp>.png`.
