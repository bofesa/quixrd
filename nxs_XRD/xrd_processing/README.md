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

The plot uses `slope` as the gradient and `slope_err` for error bars. It reuses the latest unchanged summary CSV and writes timestamped files such as `sin2psi_export\sin2psi_scan_summary_<timestamp>.csv` and `sin2psi_export\sin2psi_gradient_vs_<x>_<timestamp>.png`.

## Refit Sin2psi Trends

Existing peak fits can be reused while changing only the sin2psi regression exclusions:

```python
from nxs_XRD.xrd_processing.run_workflow import refit_sin2psi_trends

refit_sin2psi_trends(
    exclude_frames=[0, 1],
    exclude_chi_ranges=[(0, 5)],
    exclude_sin2psi_ranges=[(0.95, 1.0)],
    auto_exclude=True,
    auto_exclude_sigma=3.0,
)
```

This reads each existing `scan_N_fits.csv`, rewrites `excluded_from_sin2psi`, and regenerates only `scan_N_sin2psi_plot.png` and `sin2psi_fit_params.json`; it does not refit peaks.

## Sin2psi Correction

A stress-free reference scan can be used to generate a chi/psi correction curve:

```python
from nxs_XRD.xrd_processing.run_workflow import generate_correction_curve

generate_correction_curve(r"C:\path\to\reference_export", reference_scan=120, degree=2)
```

This writes a JSON file and matching PNG plot under `sin2psi_export\calibrations`. Use the JSON file in a full run or a trend-only refit:

```python
sin2psi_scans_fit(SCANS, correction_json=r"C:\path\to\sin2psi_correction_scan_120_20260706_120000.json")

refit_sin2psi_trends(
    scans=SCANS,
    correction_json=r"C:\path\to\sin2psi_correction_scan_120_20260706_120000.json",
)
```

When a correction is applied, `scan_N_fits.csv` includes `peak_center_uncorrected`, `sin2psi_correction`, and `peak_center_corrected`. The per-scan plot is saved as `scan_N_sin2psi_plot.png`; corrected plots also show uncorrected points at low opacity.

## Processing Logs

`run_workflow.py` writes one batch-level JSON log after each analysis run:

`<data-dir>\sin2psi_export\logs\sin2psi_processing_params_<timestamp>.json`

To replay from a saved log, pass it to the workflow helper:

```python
sin2psi_scans_fit(SCANS, params_json=r"C:\path\to\sin2psi_processing_params_20260706_120000.json")
```

Imported values act as defaults; the current `run_workflow.py` constants and function arguments override them.

## FWHM Trends

FWHM can be plotted for one exact frame number or one exact chi value:

```python
proc.plot_fwhm_trends(r"C:\path\to\export", x="scan_number", frame_index=2)
proc.plot_fwhm_trends(r"C:\path\to\export", x="temperature", chi=5.0)
```

The plot uses `fwhm_err` as error bars when available and writes timestamped `sin2psi_fwhm_summary_<timestamp>.csv` and `sin2psi_fwhm_vs_<x>_<selector>_<timestamp>.png` files.
