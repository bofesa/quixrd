# xrd_processing

Python tool for fitting `I_vs_2th_*.txt` scans frame-by-frame and exporting:

- per-frame fit PNGs
- one per-scan CSV
- `sin2psi_plot.png`
- `sin2psi_fit_params.json`

## Credits and provenance

The sin2psi analysis workflow implemented here is based on
[`materialsguy/Bessy-II-KMC-II-insitu-sin2psi`](https://github.com/materialsguy/Bessy-II-KMC-II-insitu-sin2psi),
archived at [https://doi.org/10.5281/zenodo.17349576](https://doi.org/10.5281/zenodo.17349576).

The NXS/XPAD export functionality builds on SOLEIL XPAD-S140 export routines credited to Pierre-Olivier Renault and the original NXS export-function authors, with local adaptations in this repository for sorted sample trees, metadata handling, and GUI/workflow integration.

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

## GUI and Local Cache

Launch the Tkinter GUI with:

```powershell
python -m nxs_XRD.xrd_processing.gui_app
```

The GUI File menu includes parameter import/export and local-cache controls:

- `Select Local Cache Folder...` chooses where cached input files are stored. The selection is saved between sessions in `%LOCALAPPDATA%\nxs_XRD\gui_settings.json`.
- The default cache folder is `%LOCALAPPDATA%\nxs_XRD\cache`.
- `Create Local Cache` copies filled input paths from the currently selected tab into a timestamped subfolder of the selected cache folder, then updates those GUI input fields to point at the cached copies.
- Output directories are not changed by caching, so processed results still go where the output/data fields specify.
- `Clear Local Cache` deletes the selected cache folder after confirmation.

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

The plot uses `slope` as the gradient and `slope_err` for error bars. It reuses the latest unchanged summary CSV and writes timestamped files such as `sin2psi_export\summaries\sin2psi_scan_summary_<timestamp>.csv` and `sin2psi_export\plots\sin2psi_gradient_vs_<x>_<timestamp>.png`.

To reproduce a plot from a particular saved summary, pass that CSV explicitly:

```python
proc.plot_sin2psi_gradients(
    r"C:\path\to\export",
    x="temperature",
    summary_csv=r"C:\path\to\export\sin2psi_export\summaries\sin2psi_scan_summary_20260715_120000.csv",
)
```

In the GUI, leave `Summary CSV` blank for the current collect/reuse behavior, or browse to a specific summary file for `Gradient` or `Stress` plots.

If elastic constants were supplied during processing or refitting, calculated stress can also be plotted:

```python
proc.plot_sin2psi_stress(r"C:\path\to\export", x="temperature")
```

Stress calculation is optional. Supply `elastic_E` and `elastic_nu` to processing/refitting, plus either a stress-free `stress_reference_two_theta`, a stress-free `stress_reference_d0`, or neither to use the equibiaxial inferred-d0 fallback. `E` and the reported stress use the same units; pass `elastic_E_units` such as `"MPa"` or `"GPa"` to label the saved JSON, summary CSV, and stress plot axis.

## Predicted Peak Overlays

Spectra plots can overlay predicted peak positions either from a direct 2theta list or simple lattice parameters. In the GUI, enable `Show predicted peaks` on the Plotting tab. Programmatically, pass `predicted_peaks` to `Spectrum.plot_Ivs2theta(...)`:

```python
Spectrum(directory=r"C:\path\to\export").plot_Ivs2theta(
    scanNos=[440],
    predicted_peaks={"source": "list", "two_theta_list": "31.8 TiO2, 38.5"},
)

Spectrum(directory=r"C:\path\to\export").plot_Ivs2theta(
    scanNos=[440],
    predicted_peaks={
        "source": "lattice",
        "lattice_type": "fcc",
        "a": 4.05,
        "wavelength": 1.5406,
        "max_index": 6,
        "phase_name": "Al",
    },
)
```

CIF parsing is not implemented in this pass; direct peak lists and built-in lattice formulas are supported.

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
generate_correction_curve(
    r"C:\path\to\reference_export",
    reference_scan=120,
    method="gaussian_process",
    gp_length_scale=0.25,
)
```

By default the correction is fit to the true fitted 2theta angle. Pass `reference_two_theta=...` to fit offsets from a chosen reference angle instead. This writes a JSON file and matching PNG plot under `sin2psi_export\calibrations`. Use the JSON file in a full run or a trend-only refit:

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

## FWHM and Peak-Position Trends

FWHM can be plotted for one exact frame number, several frame numbers, one chi value, or several chi values on the same plot. Chi matching uses a 0.1 degree tolerance:

```python
proc.plot_fwhm_trends(r"C:\path\to\export", x="scan_number", frame_index=2)
proc.plot_fwhm_trends(r"C:\path\to\export", x="scan_number", frame_index=[0, 2, 4])
proc.plot_fwhm_trends(r"C:\path\to\export", x="temperature", chi=5.0)
proc.plot_fwhm_trends(r"C:\path\to\export", x="temperature", chi=[0.0, 5.0, 10.0])
```

Peak position can be plotted in the same way:

```python
proc.plot_peak_position_trends(r"C:\path\to\export", x="scan_number", chi=5.0)
proc.plot_peak_position_trends(r"C:\path\to\export", x="temperature", chi=[0.0, 5.0, 10.0])
proc.plot_peak_position_trends(r"C:\path\to\export", x="temperature", frame_index=[0, 2, 4])
```

FWHM plots use `fwhm_err` as error bars when available. Peak-position plots use `peak_center_err` as error bars when available. Both helpers write timestamped summary CSV and PNG files under `sin2psi_export`.
