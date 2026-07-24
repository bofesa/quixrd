# quixrd

Quick XRD profile processing tools for exported 1D profiles, SOLEIL XPAD NXS export, plotting, 2theta calibration, and sin2psi analysis. The processing tools fit `I_vs_2th_*.txt` scans frame-by-frame and export:

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
python -m quixrd.xrd_processing.cli --data-dir "C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\export" --scan 1515 --plot-frames --force
```

Example exclusions:

```powershell
python -m quixrd.xrd_processing.cli --data-dir "C:\path\to\export" --scan 1515 --exclude "0,3,5"
```

Example with an initial peak guess and frame-to-frame tracking:

```powershell
python -m quixrd.xrd_processing.cli --data-dir "C:\path\to\export" --scan 1515 --peak-center 43.2 --track-window 0.4
```

## No-argument launcher

Run `run_workflow.py` directly to use the saved defaults:

```powershell
python C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\quixrd\xrd_processing\run_workflow.py
```

Edit the constants at the top of that file to change the data folder, scan number, exclusions, backup mode, or peak tracking settings.

## GUI and Local Cache

Launch the Tkinter GUI with:

```powershell
python -m quixrd.xrd_processing.quixrd_gui_app
```

The GUI File menu includes parameter import/export and local-cache controls:

- `Select Local Cache Folder...` chooses where cached input files are stored. The selection is saved between sessions in `%LOCALAPPDATA%\quixrd\gui_settings.json`.
- The default cache folder is `%LOCALAPPDATA%\quixrd\cache`.
- `Use Local Cache` in the File menu copies only the needed exported `I_vs_2th` TXT scan files into the cache on demand, then reads from the cached copies.
- Output directories are not changed by caching, so processed results still go where the original output/data fields specify.
- `Clear Local Cache` deletes the selected cache folder after confirmation.

The GUI Calibration menu includes 2theta calibration controls:

- `2theta Calibration...` opens the separate calibration window.
- `Select 2theta Calibration File...` remembers a quixrd 2theta-axis calibration JSON between sessions.
- `Apply 2theta Calibration by Default` applies that calibration during extraction post-processing, spectra plotting, and sin2psi peak processing.
- Corrected TXT exports include `# TwoTheta Correction: <calibration_json_path>`. Files with a non-empty `TwoTheta Correction` metadata field are treated as already corrected.

GUI Batch extraction writes both per-frame TXT files and all-frame CSV files for the selected scans.

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
from quixrd.xrd_processing import sin2psi_processor as proc

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

## 2theta Calibration

LaB6 or custom lattice references can be used to generate a 2theta-axis calibration JSON. This correction is applied before peak fitting and is separate from sin2psi chi/psi correction.

In the GUI, use `Calibration > 2theta Calibration...`. It accepts:

- one SOLEIL XPAD NXS delta scan;
- one or more exported TXT delta-frame files;
- one all-frame CSV export.

For LaB6, the GUI autofills cubic `a = 4.25695 Å`. The output folder receives:

- a combined `I_vs_2th` TXT profile with metadata;
- a combined CSV profile;
- a profile plot with predicted hkl/multiplicity markers;
- an offset-polynomial and Caglioti FWHM plot;
- a calibration JSON that quixrd can apply later.

Programmatically:

```python
from quixrd.xrd_processing import twotheta_calibration as cal

result = cal.build_twotheta_calibration(
    [r"C:\path\I_vs_2th_120_delta_0.txt", r"C:\path\I_vs_2th_120_delta_1.txt"],
    source_type="txt",
    output_dir=r"C:\path\calibration",
    material="LaB6 (cubic, Pm-3m)",
    wavelength=1.0,
    polynomial_degree=2,
)

cal.apply_calibration_to_exported_files(
    r"C:\path\to\export",
    result["path"],
    scans=[440, 441],
)
```

When combining overlapping delta frames, quixrd interpolates onto a common 2theta grid and averages overlapping intensities.

## Refit Sin2psi Trends

Existing peak fits can be reused while changing only the sin2psi regression exclusions:

```python
from quixrd.xrd_processing.run_workflow import refit_sin2psi_trends

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
from quixrd.xrd_processing.run_workflow import generate_correction_curve

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

Multiple correction JSON files may be supplied as a Python list, or as semicolon-separated paths in the GUI. The closest correction is selected once per scan from the uncorrected fitted peak position, and the correction is applied as an absolute 2theta offset with no angle-dependent scaling.

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

## Peak Analysis and Spinodal Splitting

The GUI `Peak Analysis` tab can test whether one selected peak is better represented by one pseudo-Voigt peak or by two overlapping pseudo-Voigt peaks. It can also track the selected peak or peak pair across scans.

For `delta` scans, leave the frame concept aside: quixrd combines all `I_vs_2th_<scan>_delta_<frame>.txt` frames into one homogenised profile by interpolating to a common 2theta grid and averaging overlapping regions. Other scan types use the selected exact frame index.

Programmatically:

```python
from quixrd.xrd_processing import spinodal_peak_analysis as spinodal

result = spinodal.run_peak_series(
    r"C:\path\to\export",
    scans=[440, 441, 442],
    scan_type="chi",
    frame_index=0,
    peak_center=40.0,
    fit_window=0.7,
    fit_mode="compare",
)
```

Outputs are written under `<data-dir>\peak_analysis`:

- `peak_series_<timestamp>.csv`
- `peak_series_trends_<timestamp>.png`
- `peak_series_params_<timestamp>.json`
- `diagnostics_<timestamp>\peak_series_diagnostics_<timestamp>.png`

For comparison fits, `delta_bic = BIC(single peak) - BIC(two peaks)`. BIC is the Bayesian Information Criterion: it compares fit quality while penalising extra fitted parameters, so the two-peak model must improve the residuals enough to justify its added complexity. Positive values favour two peaks, but the magnitude should be read alongside the fitted separation, relative peak intensity, and diagnostic plots rather than as a hard threshold. The CSV stores `selected_model` as quixrd's assessment of which model is preferred. The trend plot shows both the one-peak fit in orange and the two-peak fit in green where available; non-selected model points are faded. The BIC panel also shows the minor/major peak-height ratio when two-peak results exist. The diagnostic plot shows representative scans by default. If every scan diagnostic is enabled, one image is saved per successful scan so you can flick through them; each image still shows only the final one-peak and/or two-peak model fits, not intermediate optimiser attempts.

To replot an existing run on a different metadata x-axis without refitting, use the GUI `Existing results CSV` field or call:

```python
spinodal.plot_peak_series_from_csv(r"C:\path\to\peak_series_20260724_120000.csv", x="temperature")
```
