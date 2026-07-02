# nxs_processing

Python tool for fitting `I_vs_2th_*.txt` scans frame-by-frame and exporting:

- per-frame fit PNGs
- one per-scan CSV
- `sin2psi_plot.png`
- `sin2psi_fit_params.json`

## Usage

```powershell
python -m nxs_XRD.nxs_processing.cli --data-dir "C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\export" --scan 1515 --plot-frames --force
```

Example exclusions:

```powershell
python -m nxs_XRD.nxs_processing.cli --data-dir "C:\path\to\export" --scan 1515 --exclude "0,3,5"
```

## No-argument launcher

Run `run_workflow.py` directly to use the saved defaults:

```powershell
python C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\nxs_XRD\nxs_processing\run_workflow.py
```

Edit the constants at the top of that file to change the data folder, scan number, exclusions, or backup mode.

## Outputs

For scan `N` the tool writes to:

`<data-dir>\sin2psi_export\scan_N\`

Files:

- `scan_N_fits.csv`
- `sin2psi_plot.png`
- `sin2psi_fit_params.json`
- `frames\frame_000_fit.png` etc.
