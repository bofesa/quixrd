from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nxs_XRD.xrd_processing import sin2psi_processor as proc
else:
    from . import sin2psi_processor as proc


DATA_DIR = r"C:\Users\bosa\local_code\33_130113 HEN\\"
SCANS = range(2200, 2561+1)
# SCANS = [194, 204]
EXCLUDE_FRAMES = [0,1]
PLOT_FRAMES = True
BACKUP = False
PEAK_CENTER = None  # Set to a float value (2theta) to track a specific peak, or None for automatic fitting
TRACK_PEAK = True
TRACK_WINDOW = 1.2
FALLBACK_TO_AUTO = True
GRADIENT_X = "scan_number"


def sin2psi_scans_fit(
    scans,
    peak_center=PEAK_CENTER,
    track_peak=TRACK_PEAK,
    track_window=TRACK_WINDOW,
    fallback_to_auto=FALLBACK_TO_AUTO,
) -> None:
    """
    Fit sin²ψ data for a list of scans.
    args:
        scans: list of scan numbers to process
        peak_center: center of the peak to track (as a 2theta value), if None, automatic fitting will be used
        track_peak: whether to track the peak
        track_window: window around the peak to consider
        fallback_to_auto: whether to fallback to automatic fitting
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    scans = [scans] if isinstance(scans, int) else scans
    for scan in scans:
        try:
            files = proc.discover_scan_files(DATA_DIR, scan)
            if not files:
                raise FileNotFoundError(f"No matching files found for scan {scan} in {DATA_DIR}")

            result = proc.process_scan(
                data_dir=DATA_DIR,
                scan_number=scan,
                files=files,
                exclude_frames=EXCLUDE_FRAMES,
                plot_frames=PLOT_FRAMES,
                force=True,
                backup=BACKUP,
                peak_center=peak_center,
                track_peak=track_peak,
                track_window=track_window,
                fallback_to_auto=fallback_to_auto,
            )
            print(f"Wrote: {result['csv_path']}")
            print(f"Wrote: {result['scan_dir']}\\sin2psi_plot.png")
            print(f"Wrote: {result['scan_dir']}\\sin2psi_fit_params.json")

        except Exception as exc:
            logging.error("! Error processing scan %s: %s", scan, exc)


def plot_gradient_summary(x=GRADIENT_X, scans=SCANS, show=False):
    """Plot sin2psi slope against scan number or a collected metadata column."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = proc.plot_sin2psi_gradients(DATA_DIR, x=x, scans=scans, save=True, show=show)
    print(f"Wrote: {result['summary_path']}")
    print(f"Wrote: {result['plot_path']}")
    return result


if __name__ == "__main__":
    # sin2psi_scans_fit(SCANS, peak_center=PEAK_CENTER, track_peak=TRACK_PEAK, track_window=TRACK_WINDOW, fallback_to_auto=FALLBACK_TO_AUTO)
    plot_gradient_summary(x="scan_number")
    print("--- FINISHED ---")
