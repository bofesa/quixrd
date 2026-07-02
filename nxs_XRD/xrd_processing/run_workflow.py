from __future__ import annotations

import logging
import sys
import numpy as np
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nxs_XRD.xrd_processing import sin2psi_processor as proc
else:
    from . import sin2psi_processor as proc


DATA_DIR = r"C:\Users\bosa\OneDrive - empa.ch\WFH\Synchrotron\export"
SCANS = range(440, 451)
EXCLUDE_FRAMES = []
PLOT_FRAMES = True
BACKUP = False


def sin2psi_scans_fit(scans) -> None:
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
            )
            print(f"Wrote: {result['csv_path']}")
            print(f"Wrote: {result['scan_dir']}\\sin2psi_plot.png")
            print(f"Wrote: {result['scan_dir']}\\sin2psi_fit_params.json")

        except Exception as exc:
            logging.error("! Error processing scan %s: %s", scan, exc)


if __name__ == "__main__":
    sin2psi_scans_fit(SCANS)
