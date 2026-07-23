from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from quixrd.xrd_processing import sin2psi_processor as proc
else:
    from . import sin2psi_processor as proc

logger = logging.getLogger("xrd_processing.cli")


def parse_scan_spec(spec: str) -> List[int]:
    scans = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            scans.update(range(int(start), int(end) + 1))
        else:
            scans.add(int(chunk))
    return sorted(scans)


def parse_exclude_spec(spec: Optional[str]) -> List[int]:
    if spec is None or not spec.strip():
        return []
    cleaned = spec.replace(",", " ")
    return sorted({int(x) for x in cleaned.split() if x.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit .txt scans and export sin2psi results")
    parser.add_argument("--data-dir", required=True, help="Folder containing I_vs_2th_*.txt files")
    parser.add_argument("--scan", required=True, help="Single scan, comma list, or range")
    parser.add_argument("--exclude", default=None, help="Frame indices excluded from sin2psi fit")
    parser.add_argument("--force", action="store_true", help="Overwrite outputs for each scan")
    parser.add_argument("--backup", action="store_true", help="Backup scan output directory before overwrite")
    parser.add_argument("--plot-frames", dest="plot_frames", action="store_true", default=True, help="Save per-frame PNGs")
    parser.add_argument("--no-plot-frames", dest="plot_frames", action="store_false", help="Disable per-frame PNGs")
    parser.add_argument("--peak-center", type=float, default=None, help="Initial 2theta peak center guess")
    parser.add_argument("--track-peak", dest="track_peak", action="store_true", default=True, help="Track the fitted peak between frames")
    parser.add_argument("--no-track-peak", dest="track_peak", action="store_false", help="Fit each frame without previous-frame peak tracking")
    parser.add_argument("--track-window", type=float, default=0.4, help="Half-width in degrees around a seeded peak center")
    parser.add_argument("--no-fallback-to-auto", dest="fallback_to_auto", action="store_false", default=True, help="Disable automatic retry when seeded fitting fails")
    parser.add_argument("--verbosity", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.verbosity.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    scans = parse_scan_spec(args.scan)
    exclude_frames = parse_exclude_spec(args.exclude)

    logger.info("Data dir: %s", data_dir)
    logger.info("Scans: %s", scans)
    logger.info("Excluded frames: %s", exclude_frames if exclude_frames else "none")
    logger.info("Plot frames: %s", args.plot_frames)
    logger.info("Backup: %s", args.backup)
    logger.info("Peak center: %s", args.peak_center if args.peak_center is not None else "auto")
    logger.info("Track peak: %s", args.track_peak)
    logger.info("Track window: %s", args.track_window)
    logger.info("Fallback to auto: %s", args.fallback_to_auto)

    for scan_number in scans:
        files = proc.discover_scan_files(data_dir, scan_number)
        logger.info("Scan %s: %s candidate file(s)", scan_number, len(files))
        if not files:
            logger.warning("No files matched scan rules for scan %s", scan_number)
            continue
        result = proc.process_scan(
            data_dir=data_dir,
            scan_number=scan_number,
            files=files,
            exclude_frames=exclude_frames,
            plot_frames=args.plot_frames,
            force=True,
            backup=args.backup,
            peak_center=args.peak_center,
            track_peak=args.track_peak,
            track_window=args.track_window,
            fallback_to_auto=args.fallback_to_auto,
        )
        logger.info("Wrote %s", result["csv_path"])
        logger.info("Wrote %s", os.path.join(result["scan_dir"], "sin2psi_plot.png"))
        logger.info("Wrote %s", os.path.join(result["scan_dir"], "sin2psi_fit_params.json"))


if __name__ == "__main__":
    main()
