from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nxs_XRD.xrd_processing import sin2psi_processor as proc
else:
    from . import sin2psi_processor as proc


DATA_DIR = r"C:\Users\bosa\local_code\22_Al-AlOH as-dep\\"
# SCANS = range(2156, 2561+1)
SCANS = range(209, 348+1)       # Al, 2theta 31.75
# SCANS = range(562, 628+1)       # Cu at 2theta 41.6
# SCANS = range(766, 848+1)   # Cu-Al
EXCLUDE_FRAMES = []
EXCLUDE_CHI_RANGES = []
EXCLUDE_SIN2PSI_RANGES = []
AUTO_EXCLUDE = False
AUTO_EXCLUDE_SIGMA = 3.0
AUTO_EXCLUDE_MAX_ITER = 1
CORRECTION_JSON = None
PLOT_FRAMES = True
BACKUP = False
PEAK_CENTER = 41.6  # Set to a float value (2theta) to track a specific peak, or None for automatic fitting
TRACK_PEAK = True
TRACK_WINDOW = 1.0
FALLBACK_TO_AUTO = True
GRADIENT_X = "temp"
PARAMS_JSON = None
_DEFAULT = object()

### CORRECTION CURVES
# DATA_DIR = r"C:\Users\bosa\local_code\00_White paint on Si\\"
# SCANS = [197, 202]
# PEAK_CENTER = 59.1
# TRACK_WINDOW = 0.3



def _resolve_scans(scans):
    if scans is not _DEFAULT:
        return scans
    if "SCANS" not in globals():
        raise NameError("SCANS is not defined. Define SCANS at the top of run_workflow.py or pass scans=...")
    return SCANS

def sin2psi_scans_fit(
    scans,
    peak_center=_DEFAULT,
    track_peak=_DEFAULT,
    track_window=_DEFAULT,
    fallback_to_auto=_DEFAULT,
    exclude_frames=_DEFAULT,
    exclude_chi_ranges=_DEFAULT,
    exclude_sin2psi_ranges=_DEFAULT,
    auto_exclude=_DEFAULT,
    auto_exclude_sigma=_DEFAULT,
    auto_exclude_max_iter=_DEFAULT,
    correction_json=_DEFAULT,
    params_json=PARAMS_JSON,
):
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
    peak_center = PEAK_CENTER if peak_center is _DEFAULT else peak_center
    track_peak = TRACK_PEAK if track_peak is _DEFAULT else track_peak
    track_window = TRACK_WINDOW if track_window is _DEFAULT else track_window
    fallback_to_auto = FALLBACK_TO_AUTO if fallback_to_auto is _DEFAULT else fallback_to_auto
    exclude_frames = EXCLUDE_FRAMES if exclude_frames is _DEFAULT else exclude_frames
    exclude_chi_ranges = EXCLUDE_CHI_RANGES if exclude_chi_ranges is _DEFAULT else exclude_chi_ranges
    exclude_sin2psi_ranges = EXCLUDE_SIN2PSI_RANGES if exclude_sin2psi_ranges is _DEFAULT else exclude_sin2psi_ranges
    auto_exclude = AUTO_EXCLUDE if auto_exclude is _DEFAULT else auto_exclude
    auto_exclude_sigma = AUTO_EXCLUDE_SIGMA if auto_exclude_sigma is _DEFAULT else auto_exclude_sigma
    auto_exclude_max_iter = AUTO_EXCLUDE_MAX_ITER if auto_exclude_max_iter is _DEFAULT else auto_exclude_max_iter
    correction_json = CORRECTION_JSON if correction_json is _DEFAULT else correction_json
    imported = proc.load_processing_params(params_json) if params_json else {}
    imported_options = imported.get("processing_options", imported) if isinstance(imported, dict) else {}
    scans = [scans] if isinstance(scans, int) else scans
    scan_results = []
    params = proc.build_processing_params(
        created_at=proc._output_timestamp(),
        source_params_json=str(params_json) if params_json else None,
        data_dir=DATA_DIR,
        scans=[int(scan) for scan in scans],
        exclude_frames=exclude_frames,
        exclude_chi_ranges=exclude_chi_ranges,
        exclude_sin2psi_ranges=exclude_sin2psi_ranges,
        auto_exclude=auto_exclude,
        auto_exclude_sigma=auto_exclude_sigma,
        auto_exclude_max_iter=auto_exclude_max_iter,
        correction_json=correction_json,
        processing_options={
            **{key: imported_options.get(key) for key in imported_options},
            "plot_frames": PLOT_FRAMES,
            "force": True,
            "backup": BACKUP,
            "peak_center": peak_center,
            "track_peak": track_peak,
            "track_window": track_window,
            "fallback_to_auto": fallback_to_auto,
            "exclude_frames": exclude_frames,
            "exclude_chi_ranges": exclude_chi_ranges,
            "exclude_sin2psi_ranges": exclude_sin2psi_ranges,
            "auto_exclude": auto_exclude,
            "auto_exclude_sigma": auto_exclude_sigma,
            "auto_exclude_max_iter": auto_exclude_max_iter,
            "correction_json": correction_json,
        },
        scan_results=scan_results,
    )
    for scan in scans:
        try:
            files = proc.discover_scan_files(DATA_DIR, scan)
            if not files:
                raise FileNotFoundError(f"No matching files found for scan {scan} in {DATA_DIR}")

            result = proc.process_scan(
                data_dir=DATA_DIR,
                scan_number=scan,
                files=files,
                exclude_frames=exclude_frames,
                exclude_chi_ranges=exclude_chi_ranges,
                exclude_sin2psi_ranges=exclude_sin2psi_ranges,
                auto_exclude=auto_exclude,
                auto_exclude_sigma=auto_exclude_sigma,
                auto_exclude_max_iter=auto_exclude_max_iter,
                correction_json=correction_json,
                plot_frames=PLOT_FRAMES,
                force=True,
                backup=BACKUP,
                peak_center=peak_center,
                track_peak=track_peak,
                track_window=track_window,
                fallback_to_auto=fallback_to_auto,
            )
            scan_results.append(
                {
                    "scan_number": int(scan),
                    "status": "ok",
                    "csv_path": result["csv_path"],
                    "scan_dir": result["scan_dir"],
                    "frame_count": result["frame_count"],
                }
            )
            print(f"Wrote: {result['csv_path']}")
            print(f"Wrote: {result['scan_dir']}\\scan_{scan}_sin2psi_plot.png")
            print(f"Wrote: {result['scan_dir']}\\sin2psi_fit_params.json")

        except Exception as exc:
            logging.error("! Error processing scan %s: %s", scan, exc)
            scan_results.append({"scan_number": int(scan), "status": "error", "error": str(exc)})
    params["completed_at"] = proc._output_timestamp()
    params["scan_results"] = scan_results
    log_path = proc.save_processing_params(params, DATA_DIR)
    print(f"Wrote: {log_path}")
    return log_path


def plot_gradient_summary(x=GRADIENT_X, scans=_DEFAULT, show=False):
    """Plot sin2psi slope against scan number or a collected metadata column."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    scans = _resolve_scans(scans)
    result = proc.plot_sin2psi_gradients(DATA_DIR, x=x, scans=scans, save=True, show=show)
    print(f"Wrote: {result['summary_path']}")
    print(f"Wrote: {result['plot_path']}")
    return result


def refit_sin2psi_trends(
    scans=_DEFAULT,
    exclude_frames=_DEFAULT,
    exclude_chi_ranges=_DEFAULT,
    exclude_sin2psi_ranges=_DEFAULT,
    auto_exclude=_DEFAULT,
    auto_exclude_sigma=_DEFAULT,
    auto_exclude_max_iter=_DEFAULT,
    correction_json=_DEFAULT,
    params_json=PARAMS_JSON,
):
    """Recompute sin2psi regression outputs from existing scan fit CSVs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    scans = _resolve_scans(scans)
    exclude_frames = EXCLUDE_FRAMES if exclude_frames is _DEFAULT else exclude_frames
    exclude_chi_ranges = EXCLUDE_CHI_RANGES if exclude_chi_ranges is _DEFAULT else exclude_chi_ranges
    exclude_sin2psi_ranges = EXCLUDE_SIN2PSI_RANGES if exclude_sin2psi_ranges is _DEFAULT else exclude_sin2psi_ranges
    auto_exclude = AUTO_EXCLUDE if auto_exclude is _DEFAULT else auto_exclude
    auto_exclude_sigma = AUTO_EXCLUDE_SIGMA if auto_exclude_sigma is _DEFAULT else auto_exclude_sigma
    auto_exclude_max_iter = AUTO_EXCLUDE_MAX_ITER if auto_exclude_max_iter is _DEFAULT else auto_exclude_max_iter
    correction_json = CORRECTION_JSON if correction_json is _DEFAULT else correction_json
    imported = proc.load_processing_params(params_json) if params_json else {}
    imported_options = imported.get("processing_options", imported) if isinstance(imported, dict) else {}
    scans = [scans] if isinstance(scans, int) else scans
    scan_results = []
    params = proc.build_processing_params(
        created_at=proc._output_timestamp(),
        operation="refit_sin2psi_trends",
        source_params_json=str(params_json) if params_json else None,
        data_dir=DATA_DIR,
        scans=[int(scan) for scan in scans],
        exclude_frames=exclude_frames,
        exclude_chi_ranges=exclude_chi_ranges,
        exclude_sin2psi_ranges=exclude_sin2psi_ranges,
        auto_exclude=auto_exclude,
        auto_exclude_sigma=auto_exclude_sigma,
        auto_exclude_max_iter=auto_exclude_max_iter,
        correction_json=correction_json,
        processing_options={
            **{key: imported_options.get(key) for key in imported_options},
            "exclude_frames": exclude_frames,
            "exclude_chi_ranges": exclude_chi_ranges,
            "exclude_sin2psi_ranges": exclude_sin2psi_ranges,
            "auto_exclude": auto_exclude,
            "auto_exclude_sigma": auto_exclude_sigma,
            "auto_exclude_max_iter": auto_exclude_max_iter,
            "correction_json": correction_json,
        },
        scan_results=scan_results,
    )
    for scan in scans:
        try:
            result = proc.refit_sin2psi_from_csv(
                DATA_DIR,
                scan,
                excluded_frames=exclude_frames,
                exclude_chi_ranges=exclude_chi_ranges,
                exclude_sin2psi_ranges=exclude_sin2psi_ranges,
                auto_exclude=auto_exclude,
                auto_exclude_sigma=auto_exclude_sigma,
                auto_exclude_max_iter=auto_exclude_max_iter,
                correction_json=correction_json,
            )
            scan_results.append(
                {
                    "scan_number": int(scan),
                    "status": "ok",
                    "csv_path": result["csv_path"],
                    "scan_dir": result["scan_dir"],
                }
            )
            print(f"Updated: {result['csv_path']}")
            print(f"Updated: {result['scan_dir']}\\scan_{scan}_sin2psi_plot.png")
            print(f"Updated: {result['scan_dir']}\\sin2psi_fit_params.json")
        except Exception as exc:
            logging.error("! Error refitting scan %s: %s", scan, exc)
            scan_results.append({"scan_number": int(scan), "status": "error", "error": str(exc)})
    params["completed_at"] = proc._output_timestamp()
    params["scan_results"] = scan_results
    log_path = proc.save_processing_params(params, DATA_DIR)
    print(f"Wrote: {log_path}")
    return log_path


def generate_correction_curve(
    folder_path,
    reference_scan,
    degree=2,
    output_path=None,
    reference_two_theta=None,
    method="polynomial",
    gp_length_scale=0.25,
    gp_signal_variance=None,
):
    """Generate a sin2psi correction JSON from a stress-free reference scan."""
    result = proc.generate_sin2psi_correction(
        folder_path,
        reference_scan,
        degree=degree,
        method=method,
        reference_two_theta=reference_two_theta,
        gp_length_scale=gp_length_scale,
        gp_signal_variance=gp_signal_variance,
        output_path=output_path,
        excluded_frames=None,
        exclude_chi_ranges=None,
        exclude_sin2psi_ranges=None,
    )
    print(f"Wrote: {result['path']}")
    print(f"Wrote: {result['plot_path']}")
    return result


def plot_fwhm_summary(x="scan_number", frame_index=None, chi=None, scans=_DEFAULT, show=False):
    """Plot FWHM for selected frame index/indices or chi value(s) against scan metadata."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    scans = _resolve_scans(scans)
    result = proc.plot_fwhm_trends(
        DATA_DIR,
        x=x,
        scans=scans,
        frame_index=frame_index,
        chi=chi,
        save=True,
        show=show,
    )
    print(f"Wrote: {result['summary_path']}")
    print(f"Wrote: {result['plot_path']}")
    return result


def plot_peak_position_summary(x="scan_number", frame_index=None, chi=None, scans=_DEFAULT, show=False):
    """Plot peak position for selected frame index/indices or chi value(s) against scan metadata."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    scans = _resolve_scans(scans)
    result = proc.plot_peak_position_trends(
        DATA_DIR,
        x=x,
        scans=scans,
        frame_index=frame_index,
        chi=chi,
        save=True,
        show=show,
    )
    print(f"Wrote: {result['summary_path']}")
    print(f"Wrote: {result['plot_path']}")
    return result


if __name__ == "__main__":
    # sin2psi_scans_fit(SCANS, peak_center=PEAK_CENTER, track_peak=TRACK_PEAK, track_window=TRACK_WINDOW, fallback_to_auto=FALLBACK_TO_AUTO)
    plot_gradient_summary(x="temperature", scans=SCANS, show=False)
    # plot_fwhm_summary(x="temperature", chi=[0.0, 5.0], scans=SCANS)
    # plot_fwhm_summary(x="temperature", frame_index=[0, 1, 2], scans=SCANS)
    # plot_peak_position_summary(x="temperature", chi=[0.0, 5.0], scans=SCANS)
    # plot_peak_position_summary(x="temperature", frame_index=[0, 1, 2], scans=SCANS)
    
    # # Calibration scans. [scan_number, reference_two_theta, peak_hkl]
    # calib_params = [[202, 58.93, (1, 3, 1)], [203, 46.53, (2, 2, 0)], [204, 34.11, (1, 1, 1)],      # Correct z
    #                 [197, 58.93, (1, 3, 1)], [196, 46.53, (2, 2, 0)], [197, 34.11, (1, 1, 1)]]      # Incorrect z
    # for scan, ref_2theta, hkl in calib_params[:3]:  # Process only the first three calibration scans
    #     generate_correction_curve(r"C:\Users\bosa\local_code\00_White paint on Si\\", reference_scan=scan, degree=2, output_path=None, reference_two_theta=ref_2theta, method="gaussian_process", )
    print("--- FINISHED ---")
