from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib

if not os.environ.get("quixrd_GUI_INTERACTIVE"):
    matplotlib.use("Agg", force=True)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import twotheta_calibration as tth_cal
from .peak_overlay import PredictedPeak, build_predicted_peaks, lattice_predicted_peaks, parse_two_theta_peaks
from .spinodal_peak_analysis import combine_profiles, discover_peak_profiles, discover_scan_numbers as discover_txt_scan_numbers
from .spinodal_peak_analysis import read_txt_profile


DEFAULT_SHAPE_FACTOR = 0.9
DEFAULT_RESIDUAL_SHIFT_LIMIT = 0.15
DEFAULT_REGISTRATION_WINDOW = 2.0


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_suffix(value):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "")).strip("._") or "value"


def _metadata_key(key):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(key).strip().lower())).strip("_")


def _coerce(value):
    text = str(value).strip()
    try:
        return float(text)
    except Exception:
        return text


def parse_manual_targets(text):
    return parse_two_theta_peaks(text)


def build_wh_targets(
    target_source="manual",
    manual_two_theta="",
    lattice_type="cubic",
    a=None,
    b=None,
    c=None,
    wavelength=None,
    energy=None,
    max_index=8,
    phase_name="",
    min_two_theta=None,
    max_two_theta=None,
    thermal_alpha=None,
    reference_temperature=None,
    temperature=None,
):
    source = str(target_source or "manual").strip().lower()
    if source in {"manual", "list", "2theta", "two_theta"}:
        targets = parse_manual_targets(manual_two_theta)
    elif source == "lattice":
        adjusted = thermal_adjust_lattice(
            a=a,
            b=b,
            c=c,
            alpha=thermal_alpha,
            reference_temperature=reference_temperature,
            temperature=temperature,
        )
        targets = lattice_predicted_peaks(
            lattice_type=lattice_type,
            a=adjusted["a"],
            b=adjusted["b"],
            c=adjusted["c"],
            wavelength=wavelength,
            energy=energy,
            max_index=max_index,
            min_two_theta=min_two_theta,
            max_two_theta=max_two_theta,
            phase_name=phase_name,
        )
    else:
        targets = build_predicted_peaks(source=source, two_theta_list=manual_two_theta)
    return _unique_targets(targets)


def thermal_adjust_lattice(a=None, b=None, c=None, alpha=None, reference_temperature=None, temperature=None):
    result = {"a": a, "b": b, "c": c, "scale": 1.0, "applied": False}
    if alpha in (None, "") or reference_temperature in (None, "") or temperature in (None, ""):
        return result
    alpha = float(alpha)
    reference_temperature = float(reference_temperature)
    temperature = float(temperature)
    scale = 1.0 + alpha * (temperature - reference_temperature)
    result["scale"] = float(scale)
    result["applied"] = True

    def adjust(value):
        if value in (None, ""):
            return value
        return float(value) * scale

    result["a"] = adjust(a)
    result["b"] = adjust(b)
    result["c"] = adjust(c)
    return result


def _unique_targets(targets, decimals=5):
    unique = []
    seen = set()
    for target in sorted(targets or [], key=lambda item: item.two_theta):
        key = round(float(target.two_theta), int(decimals))
        if key in seen:
            continue
        seen.add(key)
        unique.append(target)
    return unique


def _parse_scan_from_csv_name(path):
    match = re.search(r"(?:scan_|I_vs_2th_)(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def discover_csv_scan_numbers(data_dir):
    root = Path(data_dir)
    scans = set()
    for path in root.rglob("*.csv"):
        scan = _parse_scan_from_csv_name(path)
        if scan is not None:
            scans.add(scan)
    return sorted(scans)


def discover_scan_numbers(data_dir, profile_source="txt", scan_type=None, frame_index=None):
    source = str(profile_source or "txt").lower()
    if source == "csv":
        return discover_csv_scan_numbers(data_dir)
    return discover_txt_scan_numbers(data_dir, scan_type=scan_type, frame_index=frame_index)


def _csv_profile_candidates(data_dir, scan):
    root = Path(data_dir)
    patterns = [
        f"scan_{int(scan)}_*.csv",
        f"I_vs_2th_{int(scan)}*.csv",
        f"*{int(scan)}*.csv",
    ]
    seen = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            seen[str(path)] = path
    return sorted(seen.values())


def _read_csv_profile(path, frame_index=None):
    profiles = tth_cal.read_csv_profiles(path)
    if frame_index is not None:
        profiles = [profile for profile in profiles if profile.get("frame") == int(frame_index)]
    if not profiles:
        raise FileNotFoundError(f"No matching CSV profile frame found in {path}")
    if len(profiles) == 1:
        profile = profiles[0]
        profile["path"] = str(path)
        return profile
    combined = tth_cal.combine_profiles(profiles)
    combined["path"] = str(path)
    return combined


def load_scan_profile(data_dir, scan, profile_source="txt", scan_type=None, frame_index=0):
    source = str(profile_source or "txt").lower()
    if source == "csv":
        candidates = _csv_profile_candidates(data_dir, scan)
        if not candidates:
            raise FileNotFoundError(f"No CSV profile found for scan {scan}")
        last_error = None
        selected = None
        profile = None
        for candidate in candidates:
            try:
                profile = _read_csv_profile(candidate, frame_index=frame_index)
                selected = candidate
                break
            except Exception as exc:
                last_error = exc
        if profile is None or selected is None:
            raise FileNotFoundError(f"No CSV profile with 2theta/intensity columns found for scan {scan}: {last_error}")
        profile["profile_info"] = {
            "scan_number": int(scan),
            "scan_type": "csv",
            "frame_index": frame_index,
            "path": str(selected),
            "source_files": [str(selected)],
        }
        return profile

    scan_type_value = str(scan_type or "").lower()
    if scan_type_value == "delta":
        matches = discover_peak_profiles(data_dir, scans=[scan], scan_type=scan_type, frame_index=None)
        if not matches:
            raise FileNotFoundError(f"No delta TXT profiles found for scan {scan}")
        profiles = [read_txt_profile(match["path"]) for match in matches]
        combined = combine_profiles(profiles)
        combined["profile_info"] = {
            "scan_number": int(scan),
            "scan_type": "delta",
            "frame_index": None,
            "path": combined["path"],
            "source_files": [str(match["path"]) for match in matches],
        }
        return combined

    matches = discover_peak_profiles(data_dir, scans=[scan], scan_type=scan_type, frame_index=frame_index)
    if not matches:
        raise FileNotFoundError(f"No matching TXT profile found for scan {scan}")
    profile_info = matches[0]
    profile = read_txt_profile(profile_info["path"])
    profile["profile_info"] = profile_info
    return profile


def _metadata_energy(metadata):
    for key, value in (metadata or {}).items():
        if _metadata_key(key) == "energy" and value not in (None, ""):
            return value
    return None


def _resolve_wavelength(wavelength=None, energy=None, metadata=None):
    if wavelength not in (None, ""):
        return float(wavelength)
    energy = energy if energy not in (None, "") else _metadata_energy(metadata)
    value = tth_cal.energy_to_wavelength(energy)
    if value is None:
        raise ValueError("Williamson-Hall analysis requires wavelength, energy, or energy metadata")
    return float(value)


def _calibration_metadata_path(metadata):
    for key, value in (metadata or {}).items():
        if _metadata_key(key) == _metadata_key(tth_cal.CORRECTION_METADATA_KEY) and str(value).strip():
            return str(value).strip()
    return None


def _apply_twotheta_calibration(profile, calibration_json=None):
    metadata = dict(profile.get("metadata") or {})
    already_corrected = tth_cal.has_twotheta_correction(metadata)
    applied = False
    calibration_path = _calibration_metadata_path(metadata)
    if calibration_json and not already_corrected:
        calibration = tth_cal.load_calibration(calibration_json)
        two_theta, intensity, applied = tth_cal.apply_calibration_to_profile(
            profile["two_theta"],
            profile["intensity"],
            calibration,
            metadata=metadata,
        )
        profile = dict(profile)
        profile["two_theta"] = two_theta
        profile["intensity"] = intensity
        metadata[tth_cal.CORRECTION_METADATA_KEY] = str(calibration_json)
        profile["metadata"] = metadata
        calibration_path = str(calibration_json)
    return profile, bool(already_corrected or applied), applied, calibration_path


def _fit_targets(
    two_theta,
    intensity,
    targets: Sequence[PredictedPeak],
    fit_window=0.35,
    calibrated=False,
    registration_window=DEFAULT_REGISTRATION_WINDOW,
    residual_shift_limit=DEFAULT_RESIDUAL_SHIFT_LIMIT,
    progress_callback=None,
):
    targets = list(targets or [])
    search_window = float(residual_shift_limit if calibrated else registration_window)
    search_window = max(search_window, 0.02)
    assignments, registration = tth_cal.assign_calibration_peaks(
        two_theta,
        intensity,
        targets,
        search_window=search_window,
        model_degree=0 if calibrated else 1,
    )
    assignments_by_peak = {int(match["peak_index"]): match for match in assignments}
    rows = []
    total = len(targets)
    for index, target in enumerate(targets):
        if progress_callback and (index == 0 or index + 1 == total or (index + 1) % 10 == 0):
            progress_callback(f"Williamson-Hall: fitting peak {index + 1} of {total}")
        assignment = assignments_by_peak.get(index)
        row = {
            "target_index": int(index),
            "target_label": target.label or f"{target.two_theta:g}",
            "hkl": "".join(str(v) for v in (target.hkl or ())),
            "multiplicity": target.intensity,
            "expected_two_theta": float(target.two_theta),
            "assignment_search_window": float(search_window),
            "calibrated_input": bool(calibrated),
            "usable": False,
        }
        try:
            if assignment is None:
                raise RuntimeError("No unique observed peak assigned")
            refine_window = max(min(float(fit_window), 0.6), 0.06)
            fit = tth_cal.fit_expected_peak(
                two_theta,
                intensity,
                target.two_theta,
                window=refine_window,
                search_center=assignment["observed_two_theta"],
            )
            center_shift = abs(float(fit["center"]) - float(target.two_theta))
            search_shift = abs(float(fit["center"]) - float(assignment["observed_two_theta"]))
            row.update(fit)
            row.update(
                {
                    "assigned_observed_two_theta": float(assignment["observed_two_theta"]),
                    "assignment_distance": float(assignment["assignment_distance"]),
                    "offset": float(fit["center"] - target.two_theta),
                    "center_shift": float(center_shift),
                    "search_shift": float(search_shift),
                }
            )
            usable = (
                np.isfinite(row.get("center", np.nan))
                and np.isfinite(row.get("fwhm", np.nan))
                and float(row.get("amplitude", 0.0)) > 0
                and float(row.get("fwhm", 0.0)) > 0
                and search_shift <= refine_window * 0.9
            )
            row["usable"] = bool(usable)
            if not usable:
                row["rejection_reason"] = "quality_gate_failed"
        except Exception as exc:
            row["error"] = str(exc)
            row["rejection_reason"] = str(exc)
        rows.append(row)
    return rows, registration


def _instrument_fwhm_deg(caglioti, two_theta):
    if not caglioti:
        return np.zeros_like(np.asarray(two_theta, dtype=float))
    theta = np.radians(np.asarray(two_theta, dtype=float) / 2.0)
    y = (
        float(caglioti.get("U", 0.0)) * np.tan(theta) ** 2
        + float(caglioti.get("V", 0.0)) * np.tan(theta)
        + float(caglioti.get("W", 0.0))
    )
    return np.sqrt(np.maximum(y, 0.0))


def calculate_wh_fit(peaks, wavelength, shape_factor=DEFAULT_SHAPE_FACTOR, caglioti=None):
    rows = [dict(peak) for peak in peaks]
    usable = []
    for row in rows:
        if not row.get("usable"):
            continue
        center = float(row.get("center", np.nan))
        fwhm = float(row.get("fwhm", np.nan))
        if not (np.isfinite(center) and np.isfinite(fwhm) and fwhm > 0):
            row["usable"] = False
            row["rejection_reason"] = "nonfinite_center_or_fwhm"
            continue
        theta = math.radians(center / 2.0)
        beta_obs = math.radians(fwhm)
        beta_inst = math.radians(float(_instrument_fwhm_deg(caglioti, [center])[0]))
        beta_sample_sq = beta_obs**2 - beta_inst**2
        if beta_sample_sq <= 0:
            row["usable"] = False
            row["rejection_reason"] = "instrument_broadening_exceeds_observed"
            continue
        beta_sample = math.sqrt(beta_sample_sq)
        row["theta_deg"] = math.degrees(theta)
        row["beta_observed_rad"] = beta_obs
        row["beta_instrument_rad"] = beta_inst
        row["beta_sample_rad"] = beta_sample
        row["wh_x"] = 4.0 * math.sin(theta)
        row["wh_y"] = beta_sample * math.cos(theta)
        fwhm_err = row.get("fwhm_err", np.nan)
        if np.isfinite(fwhm_err):
            row["wh_y_err"] = math.radians(abs(float(fwhm_err))) * math.cos(theta)
        usable.append(row)

    summary = {
        "usable_peak_count": int(len(usable)),
        "target_peak_count": int(len(rows)),
        "shape_factor": float(shape_factor),
        "wavelength": float(wavelength),
        "wh_success": False,
    }
    if len(usable) < 2:
        summary["warning"] = "Need at least two usable peaks for Williamson-Hall fit"
        return rows, summary

    x = np.asarray([row["wh_x"] for row in usable], dtype=float)
    y = np.asarray([row["wh_y"] for row in usable], dtype=float)
    yerr = np.asarray([row.get("wh_y_err", np.nan) for row in usable], dtype=float)
    if len(usable) == 2:
        coeffs = np.polyfit(x, y, 1)
        cov = None
        summary["warning"] = "Only two usable peaks; WH line fit has no covariance/uncertainty estimate"
    elif np.isfinite(yerr).all() and np.any(yerr > 0):
        weights = 1.0 / np.maximum(yerr, 1e-12)
        coeffs, cov = np.polyfit(x, y, 1, w=weights, cov=True)
    else:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    slope_err = float(math.sqrt(max(cov[0, 0], 0.0))) if cov is not None and np.ndim(cov) == 2 else np.nan
    intercept_err = float(math.sqrt(max(cov[1, 1], 0.0))) if cov is not None and np.ndim(cov) == 2 else np.nan
    residuals = y - (slope * x + intercept)
    summary.update(
        {
            "wh_success": True,
            "microstrain": slope,
            "microstrain_err": slope_err,
            "intercept": intercept,
            "intercept_err": intercept_err,
            "rmse": float(math.sqrt(np.mean(residuals**2))),
        }
    )
    if intercept > 0:
        crystallite_size = float(shape_factor) * float(wavelength) / intercept
        summary["crystallite_size_A"] = crystallite_size
        summary["crystallite_size_nm"] = crystallite_size / 10.0
        summary["crystallite_size_err_A"] = abs(float(shape_factor) * float(wavelength) / (intercept**2) * intercept_err)
        summary["crystallite_size_err_nm"] = summary["crystallite_size_err_A"] / 10.0
    else:
        summary["crystallite_size_A"] = np.nan
        summary["crystallite_size_nm"] = np.nan
        summary["warning"] = "WH intercept is not positive; crystallite size is not physical"
    return rows, summary


def _metadata_from_profile(profile):
    metadata = dict(profile.get("metadata") or {})
    normalised = {}
    for key, value in metadata.items():
        normalised[str(key)] = value
        normalised[_metadata_key(key)] = value
    return normalised


def _plot_scan_wh(path, peak_df, summary_row, title, show=False, save=True):
    usable = peak_df[peak_df["usable"] == True] if "usable" in peak_df.columns else peak_df
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    if not usable.empty:
        yerr = usable["wh_y_err"].to_numpy(dtype=float) if "wh_y_err" in usable and usable["wh_y_err"].notna().any() else None
        ax.errorbar(
            usable["wh_x"],
            usable["wh_y"],
            yerr=yerr,
            fmt=".",
            markersize=4,
            color="tab:blue",
            ecolor="tab:blue",
            elinewidth=0.8,
            alpha=0.8,
            label="used peaks",
        )
        if bool(summary_row.get("wh_success")):
            xs = np.linspace(float(usable["wh_x"].min()), float(usable["wh_x"].max()), 200)
            ax.plot(xs, float(summary_row["microstrain"]) * xs + float(summary_row["intercept"]), "-", color="tab:orange", label="WH fit")
    excluded = peak_df[(peak_df.get("usable", False) == False)] if "usable" in peak_df.columns else pd.DataFrame()
    if not excluded.empty and {"wh_x", "wh_y"}.issubset(excluded.columns):
        ax.plot(excluded["wh_x"], excluded["wh_y"], ".", color="tab:cyan", alpha=0.75, label="excluded")
    ax.set_xlabel("4 sin(theta)")
    ax.set_ylabel("beta cos(theta) (radians)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(path, dpi=150)
        fig.savefig(Path(path).with_suffix(".svg"))
    if not show:
        plt.close(fig)
    return fig


def _trend_x_values(df, x):
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found in Williamson-Hall summary")
    values = df[x]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric, False
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().any():
        return parsed, True
    return values.astype(str), False


def _add_secondary_y_axis(ax, x_values, df, secondary_y):
    if secondary_y in (None, "", "none", "off", "false"):
        return None
    if secondary_y not in df.columns:
        raise ValueError(f"Column '{secondary_y}' not found in Williamson-Hall summary")
    values = pd.to_numeric(df[secondary_y], errors="coerce")
    mask = values.notna()
    if not mask.any():
        return None
    sec = ax.twinx()
    sec.plot(np.asarray(x_values)[mask.to_numpy()], values[mask], ".--", color="tab:red", alpha=0.75, linewidth=0.8)
    sec.set_ylabel(str(secondary_y).replace("_", " "), color="tab:red")
    sec.tick_params(axis="y", labelcolor="tab:red")
    return sec


def plot_wh_trends_from_csv(csv_path, x="scan_number", save=True, show=False, secondary_y=None):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Williamson-Hall summary is empty: {csv_path}")
    plot_df = df[df.get("wh_success", True) == True] if "wh_success" in df.columns else df
    if plot_df.empty:
        raise ValueError("No successful Williamson-Hall fits to plot")
    x_values, is_datetime = _trend_x_values(plot_df, x)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True)
    axes[0].errorbar(
        x_values,
        pd.to_numeric(plot_df["microstrain"], errors="coerce"),
        yerr=pd.to_numeric(plot_df.get("microstrain_err"), errors="coerce") if "microstrain_err" in plot_df else None,
        fmt=".",
        markersize=4,
        color="tab:blue",
        ecolor="tab:blue",
        elinewidth=0.8,
        alpha=0.85,
        label="microstrain",
    )
    if "crystallite_size_nm" in plot_df:
        axes[1].errorbar(
            x_values,
            pd.to_numeric(plot_df["crystallite_size_nm"], errors="coerce"),
            yerr=pd.to_numeric(plot_df.get("crystallite_size_err_nm"), errors="coerce") if "crystallite_size_err_nm" in plot_df else None,
            fmt=".",
            markersize=4,
            color="tab:green",
            ecolor="tab:green",
            elinewidth=0.8,
            alpha=0.85,
            label="crystallite size",
        )
    _add_secondary_y_axis(axes[0], x_values, plot_df, secondary_y)
    axes[0].set_ylabel("Microstrain")
    axes[1].set_ylabel("Crystallite size (nm)")
    axes[1].set_xlabel(str(x).replace("_", " "))
    if is_datetime:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        axes[1].xaxis.set_major_locator(locator)
        axes[1].xaxis.set_major_formatter(formatter)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Williamson-Hall trends vs {str(x).replace('_', ' ')}")
    fig.tight_layout()
    plot_path = csv_path.parent / f"{csv_path.stem}_trends_vs_{_safe_suffix(x)}_{_timestamp()}.png"
    if save:
        fig.savefig(plot_path, dpi=150)
        fig.savefig(plot_path.with_suffix(".svg"))
    else:
        plot_path = None
    if not show:
        plt.close(fig)
    return {"data": plot_df, "plot_path": str(plot_path) if plot_path else None, "csv_path": str(csv_path)}


def _write_profile_diagnostic(path, profile, fitted_peaks, title):
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.plot(profile["two_theta"], profile["intensity"], "-", linewidth=0.7, color="tab:blue", label="profile")
    ymin, ymax = ax.get_ylim()
    y_text = ymin + 0.96 * (ymax - ymin)
    labelled_at = -np.inf
    handles = {}
    for peak in fitted_peaks:
        label = peak.get("target_label") or peak.get("hkl") or f"{peak.get('expected_two_theta', np.nan):.2f}"
        expected = peak.get("expected_two_theta")
        assigned = peak.get("assigned_observed_two_theta")
        center = peak.get("center")
        usable = bool(peak.get("usable"))

        def finite(value):
            try:
                return np.isfinite(float(value))
            except Exception:
                return False

        if finite(expected):
            line = ax.axvline(
                float(expected),
                color="0.55",
                linewidth=0.55,
                alpha=0.45,
                linestyle="--",
                label="attempted target",
            )
            handles.setdefault("attempted target", line)
            if float(expected) - labelled_at >= 1.0:
                ax.annotate(
                    label,
                    xy=(float(expected), y_text),
                    xytext=(-3, 0),
                    textcoords="offset points",
                    rotation=90,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color="0.35",
                    alpha=0.8,
                )
                labelled_at = float(expected)
        if finite(assigned):
            line = ax.axvline(
                float(assigned),
                color="tab:purple",
                linewidth=0.65,
                alpha=0.45,
                linestyle=":",
                label="assigned observed seed",
            )
            handles.setdefault("assigned observed seed", line)
        if finite(center):
            fit_label = "successful fitted center" if usable else "failed/rejected fitted center"
            line = ax.axvline(
                float(center),
                color="tab:orange" if usable else "tab:red",
                linewidth=0.9 if usable else 0.7,
                alpha=0.9 if usable else 0.5,
                label=fit_label,
            )
            handles.setdefault(fit_label, line)
        elif not usable and finite(expected):
            marker = ax.plot(
                [float(expected)],
                [ymin + 0.05 * (ymax - ymin)],
                "x",
                color="tab:red",
                alpha=0.65,
                markersize=4,
                label="no fitted center",
            )[0]
            handles.setdefault("no fitted center", marker)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("2theta (deg)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if handles:
        ax.legend(handles.values(), handles.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(Path(path).with_suffix(".svg"))
    plt.close(fig)


def run_williamson_hall_series(
    data_dir,
    scans=None,
    profile_source="txt",
    scan_type="delta",
    frame_index=None,
    target_source="manual",
    manual_two_theta="",
    lattice_type="cubic",
    a=None,
    b=None,
    c=None,
    wavelength=None,
    energy=None,
    max_index=8,
    phase_name="",
    thermal_alpha=None,
    reference_temperature=None,
    fit_window=0.35,
    shape_factor=DEFAULT_SHAPE_FACTOR,
    twotheta_calibration_json=None,
    residual_shift_limit=DEFAULT_RESIDUAL_SHIFT_LIMIT,
    registration_window=DEFAULT_REGISTRATION_WINDOW,
    x="scan_number",
    secondary_y=None,
    save=True,
    show=False,
    progress_callback=None,
    cancel_check=None,
):
    data_dir = Path(data_dir)
    scan_type_value = str(scan_type or "").lower()
    effective_frame_index = None if scan_type_value == "delta" else frame_index
    if scans is None:
        scans = discover_scan_numbers(data_dir, profile_source=profile_source, scan_type=scan_type, frame_index=effective_frame_index)
    scans = [int(scan) for scan in scans]
    if not scans:
        raise ValueError("No scans found for Williamson-Hall analysis")

    output_root = data_dir / "williamson_hall"
    stamp = _timestamp()
    output_dir = output_root / f"williamson_hall_{stamp}"
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    peak_rows = []
    wh_plot_paths = []
    calibration = tth_cal.load_calibration(twotheta_calibration_json) if twotheta_calibration_json else None
    caglioti = calibration.get("caglioti") if calibration else None

    total = len(scans)
    if progress_callback:
        progress_callback(f"Williamson-Hall: fitting {total} scan(s)")
    for scan_idx, scan in enumerate(scans, start=1):
        if cancel_check and cancel_check():
            raise RuntimeError("Williamson-Hall cancelled by user")
        if progress_callback:
            progress_callback(f"Williamson-Hall: scan {scan} ({scan_idx}/{total})")
        try:
            profile = load_scan_profile(
                data_dir,
                scan,
                profile_source=profile_source,
                scan_type=scan_type,
                frame_index=effective_frame_index,
            )
            profile, calibrated_input, applied_calibration, correction_path = _apply_twotheta_calibration(
                profile,
                twotheta_calibration_json,
            )
            metadata = _metadata_from_profile(profile)
            wl = _resolve_wavelength(wavelength=wavelength, energy=energy, metadata=metadata)
            scan_temperature = metadata.get("temperature")
            tth = np.asarray(profile["two_theta"], dtype=float)
            min_tth = float(np.nanmin(tth))
            max_tth = float(np.nanmax(tth))
            adjusted_lattice = thermal_adjust_lattice(
                a=a,
                b=b,
                c=c,
                alpha=thermal_alpha,
                reference_temperature=reference_temperature,
                temperature=scan_temperature,
            )
            targets = build_wh_targets(
                target_source=target_source,
                manual_two_theta=manual_two_theta,
                lattice_type=lattice_type,
                a=a,
                b=b,
                c=c,
                wavelength=wl,
                max_index=max_index,
                phase_name=phase_name,
                min_two_theta=min_tth,
                max_two_theta=max_tth,
                thermal_alpha=thermal_alpha,
                reference_temperature=reference_temperature,
                temperature=scan_temperature,
            )
            fitted, registration = _fit_targets(
                profile["two_theta"],
                profile["intensity"],
                targets,
                fit_window=fit_window,
                calibrated=calibrated_input,
                registration_window=registration_window,
                residual_shift_limit=residual_shift_limit,
                progress_callback=progress_callback if total == 1 else None,
            )
            try:
                wh_rows, summary = calculate_wh_fit(
                    fitted,
                    wavelength=wl,
                    shape_factor=shape_factor,
                    caglioti=caglioti,
                )
            except Exception as exc:
                wh_rows = [dict(peak) for peak in fitted]
                summary = {
                    "usable_peak_count": int(sum(1 for peak in wh_rows if peak.get("usable"))),
                    "target_peak_count": int(len(wh_rows)),
                    "shape_factor": float(shape_factor),
                    "wavelength": float(wl),
                    "wh_success": False,
                    "warning": f"Peak diagnostics written, but WH line fit failed: {exc}",
                }
            profile_info = profile.get("profile_info", {})
            summary_row = {
                "scan_number": int(scan),
                "scan_type": profile_info.get("scan_type", scan_type),
                "frame_index": profile_info.get("frame_index", effective_frame_index),
                "profile_source": profile_source,
                "profile_path": profile_info.get("path", profile.get("path")),
                "target_source": target_source,
                "target_count": len(targets),
                "fit_window": float(fit_window),
                "calibrated_input": bool(calibrated_input),
                "twotheta_calibration_applied": bool(applied_calibration),
                "twotheta_correction_path": correction_path,
                "registration_window": float(registration_window),
                "residual_shift_limit": float(residual_shift_limit),
                "registration_initial_shift": registration.get("initial_shift"),
                "registration_assigned_peak_count": len(registration.get("matches", [])),
                "thermal_alpha": thermal_alpha,
                "thermal_reference_temperature": reference_temperature,
                "thermal_temperature": scan_temperature,
                "thermal_lattice_scale": adjusted_lattice["scale"],
                "thermal_lattice_applied": adjusted_lattice["applied"],
                "adjusted_a": adjusted_lattice["a"],
                "adjusted_b": adjusted_lattice["b"],
                "adjusted_c": adjusted_lattice["c"],
                **summary,
            }
            for key in ("temperature", "energy", "start_time", "frame_time", "chi"):
                if key in metadata:
                    summary_row[key] = metadata[key]
            summary_rows.append(summary_row)
            for peak in wh_rows:
                peak_rows.append({"scan_number": int(scan), **peak})
            if save:
                peak_df_one = pd.DataFrame([{**{"scan_number": int(scan)}, **peak} for peak in wh_rows])
                _write_profile_diagnostic(
                    output_dir / f"scan_{int(scan)}_profile_diagnostic.png",
                    profile,
                    wh_rows,
                    f"Scan {scan} fitted WH peaks",
                )
                if {"wh_x", "wh_y"}.issubset(peak_df_one.columns):
                    wh_path = output_dir / f"scan_{int(scan)}_williamson_hall.png"
                    _plot_scan_wh(wh_path, peak_df_one, summary_row, f"Scan {scan} Williamson-Hall", show=False, save=True)
                    wh_plot_paths.append(str(wh_path))
            if progress_callback:
                success_text = "fit" if summary.get("wh_success") else "not fitted"
                progress_callback(f"Williamson-Hall: scan {scan} {success_text} ({summary.get('usable_peak_count', 0)} usable peaks)")
        except Exception as exc:
            summary_rows.append({"scan_number": int(scan), "wh_success": False, "error": str(exc)})
            if progress_callback:
                progress_callback(f"Williamson-Hall: scan {scan} failed: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    peaks_df = pd.DataFrame(peak_rows)
    summary_path = output_dir / f"williamson_hall_summary_{stamp}.csv"
    peaks_path = output_dir / f"williamson_hall_peaks_{stamp}.csv"
    params_path = output_dir / f"williamson_hall_params_{stamp}.json"
    trend_result = {"plot_path": None}
    if save:
        summary_df.to_csv(summary_path, index=False)
        peaks_df.to_csv(peaks_path, index=False)
        params = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_dir": str(data_dir),
            "scans": scans,
            "profile_source": profile_source,
            "scan_type": scan_type,
            "frame_index": effective_frame_index,
            "target_source": target_source,
            "manual_two_theta": manual_two_theta,
            "lattice_type": lattice_type,
            "a": a,
            "b": b,
            "c": c,
            "wavelength": wavelength,
            "energy": energy,
            "max_index": max_index,
            "phase_name": phase_name,
            "thermal_alpha": thermal_alpha,
            "reference_temperature": reference_temperature,
            "fit_window": fit_window,
            "shape_factor": shape_factor,
            "twotheta_calibration_json": twotheta_calibration_json,
            "residual_shift_limit": residual_shift_limit,
            "registration_window": registration_window,
            "summary_path": str(summary_path),
            "peaks_path": str(peaks_path),
            "wh_plot_paths": wh_plot_paths,
        }
        params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        if "wh_success" in summary_df.columns and summary_df["wh_success"].astype(bool).any():
            trend_result = plot_wh_trends_from_csv(summary_path, x=x, save=True, show=show, secondary_y=secondary_y)
            if show:
                plt.show()
    if progress_callback:
        ok = int(summary_df.get("wh_success", pd.Series(dtype=bool)).astype(bool).sum()) if not summary_df.empty else 0
        progress_callback(f"Williamson-Hall: finished ({ok} succeeded, {len(summary_rows) - ok} failed)")
    return {
        "data": summary_df,
        "peaks": peaks_df,
        "summary_path": str(summary_path),
        "peaks_path": str(peaks_path),
        "params_path": str(params_path),
        "plot_path": trend_result.get("plot_path"),
        "wh_plot_paths": wh_plot_paths,
        "output_dir": str(output_dir),
        "success_count": int(summary_df.get("wh_success", pd.Series(dtype=bool)).astype(bool).sum()) if not summary_df.empty else 0,
        "failed_count": int(len(summary_rows) - (summary_df.get("wh_success", pd.Series(dtype=bool)).astype(bool).sum() if not summary_df.empty else 0)),
    }
