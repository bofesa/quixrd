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
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

MAX_FIT_POINTS = 900
SINGLE_MAX_NFEV = 2500
TWO_MAX_NFEV = 4000
TREND_MARKER_SIZE = 3.0
TREND_ERROR_ALPHA = 0.45


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _coerce(value):
    text = str(value).strip()
    try:
        return float(text)
    except Exception:
        return text


def _metadata_key(key):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(key).strip().lower())).strip("_")


def read_txt_profile(path):
    metadata = {}
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                match = re.match(r"#\s*([^:=]+)\s*[:=]\s*(.+)$", line)
                if match:
                    value = _coerce(match.group(2))
                    metadata[match.group(1).strip()] = value
                    metadata[_metadata_key(match.group(1))] = value
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except Exception:
                continue
    if not rows:
        raise RuntimeError(f"No numeric 2theta/intensity data found in {path}")
    arr = np.asarray(rows, dtype=float)
    return {"path": str(path), "two_theta": arr[:, 0], "intensity": arr[:, 1], "metadata": metadata}


def _profile_step(two_theta):
    diffs = np.diff(np.asarray(two_theta, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return np.nan
    return float(np.median(diffs))


def combine_profiles(profiles):
    if not profiles:
        raise ValueError("No profiles supplied for homogenisation")
    ranges = []
    steps = []
    for profile in profiles:
        two_theta = np.asarray(profile["two_theta"], dtype=float)
        finite = two_theta[np.isfinite(two_theta)]
        if finite.size:
            ranges.append((float(np.min(finite)), float(np.max(finite))))
            step = _profile_step(finite)
            if np.isfinite(step) and step > 0:
                steps.append(step)
    if not ranges:
        raise ValueError("Profiles have no finite 2theta values")
    step = float(np.median(steps)) if steps else 0.01
    grid = np.arange(min(low for low, _ in ranges), max(high for _, high in ranges) + step * 0.5, step)
    summed = np.zeros_like(grid, dtype=float)
    counts = np.zeros_like(grid, dtype=float)
    for profile in profiles:
        two_theta = np.asarray(profile["two_theta"], dtype=float)
        intensity = np.asarray(profile["intensity"], dtype=float)
        order = np.argsort(two_theta)
        two_theta = two_theta[order]
        intensity = intensity[order]
        mask = np.isfinite(two_theta) & np.isfinite(intensity)
        two_theta = two_theta[mask]
        intensity = intensity[mask]
        if len(two_theta) < 2:
            continue
        in_range = (grid >= two_theta.min()) & (grid <= two_theta.max())
        summed[in_range] += np.interp(grid[in_range], two_theta, intensity)
        counts[in_range] += 1
    valid = counts > 0
    metadata = dict(profiles[0].get("metadata", {}))
    metadata.update(
        {
            "homogenised_profile": True,
            "source_frame_count": len(profiles),
            "overlap_policy": "blend",
        }
    )
    return {
        "path": "; ".join(str(profile.get("path", "")) for profile in profiles),
        "two_theta": grid[valid],
        "intensity": summed[valid] / counts[valid],
        "metadata": metadata,
    }


def _parse_scan_file(path):
    match = re.match(r"I_vs_2th_(\d+)(?:_([A-Za-z]+))?_(\d+)\.txt$", Path(path).name)
    if not match:
        return None
    return {
        "scan_number": int(match.group(1)),
        "scan_type": match.group(2) or "",
        "frame_index": int(match.group(3)),
        "path": Path(path),
    }


def discover_peak_profiles(data_dir, scans=None, scan_type=None, frame_index=None):
    root = Path(data_dir)
    scan_set = {int(scan) for scan in scans} if scans is not None else None
    profiles = []
    for path in sorted(root.rglob("I_vs_2th_*.txt")):
        parsed = _parse_scan_file(path)
        if not parsed:
            continue
        if scan_set is not None and parsed["scan_number"] not in scan_set:
            continue
        if scan_type and parsed["scan_type"] and parsed["scan_type"].lower() != str(scan_type).lower():
            continue
        if frame_index is not None and parsed["frame_index"] != int(frame_index):
            continue
        profiles.append(parsed)
    return profiles


def discover_scan_numbers(data_dir, scan_type=None, frame_index=None):
    return sorted(
        {
            item["scan_number"]
            for item in discover_peak_profiles(data_dir, scan_type=scan_type, frame_index=frame_index)
        }
    )


def load_scan_profile(data_dir, scan, scan_type=None, frame_index=0):
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


def _pseudo_voigt(x, amplitude, center, fwhm, eta):
    width = np.maximum(float(fwhm), 1e-9)
    gaussian = np.exp(-np.log(2.0) * ((x - center) / (width / 2.0)) ** 2)
    lorentzian = 1.0 / (1.0 + ((x - center) / (width / 2.0)) ** 2)
    return amplitude * (eta * gaussian + (1.0 - eta) * lorentzian)


def _single_peak_model(x, b0, b1, amp1, center1, fwhm1, eta1):
    return b0 + b1 * x + _pseudo_voigt(x, amp1, center1, fwhm1, eta1)


def _two_peak_model(x, b0, b1, amp1, center1, fwhm1, amp2, center2, fwhm2, eta):
    return (
        b0
        + b1 * x
        + _pseudo_voigt(x, amp1, center1, fwhm1, eta)
        + _pseudo_voigt(x, amp2, center2, fwhm2, eta)
    )


def _pseudo_voigt_area(amplitude, fwhm, eta):
    hwhm = abs(float(fwhm)) / 2.0
    gaussian_area = hwhm * math.sqrt(math.pi / math.log(2.0))
    lorentzian_area = math.pi * hwhm
    return float(abs(amplitude) * (float(eta) * gaussian_area + (1.0 - float(eta)) * lorentzian_area))


def _two_peak_components(x, fit):
    params = fit["params"]
    b0, b1, amp1, center1, fwhm1, amp2, center2, fwhm2, eta = params
    background = b0 + b1 * x
    component_1 = background + _pseudo_voigt(x, amp1, center1, fwhm1, eta)
    component_2 = background + _pseudo_voigt(x, amp2, center2, fwhm2, eta)
    return component_1, component_2


def _fit_metrics(y, y_fit, param_count):
    residuals = np.asarray(y, dtype=float) - np.asarray(y_fit, dtype=float)
    n = max(len(residuals), 1)
    rss = float(np.sum(residuals**2))
    rss = max(rss, np.finfo(float).tiny)
    return {
        "rss": rss,
        "aic": float(n * math.log(rss / n) + 2 * param_count),
        "bic": float(n * math.log(rss / n) + param_count * math.log(n)),
        "rmse": float(math.sqrt(rss / n)),
    }


def _limit_fit_points(x, y, max_points=MAX_FIT_POINTS):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= int(max_points):
        return x, y
    indices = np.linspace(0, len(x) - 1, int(max_points), dtype=int)
    return x[indices], y[indices]


def _unique_center_pair(pair, lower, upper, min_separation):
    centers = sorted(float(np.clip(center, lower, upper)) for center in pair)
    if centers[1] - centers[0] < min_separation:
        return None
    return (round(centers[0], 6), round(centers[1], 6))


def _initial_peak_guess_pairs(x, y, peak_center, fit_window, split_guess=None, single_fit=None):
    baseline = float(np.nanpercentile(y, 10))
    signal = y - baseline
    max_signal = max(float(np.nanmax(signal)), 1.0)
    lower = float(peak_center) - float(fit_window)
    upper = float(peak_center) + float(fit_window)
    step = _profile_step(x)
    min_separation = max(float(step) * 2.0 if np.isfinite(step) else 0.0, 0.015)
    main_center = float(x[int(np.nanargmax(signal))])
    seed_centers = [main_center]

    indices, props = find_peaks(signal, prominence=max(max_signal * 0.01, np.nanstd(signal) * 0.25, 1e-9))
    if len(indices):
        prominences = props.get("prominences", signal[indices])
        for idx in np.argsort(prominences)[-6:]:
            seed_centers.append(float(x[int(indices[int(idx)])]))

    if single_fit is not None and "y_fit" in single_fit:
        residual = np.asarray(y, dtype=float) - np.asarray(single_fit["y_fit"], dtype=float)
        positive = residual - np.nanpercentile(residual, 40)
        positive[positive < 0.0] = 0.0
        if np.nanmax(positive) > 0:
            residual_indices, residual_props = find_peaks(
                positive,
                prominence=max(float(np.nanmax(positive)) * 0.08, np.nanstd(residual) * 0.4, 1e-9),
            )
            if len(residual_indices):
                prominences = residual_props.get("prominences", positive[residual_indices])
                for idx in np.argsort(prominences)[-4:]:
                    seed_centers.append(float(x[int(residual_indices[int(idx)])]))

    pairs = []
    seen = set()

    def add_pair(pair):
        key = _unique_center_pair(pair, lower, upper, min_separation)
        if key is None or key in seen:
            return
        seen.add(key)
        pairs.append(key)

    if split_guess not in (None, ""):
        split_values = [abs(float(split_guess))]
    else:
        width = float(x.max() - x.min())
        split_values = [0.08, 0.14, 0.20, 0.30, 0.45, width * 0.12, width * 0.20, width * 0.32]
    for split in split_values:
        if split > 0:
            add_pair((main_center - split / 2.0, main_center + split / 2.0))
            add_pair((float(peak_center) - split / 2.0, float(peak_center) + split / 2.0))

    seed_centers = [center for center in seed_centers if lower <= center <= upper and np.isfinite(center)]
    for center in seed_centers:
        if abs(center - main_center) >= min_separation:
            add_pair((main_center, center))
    ranked = sorted(set(seed_centers), key=lambda center: float(np.interp(center, x, signal)), reverse=True)[:8]
    for left_idx, left in enumerate(ranked):
        for right in ranked[left_idx + 1 :]:
            add_pair((left, right))

    if not pairs:
        add_pair((float(peak_center) - 0.05, float(peak_center) + 0.05))
    return pairs


def _fit_single(x, y, peak_center, fit_window):
    baseline = float(np.nanpercentile(y, 10))
    signal = y - baseline
    max_idx = int(np.nanargmax(signal))
    amp = max(float(signal[max_idx]), 1.0)
    center = float(x[max_idx])
    width_guess = max((x.max() - x.min()) * 0.08, 0.02)
    bounds = (
        [-np.inf, -np.inf, 0.0, float(peak_center) - fit_window, 1e-5, 0.0],
        [np.inf, np.inf, np.inf, float(peak_center) + fit_window, fit_window * 2.0, 1.0],
    )
    popt, pcov = curve_fit(
        _single_peak_model,
        x,
        y,
        p0=[baseline, 0.0, amp, center, width_guess, 0.5],
        bounds=bounds,
        method="trf",
        max_nfev=SINGLE_MAX_NFEV,
    )
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.ndim(pcov) == 2 else np.full(len(popt), np.nan)
    y_fit = _single_peak_model(x, *popt)
    metrics = _fit_metrics(y, y_fit, len(popt))
    return {
        "model": "single",
        "params": [float(v) for v in popt],
        "errors": [float(v) if np.isfinite(v) else np.nan for v in perr],
        "x_fit": x,
        "y_fit": y_fit,
        **metrics,
        "center_1": float(popt[3]),
        "center_1_err": float(perr[3]) if np.isfinite(perr[3]) else np.nan,
        "fwhm_1": float(abs(popt[4])),
        "fwhm_1_err": float(perr[4]) if np.isfinite(perr[4]) else np.nan,
        "amplitude_1": float(popt[2]),
    }


def _fit_two_from_centers(x, y, peak_center, fit_window, centers):
    baseline = float(np.nanpercentile(y, 10))
    signal = y - baseline
    width_guess = max((x.max() - x.min()) * 0.06, 0.02)
    amps = [max(float(np.interp(center, x, signal)), 1.0) for center in centers]
    bounds = (
        [-np.inf, -np.inf, 0.0, float(peak_center) - fit_window, 1e-5, 0.0, float(peak_center) - fit_window, 1e-5, 0.0],
        [np.inf, np.inf, np.inf, float(peak_center) + fit_window, fit_window * 2.0, np.inf, float(peak_center) + fit_window, fit_window * 2.0, 1.0],
    )
    popt, pcov = curve_fit(
        _two_peak_model,
        x,
        y,
        p0=[baseline, 0.0, amps[0], centers[0], width_guess, amps[1], centers[1], width_guess, 0.5],
        bounds=bounds,
        method="trf",
        max_nfev=TWO_MAX_NFEV,
    )
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.ndim(pcov) == 2 else np.full(len(popt), np.nan)
    if popt[3] > popt[6]:
        order = [0, 1, 5, 6, 7, 2, 3, 4, 8]
        popt = popt[order]
        perr = perr[order]
    y_fit = _two_peak_model(x, *popt)
    metrics = _fit_metrics(y, y_fit, len(popt))
    separation = abs(float(popt[6] - popt[3]))
    area_1 = _pseudo_voigt_area(popt[2], popt[4], popt[8])
    area_2 = _pseudo_voigt_area(popt[5], popt[7], popt[8])
    amplitude_1 = float(popt[2])
    amplitude_2 = float(popt[5])
    major_amplitude = max(abs(amplitude_1), abs(amplitude_2))
    major_area = max(area_1, area_2)
    return {
        "model": "two",
        "params": [float(v) for v in popt],
        "errors": [float(v) if np.isfinite(v) else np.nan for v in perr],
        "x_fit": x,
        "y_fit": y_fit,
        **metrics,
        "center_1": float(popt[3]),
        "center_1_err": float(perr[3]) if np.isfinite(perr[3]) else np.nan,
        "fwhm_1": float(abs(popt[4])),
        "fwhm_1_err": float(perr[4]) if np.isfinite(perr[4]) else np.nan,
        "amplitude_1": amplitude_1,
        "center_2": float(popt[6]),
        "center_2_err": float(perr[6]) if np.isfinite(perr[6]) else np.nan,
        "fwhm_2": float(abs(popt[7])),
        "fwhm_2_err": float(perr[7]) if np.isfinite(perr[7]) else np.nan,
        "amplitude_2": amplitude_2,
        "area_1": area_1,
        "area_2": area_2,
        "height_ratio_2_over_1": float(amplitude_2 / amplitude_1) if amplitude_1 else np.nan,
        "area_ratio_2_over_1": float(area_2 / area_1) if area_1 else np.nan,
        "minor_major_height_ratio": float(min(abs(amplitude_1), abs(amplitude_2)) / major_amplitude) if major_amplitude else np.nan,
        "minor_major_area_ratio": float(min(area_1, area_2) / major_area) if major_area else np.nan,
        "peak_separation": separation,
        "initial_center_1": float(centers[0]),
        "initial_center_2": float(centers[1]),
    }


def _fit_two(x, y, peak_center, fit_window, split_guess=None, single_fit=None):
    pairs = _initial_peak_guess_pairs(x, y, peak_center, fit_window, split_guess=split_guess, single_fit=single_fit)
    best = None
    last_error = None
    for centers in pairs:
        try:
            fit = _fit_two_from_centers(x, y, peak_center, fit_window, centers)
        except Exception as exc:
            last_error = exc
            continue
        if best is None or fit["bic"] < best["bic"]:
            best = fit
    if best is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Two-peak fit failed for all initial guesses")
    best["initial_guess_count"] = len(pairs)
    return best


def fit_peak_models(two_theta, intensity, peak_center, fit_window=0.5, fit_mode="compare", split_guess=None):
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    peak_center = float(peak_center)
    fit_window = float(fit_window)
    mask = (two_theta >= peak_center - fit_window) & (two_theta <= peak_center + fit_window)
    x_raw = two_theta[mask]
    y_raw = intensity[mask]
    if len(x_raw) < 12:
        raise RuntimeError("Too few points in selected peak window")
    x, y = _limit_fit_points(x_raw, y_raw)
    result = {
        "fit_mode": fit_mode,
        "peak_center": peak_center,
        "fit_window": fit_window,
        "data_point_count": int(len(x_raw)),
        "fit_point_count": int(len(x)),
    }
    single = None
    two = None
    if fit_mode in {"single", "compare"}:
        single = _fit_single(x, y, peak_center, fit_window)
        result["single"] = single
    if fit_mode in {"two", "compare"}:
        two = _fit_two(x, y, peak_center, fit_window, split_guess=split_guess, single_fit=single)
        result["two"] = two
    if single and two:
        delta_bic = float(single["bic"] - two["bic"])
        result["delta_bic"] = delta_bic
        result["delta_aic"] = float(single["aic"] - two["aic"])
        min_sep = max(0.03, min(two["fwhm_1"], two["fwhm_2"]) * 0.35)
        result["two_peak_preferred"] = bool(delta_bic > 10.0 and two["peak_separation"] >= min_sep)
        result["selected_model"] = "two" if result["two_peak_preferred"] else "single"
    else:
        result["two_peak_preferred"] = fit_mode == "two"
        result["selected_model"] = fit_mode
    result["x"] = x_raw
    result["y"] = y_raw
    return result


def _row_from_fit(scan_number, profile_path, fit_result, metadata=None):
    selected = fit_result["selected_model"]
    model = fit_result.get(selected)
    row = {
        "scan_number": int(scan_number),
        "filename": str(profile_path),
        "selected_model": selected,
    }
    if "delta_bic" in fit_result:
        row["delta_bic"] = fit_result["delta_bic"]
    if "delta_aic" in fit_result:
        row["delta_aic"] = fit_result["delta_aic"]
    for prefix in ("single", "two"):
        fit = fit_result.get(prefix)
        if not fit:
            continue
        row[f"{prefix}_bic"] = fit["bic"]
        row[f"{prefix}_aic"] = fit["aic"]
        for key in (
            "center_1",
            "center_1_err",
            "fwhm_1",
            "fwhm_1_err",
            "amplitude_1",
            "center_2",
            "center_2_err",
            "fwhm_2",
            "fwhm_2_err",
            "amplitude_2",
            "area_1",
            "area_2",
            "height_ratio_2_over_1",
            "area_ratio_2_over_1",
            "minor_major_height_ratio",
            "minor_major_area_ratio",
            "peak_separation",
        ):
            if key in fit:
                row[f"{prefix}_{key}"] = fit[key]
    if model:
        for key, value in model.items():
            if key in {"params", "errors", "x_fit", "y_fit", "model"}:
                continue
            if np.isscalar(value):
                row[key] = value
    for key in ("temperature", "energy", "start_time", "frame_time", "chi"):
        if metadata and key in metadata:
            row[key] = metadata[key]
    return row


def _plot_fit(path, fit_result, title, show=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fit_result["x"], fit_result["y"], ".", color="tab:blue", label="data")
    if "single" in fit_result:
        ax.plot(fit_result["single"]["x_fit"], fit_result["single"]["y_fit"], "-", color="tab:orange", label="single peak")
    if "two" in fit_result:
        ax.plot(fit_result["two"]["x_fit"], fit_result["two"]["y_fit"], "-", color="tab:green", label="two peaks")
        ax.axvline(fit_result["two"]["center_1"], color="tab:green", linewidth=0.7, alpha=0.8)
        ax.axvline(fit_result["two"]["center_2"], color="tab:green", linewidth=0.7, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("2theta (deg)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if not show:
        plt.close(fig)
    return fig


def _trend_x_values(df, x):
    if x in df.columns:
        values = df[x]
    else:
        values = df["scan_number"]
    if pd.api.types.is_numeric_dtype(values):
        return values, False
    if pd.api.types.is_datetime64_any_dtype(values):
        return values, True
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().sum() >= max(2, int(len(values) * 0.6)):
        return parsed, True
    return values, False


def _errorbar(ax, x_values, y_values, yerr=None, label=None, color=None, alpha=1.0):
    return ax.errorbar(
        x_values,
        y_values,
        yerr=yerr,
        fmt=".",
        markersize=TREND_MARKER_SIZE,
        elinewidth=0.8,
        capsize=1.5,
        alpha=alpha,
        ecolor=color,
        color=color,
        label=label,
    )


def _apply_errorbar_alpha(container, alpha=TREND_ERROR_ALPHA):
    for artist in container.lines[1]:
        artist.set_alpha(alpha)
    for artist in container.lines[2]:
        artist.set_alpha(alpha)


def _subset_values(values, mask):
    if isinstance(values, pd.Series):
        return values.loc[mask]
    return np.asarray(values)[np.asarray(mask, dtype=bool)]


def _plot_selected_series(ax, x_values, df, y_col, err_col, label, color, selected_model=None):
    if y_col not in df.columns or not df[y_col].notna().any():
        return
    if selected_model is None or "selected_model" not in df.columns:
        container = _errorbar(ax, x_values, df[y_col], yerr=df.get(err_col), label=label, color=color)
        _apply_errorbar_alpha(container)
        return
    selected = df["selected_model"].astype(str).str.lower() == selected_model
    masks = [(selected, 1.0, label), (~selected, 0.25, None)]
    for mask, alpha, mask_label in masks:
        mask = mask & df[y_col].notna()
        if not mask.any():
            continue
        yerr = df.loc[mask, err_col] if err_col in df.columns else None
        container = _errorbar(
            ax,
            _subset_values(x_values, mask),
            df.loc[mask, y_col],
            yerr=yerr,
            label=mask_label,
            color=color,
            alpha=alpha,
        )
        _apply_errorbar_alpha(container, TREND_ERROR_ALPHA * alpha)


def _plot_trends(path, df, x="scan_number", show=False):
    has_comparison = "delta_bic" in df.columns and df["delta_bic"].notna().any()
    ratio_column = None
    for candidate in ("minor_major_height_ratio", "height_ratio_2_over_1", "two_minor_major_height_ratio"):
        if candidate in df.columns and df[candidate].notna().any():
            ratio_column = candidate
            break
    has_comparison_panel = has_comparison or ratio_column is not None
    panel_count = 2 + int(has_comparison_panel)
    fig, axes = plt.subplots(panel_count, 1, figsize=(8.5, 3.2 * panel_count + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    x_values, is_datetime = _trend_x_values(df, x)

    if "single_center_1" in df.columns and df["single_center_1"].notna().any():
        _plot_selected_series(axes[0], x_values, df, "single_center_1", "single_center_1_err", "1-peak center", "tab:orange", "single")
        _plot_selected_series(axes[1], x_values, df, "single_fwhm_1", "single_fwhm_1_err", "1-peak FWHM", "tab:orange", "single")
    else:
        container = _errorbar(axes[0], x_values, df["center_1"], yerr=df.get("center_1_err"), label="peak 1", color="tab:orange")
        _apply_errorbar_alpha(container)
        container = _errorbar(axes[1], x_values, df["fwhm_1"], yerr=df.get("fwhm_1_err"), label="peak 1", color="tab:orange")
        _apply_errorbar_alpha(container)

    if "two_center_1" in df.columns and df["two_center_1"].notna().any():
        _plot_selected_series(axes[0], x_values, df, "two_center_1", "two_center_1_err", "2-peak center 1", "tab:green", "two")
        _plot_selected_series(axes[0], x_values, df, "two_center_2", "two_center_2_err", "2-peak center 2", "tab:green", "two")
        _plot_selected_series(axes[1], x_values, df, "two_fwhm_1", "two_fwhm_1_err", "2-peak FWHM 1", "tab:green", "two")
        _plot_selected_series(axes[1], x_values, df, "two_fwhm_2", "two_fwhm_2_err", "2-peak FWHM 2", "tab:green", "two")
    elif "center_2" in df.columns and df["center_2"].notna().any():
        container = _errorbar(axes[0], x_values, df["center_1"], yerr=df.get("center_1_err"), label="2-peak center 1", color="tab:green")
        _apply_errorbar_alpha(container)
        container = _errorbar(axes[0], x_values, df["center_2"], yerr=df.get("center_2_err"), label="2-peak center 2", color="tab:green")
        _apply_errorbar_alpha(container)
        container = _errorbar(axes[1], x_values, df["fwhm_1"], yerr=df.get("fwhm_1_err"), label="2-peak FWHM 1", color="tab:green")
        _apply_errorbar_alpha(container)
        container = _errorbar(axes[1], x_values, df["fwhm_2"], yerr=df.get("fwhm_2_err"), label="2-peak FWHM 2", color="tab:green")
        _apply_errorbar_alpha(container)
    if has_comparison_panel:
        comparison_ax = axes[2]
        handles = []
        labels = []
        if has_comparison:
            line = comparison_ax.plot(x_values, df["delta_bic"], ".", markersize=TREND_MARKER_SIZE, color="tab:purple", label="delta BIC")[0]
            comparison_ax.set_ylabel("BIC(single)-BIC(two)")
            handles.append(line)
            labels.append("delta BIC")
        if ratio_column is not None:
            ratio_ax = comparison_ax.twinx() if has_comparison else comparison_ax
            ratio_line = ratio_ax.plot(x_values, df[ratio_column], ".", markersize=TREND_MARKER_SIZE, color="tab:brown", label="minor/major height ratio")[0]
            ratio_ax.set_ylabel("Relative intensity")
            ratio_ax.set_ylim(bottom=0.0)
            handles.append(ratio_line)
            labels.append("minor/major height ratio")
        if handles:
            comparison_ax.legend(handles, labels, fontsize=8)
    axes[0].set_ylabel("2theta center")
    axes[1].set_ylabel("FWHM")
    axes[-1].set_xlabel(x)
    if is_datetime:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(formatter)
    for ax in axes:
        ax.grid(alpha=0.25)
        if ax is not axes[-1] or not has_comparison_panel:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if not show:
        plt.close(fig)
    return fig


def _diagnostic_candidates(fit_results):
    if not fit_results:
        return []
    candidates = [("First successful fit", fit_results[0])]
    comparable = [(scan, fit, row) for scan, fit, row in fit_results if np.isfinite(row.get("delta_bic", np.nan))]
    if comparable:
        values = np.asarray([row["delta_bic"] for _scan, _fit, row in comparable], dtype=float)
        median_idx = int(np.argmin(np.abs(values - np.nanmedian(values))))
        candidates.append(("Median delta BIC fit", comparable[median_idx]))
    worst = max(fit_results, key=lambda item: float(item[2].get("rmse", -np.inf)))
    candidates.append(("Worst RMSE fit", worst))
    unique = []
    by_scan = {}
    for label, item in candidates:
        scan = item[0]
        if scan in by_scan:
            by_scan[scan][0].append(label)
            continue
        by_scan[scan] = ([label], item)
    for labels, item in by_scan.values():
        unique.append((" / ".join(labels), item))
    return unique[:3]


def _diagnostic_scan_path(path, scan_number, index):
    path = Path(path)
    return path.with_name(f"{path.stem}_scan_{int(scan_number):04d}_{index + 1:03d}{path.suffix}")


def _diagnostic_stats_text(label, row):
    parts = []
    selected = row.get("selected_model")
    if selected:
        parts.append(f"model={selected}")
    rmse = row.get("rmse", np.nan)
    if np.isfinite(rmse):
        parts.append(f"RMSE={rmse:.3g}")
    delta = row.get("delta_bic", np.nan)
    if np.isfinite(delta):
        parts.append(f"delta BIC={delta:.2f}")
    ratio = row.get("minor_major_height_ratio", np.nan)
    if np.isfinite(ratio):
        parts.append(f"minor/major height={ratio:.2f}")
    if label and not str(label).startswith("Scan diagnostic"):
        parts.append(str(label))
    return "   |   ".join(parts)


def _plot_diagnostic_page(path, selected, show=False):
    if not selected:
        return None
    fig, axes = plt.subplots(len(selected), 1, figsize=(8.5, 3.4 * len(selected)), squeeze=False)
    for ax, (label, (scan, fit, row)) in zip(axes[:, 0], selected):
        ax.plot(fit["x"], fit["y"], ".", color="tab:blue", label="data")
        if "single" in fit:
            ax.plot(fit["single"]["x_fit"], fit["single"]["y_fit"], "-", color="tab:orange", label="final 1-peak fit")
            ax.axvline(fit["single"]["center_1"], color="tab:orange", linewidth=0.7, alpha=0.8)
        if "two" in fit:
            ax.plot(fit["two"]["x_fit"], fit["two"]["y_fit"], "-", color="tab:green", label="final 2-peak fit")
            component_1, component_2 = _two_peak_components(fit["two"]["x_fit"], fit["two"])
            ax.plot(fit["two"]["x_fit"], component_1, ":", color="tab:green", linewidth=0.8, alpha=0.45, label="2-peak component 1")
            ax.plot(fit["two"]["x_fit"], component_2, ":", color="tab:green", linewidth=0.8, alpha=0.45, label="2-peak component 2")
            ax.axvline(fit["two"]["center_1"], color="tab:green", linewidth=0.7, alpha=0.8)
            ax.axvline(fit["two"]["center_2"], color="tab:green", linewidth=0.7, alpha=0.8)
        ax.text(0.5, 1.14, f"Scan {scan}", transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
        stats = _diagnostic_stats_text(label, row)
        if stats:
            ax.text(0.5, 1.06, stats, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="0.25")
        ax.set_xlabel("2theta (deg)")
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if not show:
        plt.close(fig)
    return fig


def _plot_diagnostics(path, fit_results, show=False, all_fits=False):
    selected = (
        [(f"Scan diagnostic {idx + 1} of {len(fit_results)}", item) for idx, item in enumerate(fit_results)]
        if all_fits
        else _diagnostic_candidates(fit_results)
    )
    if not selected:
        return []
    if not all_fits:
        _plot_diagnostic_page(path, selected, show=show)
        return [str(path)]

    written = []
    for index, item in enumerate(selected):
        _label, (scan, _fit, _row) = item
        scan_path = _diagnostic_scan_path(path, scan, index)
        _plot_diagnostic_page(scan_path, [item], show=show)
        written.append(str(scan_path))
    return written


def plot_peak_series_from_csv(csv_path, x="scan_number", save=True, show=False):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Peak Analysis CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Peak Analysis CSV is empty: {csv_path}")
    usable = df[df.get("fit_success", True) == True] if "fit_success" in df.columns else df
    if usable.empty:
        raise ValueError(f"Peak Analysis CSV has no successful fits: {csv_path}")
    stamp = _timestamp()
    plot_path = csv_path.parent / f"{csv_path.stem}_replot_vs_{x}_{stamp}.png"
    if save:
        _plot_trends(plot_path, usable, x=x, show=show)
    elif show:
        _plot_trends(plot_path, usable, x=x, show=True)
    return {"data": usable, "plot_path": str(plot_path), "csv_path": str(csv_path), "x": x}


def _plot_scan_diagnostic(path, fit_item, index, total, show=False):
    scan, _fit, _row = fit_item
    label = f"Scan diagnostic {index} of {total}"
    scan_path = _diagnostic_scan_path(path, scan, index - 1)
    _plot_diagnostic_page(scan_path, [(label, fit_item)], show=show)
    return str(scan_path)


def run_peak_series(
    data_dir,
    scans=None,
    scan_type=None,
    frame_index=0,
    peak_center=None,
    fit_window=0.5,
    fit_mode="compare",
    split_guess=None,
    x="scan_number",
    save=True,
    show=False,
    diagnostic_all_fits=False,
    progress_callback=None,
):
    if peak_center in (None, ""):
        raise ValueError("peak_center is required for peak analysis")
    scan_type_value = str(scan_type or "").lower()
    effective_frame_index = None if scan_type_value == "delta" else frame_index
    scans = list(scans) if scans is not None else discover_scan_numbers(data_dir, scan_type=scan_type, frame_index=effective_frame_index)
    if not scans:
        raise ValueError("No scans found for peak analysis")
    output_dir = Path(data_dir) / "peak_analysis"
    stamp = _timestamp()
    diagnostic_dir = output_dir / f"diagnostics_{stamp}"
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"peak_series_{stamp}.csv"
    plot_path = output_dir / f"peak_series_trends_{stamp}.png"
    diagnostic_plot_path = diagnostic_dir / f"peak_series_diagnostics_{stamp}.png"
    params_path = output_dir / f"peak_series_params_{stamp}.json"
    rows = []
    fit_results = []
    total = len(scans)
    if progress_callback:
        progress_callback(f"Peak Analysis: fitting {total} scan(s)")
    diagnostic_plot_paths = []
    for scan_idx, scan in enumerate(scans, start=1):
        if progress_callback:
            progress_callback(f"Peak Analysis: scan {scan} ({scan_idx}/{total})")
        try:
            profile = load_scan_profile(data_dir, scan, scan_type=scan_type, frame_index=effective_frame_index)
            profile_info = profile["profile_info"]
            fit = fit_peak_models(
                profile["two_theta"],
                profile["intensity"],
                peak_center=peak_center,
                fit_window=fit_window,
                fit_mode=fit_mode,
                split_guess=split_guess,
            )
            row = _row_from_fit(scan, profile_info["path"], fit, metadata=profile.get("metadata"))
            row["fit_success"] = True
            row["scan_type"] = profile_info["scan_type"]
            row["frame_index"] = profile_info["frame_index"]
            row["homogenised_profile"] = bool(profile.get("metadata", {}).get("homogenised_profile", False))
            row["source_frame_count"] = profile.get("metadata", {}).get("source_frame_count", 1)
            rows.append(row)
            fit_item = (scan, fit, row)
            fit_results.append(fit_item)
            if progress_callback:
                model_text = row.get("selected_model", "unknown")
                delta = row.get("delta_bic", np.nan)
                delta_text = f", delta BIC={delta:.2f}" if np.isfinite(delta) else ""
                progress_callback(f"Peak Analysis: scan {scan} fitted ({model_text}{delta_text})")
            if save and diagnostic_all_fits:
                diagnostic_path = _plot_scan_diagnostic(
                    diagnostic_plot_path,
                    fit_item,
                    len(diagnostic_plot_paths) + 1,
                    total,
                    show=False,
                )
                diagnostic_plot_paths.append(diagnostic_path)
                if progress_callback:
                    progress_callback(f"Peak Analysis: saved diagnostic {diagnostic_path}")
        except Exception as exc:
            rows.append({"scan_number": int(scan), "fit_success": False, "error": str(exc)})
            if progress_callback:
                progress_callback(f"Peak Analysis: scan {scan} failed: {exc}")
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(csv_path, index=False)
        usable = df[df.get("fit_success", False) == True] if "fit_success" in df.columns else df
        if not usable.empty and "center_1" in usable.columns:
            _plot_trends(plot_path, usable, x=x, show=show)
            if not diagnostic_all_fits:
                diagnostic_plot_paths = _plot_diagnostics(
                    diagnostic_plot_path,
                    fit_results,
                    show=show,
                    all_fits=False,
                )
            if show:
                plt.show()
        else:
            diagnostic_plot_paths = []
        diagnostic_primary_path = diagnostic_plot_paths[0] if diagnostic_plot_paths else str(diagnostic_plot_path)
        params = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_dir": str(data_dir),
            "scans": scans,
            "scan_type": scan_type,
            "frame_index": effective_frame_index,
            "peak_center": peak_center,
            "fit_window": fit_window,
            "fit_mode": fit_mode,
            "split_guess": split_guess,
            "x": x,
            "diagnostic_all_fits": diagnostic_all_fits,
            "csv_path": str(csv_path),
            "plot_path": str(plot_path),
            "diagnostic_plot_path": diagnostic_primary_path,
            "diagnostic_plot_paths": diagnostic_plot_paths,
        }
        params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        if fit_results == []:
            raise RuntimeError(f"Peak Analysis failed for all {len(scans)} scan(s). Details were written to {csv_path}")
    else:
        diagnostic_plot_paths = []
        diagnostic_primary_path = str(diagnostic_plot_path)
    success_count = len(fit_results)
    failed_count = len(rows) - success_count
    if progress_callback:
        progress_callback(f"Peak Analysis: finished ({success_count} succeeded, {failed_count} failed)")
    return {
        "data": df,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "diagnostic_plot_path": diagnostic_primary_path,
        "diagnostic_plot_paths": diagnostic_plot_paths,
        "params_path": str(params_path),
        "success_count": success_count,
        "failed_count": failed_count,
    }
