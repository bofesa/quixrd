from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

if not os.environ.get("quixrd_GUI_INTERACTIVE"):
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .peak_overlay import ENERGY_TO_WAVELENGTH_KEV_A, PredictedPeak, lattice_predicted_peaks, overlay_predicted_peaks


LAB6_A = 4.25695
CORRECTION_METADATA_KEY = "TwoTheta Correction"


def _tick_spacing_for_range(x_min, x_max):
    span = max(float(x_max) - float(x_min), 1e-9)
    if span <= 12:
        return 1.0, 0.2
    if span <= 30:
        return 2.0, 0.5
    if span <= 70:
        return 5.0, 1.0
    return 10.0, 2.0


def energy_to_wavelength(energy):
    if energy in (None, ""):
        return None
    energy = float(energy)
    if energy <= 0:
        raise ValueError("Energy must be positive")
    if energy > 1000:
        energy /= 1000.0
    return ENERGY_TO_WAVELENGTH_KEV_A / energy


def wavelength_to_energy(wavelength):
    if wavelength in (None, ""):
        return None
    wavelength = float(wavelength)
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive")
    return ENERGY_TO_WAVELENGTH_KEV_A / wavelength


def _safe_suffix(value):
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "")).strip("._")
    return text or "calibration"


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _metadata_key(key):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(key).strip().lower())).strip("_")


def _coerce(value):
    text = str(value).strip()
    try:
        return float(text)
    except Exception:
        return text


def read_txt_profile(path):
    metadata = {}
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                clean = line[1:].strip()
                match = re.match(r"([^:=]+)\s*[:=]\s*(.+)$", clean)
                if match:
                    metadata[match.group(1).strip()] = _coerce(match.group(2))
                    metadata[_metadata_key(match.group(1))] = _coerce(match.group(2))
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append([float(part) for part in parts[:3]])
            except Exception:
                continue
    if not rows:
        raise RuntimeError(f"No numeric profile data found in {path}")
    arr = np.asarray(rows, dtype=float)
    return {
        "path": str(path),
        "two_theta": arr[:, 0],
        "intensity": arr[:, 1],
        "q": arr[:, 2] if arr.shape[1] > 2 else None,
        "metadata": metadata,
    }


def read_csv_profiles(path):
    df = pd.read_csv(path)
    lower = {str(column).lower(): column for column in df.columns}
    tth_col = lower.get("2theta") or lower.get("two_theta") or lower.get("twotheta")
    intensity_col = lower.get("intensity")
    if tth_col is None or intensity_col is None:
        raise ValueError(f"CSV calibration input needs 2theta and intensity columns: {path}")
    frame_col = (
        lower.get("frame_index")
        or lower.get("pointindex")
        or lower.get("point_index")
        or lower.get("idx")
        or lower.get("timestamp")
        or lower.get("delta")
    )
    if frame_col is None:
        groups = [(0, df)]
    else:
        groups = sorted(df.groupby(frame_col), key=lambda item: item[0])
    profiles = []
    for frame, frame_df in groups:
        try:
            frame_value = int(frame)
        except Exception:
            frame_value = str(frame)
        metadata = {}
        for column in frame_df.columns:
            if column in {tth_col, intensity_col}:
                continue
            value = frame_df[column].iloc[0]
            if pd.notna(value):
                metadata[str(column)] = _coerce(value)
                metadata[_metadata_key(column)] = _coerce(value)
        profiles.append(
            {
                "path": str(path),
                "frame": frame_value if pd.notna(frame) else 0,
                "two_theta": frame_df[tth_col].to_numpy(dtype=float),
                "intensity": frame_df[intensity_col].to_numpy(dtype=float),
                "q": frame_df[lower["q"]].to_numpy(dtype=float) if "q" in lower else None,
                "metadata": metadata,
            }
        )
    return profiles


def read_txt_profiles(paths):
    profiles = []
    for idx, path in enumerate(paths):
        profile = read_txt_profile(path)
        profile["frame"] = idx
        match = re.search(r"_(\d+)\.txt$", Path(path).name)
        if match:
            profile["frame"] = int(match.group(1))
        profiles.append(profile)
    return sorted(profiles, key=lambda item: (item.get("frame", 0), item["path"]))


def _profile_step(two_theta):
    diffs = np.diff(np.asarray(two_theta, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return np.nan
    return float(np.median(diffs))


def combine_profiles(profiles, overlap="blend"):
    if not profiles:
        raise ValueError("No profiles supplied for calibration")
    overlap = str(overlap or "blend").lower()
    if overlap != "blend":
        raise ValueError("Only blended overlap handling is implemented")
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
        raise ValueError("Calibration profiles have no finite 2theta values")
    step = float(np.median(steps)) if steps else 0.01
    min_tth = min(low for low, _ in ranges)
    max_tth = max(high for _, high in ranges)
    grid = np.arange(min_tth, max_tth + step * 0.5, step)
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
        values = np.interp(grid[in_range], two_theta, intensity)
        summed[in_range] += values
        counts[in_range] += 1
    valid = counts > 0
    metadata = dict(profiles[0].get("metadata", {}))
    metadata.update(
        {
            "overlap_policy": overlap,
            "source_frame_count": len(profiles),
            "source_frame_ranges": [
                {
                    "frame": profile.get("frame", idx),
                    "min_two_theta": float(np.nanmin(profile["two_theta"])),
                    "max_two_theta": float(np.nanmax(profile["two_theta"])),
                    "source": profile.get("path"),
                }
                for idx, profile in enumerate(profiles)
            ],
        }
    )
    return {
        "two_theta": grid[valid],
        "intensity": summed[valid] / counts[valid],
        "metadata": metadata,
    }


def _normalise_material(material, lattice_type=None, a=None, b=None, c=None):
    material = material or "LaB6 (cubic, Pm-3m)"
    if str(material).lower().startswith("lab6"):
        return {
            "material": "LaB6 (cubic, Pm-3m)",
            "lattice_type": "cubic",
            "a": LAB6_A,
            "b": None,
            "c": None,
        }
    return {
        "material": "custom",
        "lattice_type": lattice_type or "cubic",
        "a": float(a),
        "b": float(b) if b not in (None, "") else None,
        "c": float(c) if c not in (None, "") else None,
    }


def _pseudo_voigt_with_linear_bg(x, amplitude, center, fwhm, eta, m, b):
    width = max(float(fwhm), 1e-9)
    gaussian = np.exp(-np.log(2.0) * ((x - center) / (width / 2.0)) ** 2)
    lorentzian = 1.0 / (1.0 + ((x - center) / (width / 2.0)) ** 2)
    return amplitude * (eta * gaussian + (1.0 - eta) * lorentzian) + m * x + b


def fit_expected_peak(two_theta, intensity, expected_two_theta, window=0.35, search_center=None):
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    window = float(window)
    search_center = float(expected_two_theta if search_center is None else search_center)
    mask = (two_theta >= search_center - window) & (two_theta <= search_center + window)
    x = two_theta[mask]
    y = intensity[mask]
    if len(x) < 8:
        raise RuntimeError("Too few points for peak fit")
    baseline = float(np.nanpercentile(y, 10))
    y_above = y - baseline
    prominence = max(float(np.nanmax(y_above) - np.nanmin(y_above)) * 0.05, 0.0)
    peak_indices, properties = find_peaks(y_above, prominence=prominence)
    if len(peak_indices):
        if "prominences" in properties and len(properties["prominences"]):
            local_peak_index = int(peak_indices[int(np.argmax(properties["prominences"]))])
        else:
            local_peak_index = int(peak_indices[int(np.argmax(y_above[peak_indices]))])
    else:
        local_peak_index = int(np.nanargmax(y_above))
    observed_seed = float(x[local_peak_index])
    amplitude = float(y_above[local_peak_index])
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise RuntimeError("Peak has no positive amplitude")
    step = _profile_step(x)
    fwhm_guess = max(float(step) * 4 if np.isfinite(step) else 0.05, 0.02)
    half_height = baseline + amplitude * 0.5
    above_half = np.where(y >= half_height)[0]
    if len(above_half) >= 2:
        fwhm_guess = max(float(x[above_half[-1]] - x[above_half[0]]), fwhm_guess)
    bounds = (
        [0.0, search_center - window, max(fwhm_guess * 0.25, 1e-4), 0.0, -np.inf, -np.inf],
        [np.inf, search_center + window, window * 2.0, 1.0, np.inf, np.inf],
    )
    popt, pcov = curve_fit(
        _pseudo_voigt_with_linear_bg,
        x,
        y,
        p0=[amplitude, observed_seed, fwhm_guess, 0.5, 0.0, baseline],
        bounds=bounds,
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.ndim(pcov) == 2 else np.full(len(popt), np.nan)
    return {
        "center": float(popt[1]),
        "center_err": float(perr[1]) if np.isfinite(perr[1]) else np.nan,
        "amplitude": float(popt[0]),
        "fwhm": float(abs(popt[2])),
        "fwhm_err": float(perr[2]) if np.isfinite(perr[2]) else np.nan,
        "eta": float(popt[3]),
        "background_slope": float(popt[4]),
        "background_intercept": float(popt[5]),
        "observed_seed": observed_seed,
        "search_center": search_center,
    }


def _observed_peak_candidates(two_theta, intensity, prominence_fraction=0.015, min_distance_deg=0.08):
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    baseline = float(np.nanpercentile(intensity, 10))
    y_above = intensity - baseline
    prominence = max(float(np.nanmax(y_above) - np.nanmin(y_above)) * float(prominence_fraction), 0.0)
    step = _profile_step(two_theta)
    distance = 1
    if np.isfinite(step) and step > 0:
        distance = max(1, int(round(float(min_distance_deg) / step)))
    indices, properties = find_peaks(y_above, prominence=prominence, distance=distance)
    candidates = []
    prominences = properties.get("prominences", np.zeros(len(indices)))
    for idx, prominence_value in zip(indices, prominences):
        candidates.append(
            {
                "index": int(idx),
                "two_theta": float(two_theta[int(idx)]),
                "intensity": float(intensity[int(idx)]),
                "prominence": float(prominence_value),
            }
        )
    return sorted(candidates, key=lambda item: item["two_theta"])


def _fit_offset_model(matches, degree=1):
    if not matches:
        return np.asarray([0.0], dtype=float)
    degree = min(max(int(degree), 0), len(matches) - 1)
    x = np.asarray([match["expected_two_theta"] for match in matches], dtype=float)
    y = np.asarray([match["offset"] for match in matches], dtype=float)
    if degree == 0:
        weights = np.asarray([max(match.get("prominence", 0.0), 1e-9) for match in matches], dtype=float)
        return np.asarray([float(np.average(y, weights=weights))], dtype=float)
    weights = np.asarray([math.sqrt(max(match.get("prominence", 0.0), 1e-9)) for match in matches], dtype=float)
    return np.asarray(np.polyfit(x, y, degree, w=weights), dtype=float)


def _eval_offset_model(coefficients, expected_two_theta):
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.size == 1:
        return float(coefficients[0])
    return float(np.polyval(coefficients, float(expected_two_theta)))


def _score_assignments(assignments, tolerance):
    if not assignments:
        return -np.inf
    distance_penalty = sum(match["assignment_distance"] for match in assignments) / max(tolerance, 1e-9)
    prominence_score = sum(math.log1p(match["prominence"]) for match in assignments)
    offset_values = np.asarray([match["offset"] for match in assignments], dtype=float)
    smooth_penalty = float(np.nanstd(np.diff(offset_values))) if len(offset_values) > 2 else 0.0
    return len(assignments) * 100.0 + prominence_score - distance_penalty - smooth_penalty


def _assign_for_model(peaks, candidates, coefficients, tolerance):
    if not peaks or not candidates:
        return []
    max_prominence = max((candidate["prominence"] for candidate in candidates), default=1.0) or 1.0
    edges = []
    for peak_index, peak in enumerate(peaks):
        expected = float(peak.two_theta)
        predicted_observed = expected + _eval_offset_model(coefficients, expected)
        for candidate_index, candidate in enumerate(candidates):
            distance = abs(candidate["two_theta"] - predicted_observed)
            if distance <= tolerance:
                prominence_bonus = 0.12 * math.log1p(candidate["prominence"] / max_prominence)
                score = distance / max(tolerance, 1e-9) - prominence_bonus
                edges.append((score, distance, -candidate["prominence"], peak_index, candidate_index))
    used_peaks = set()
    used_candidates = set()
    assignments = []
    for _score, distance, _negative_prominence, peak_index, candidate_index in sorted(edges):
        if peak_index in used_peaks or candidate_index in used_candidates:
            continue
        peak = peaks[peak_index]
        candidate = candidates[candidate_index]
        expected = float(peak.two_theta)
        assignments.append(
            {
                "peak_index": int(peak_index),
                "candidate_index": int(candidate_index),
                "expected_two_theta": expected,
                "observed_two_theta": float(candidate["two_theta"]),
                "offset": float(candidate["two_theta"] - expected),
                "prominence": float(candidate["prominence"]),
                "intensity": float(candidate["intensity"]),
                "assignment_distance": float(distance),
                "label": peak.label,
            }
        )
        used_peaks.add(peak_index)
        used_candidates.add(candidate_index)
    return sorted(assignments, key=lambda item: item["peak_index"])


def _candidate_offset_models(peaks, candidates, search_window, tolerance):
    models = []
    for peak in peaks:
        expected = float(peak.two_theta)
        for candidate in candidates:
            offset = candidate["two_theta"] - expected
            if abs(offset) <= search_window:
                models.append({"coefficients": [float(offset)], "kind": "constant"})

    pairs = []
    for peak_index, peak in enumerate(peaks):
        expected = float(peak.two_theta)
        for candidate_index, candidate in enumerate(candidates):
            offset = candidate["two_theta"] - expected
            if abs(offset) <= search_window:
                pairs.append((peak_index, candidate_index, expected, candidate["two_theta"], offset))
    for i, first in enumerate(pairs):
        for second in pairs[i + 1 :]:
            first_peak, first_candidate, first_expected, first_observed, _first_offset = first
            second_peak, second_candidate, second_expected, second_observed, _second_offset = second
            if first_peak == second_peak or first_candidate == second_candidate:
                continue
            if abs(second_expected - first_expected) < 5.0:
                continue
            slope = (second_observed - first_observed) / (second_expected - first_expected)
            if not 0.85 <= slope <= 1.15:
                continue
            intercept = first_observed - slope * first_expected
            models.append({"coefficients": [float(slope - 1.0), float(intercept)], "kind": "linear"})

    best_by_key = {}
    for model in models:
        coefficients = model["coefficients"]
        if len(coefficients) == 1:
            key = ("constant", round(coefficients[0], 4))
        else:
            key = ("linear", round(coefficients[0], 5), round(coefficients[1], 3))
        best_by_key[key] = model
    return list(best_by_key.values())


def fingerprint_offset_model(peaks, candidates, search_window=2.0):
    """Register observed and predicted peak fingerprints with a simple smooth transform."""
    if not peaks or not candidates:
        return {
            "coefficients": [0.0],
            "kind": "constant",
            "score": -np.inf,
            "assignments": [],
            "tolerance": 0.0,
        }
    search_window = float(search_window)
    tolerance = max(0.08, min(search_window, max(search_window * 0.25, 0.18)))
    best = {
        "coefficients": [0.0],
        "kind": "constant",
        "score": -np.inf,
        "assignments": [],
        "tolerance": float(tolerance),
    }
    for model in _candidate_offset_models(peaks, candidates, search_window, tolerance):
        assignments = _assign_for_model(peaks, candidates, model["coefficients"], tolerance)
        score = _score_assignments(assignments, tolerance)
        if score > best["score"]:
            best = {
                "coefficients": [float(value) for value in model["coefficients"]],
                "kind": model["kind"],
                "score": float(score),
                "assignments": assignments,
                "tolerance": float(tolerance),
            }
    return best


def assign_calibration_peaks(
    two_theta,
    intensity,
    peaks: Sequence[PredictedPeak],
    search_window=2.0,
    model_degree=1,
):
    """Assign predicted hkl lines to detected observed peaks using one smooth offset model."""
    peaks = list(peaks or [])
    candidates = _observed_peak_candidates(two_theta, intensity)
    if not peaks or not candidates:
        return [], {
            "observed_peak_count": len(candidates),
            "initial_shift": 0.0,
            "model_coefficients": [0.0],
            "matches": [],
        }

    search_window = float(search_window)
    fingerprint = fingerprint_offset_model(peaks, candidates, search_window=search_window)
    best_assignments = fingerprint["assignments"]
    if not best_assignments:
        return [], {
            "observed_peak_count": len(candidates),
            "initial_shift": 0.0,
            "model_coefficients": [0.0],
            "fingerprint_model": fingerprint,
            "matches": [],
        }

    if best_assignments:
        robust_offsets = np.asarray([match["offset"] for match in best_assignments], dtype=float)
        median = float(np.nanmedian(robust_offsets))
        mad = float(np.nanmedian(np.abs(robust_offsets - median)))
        if np.isfinite(mad) and mad > 0:
            keep = np.abs(robust_offsets - median) <= max(3.0 * mad, 0.08)
            if np.any(keep):
                best_assignments = [match for match, keep_value in zip(best_assignments, keep) if keep_value]
        best_shift = float(np.nanmedian([match["offset"] for match in best_assignments]))

    coefficients = np.asarray(fingerprint["coefficients"], dtype=float)
    assignments = best_assignments
    final_tolerance = float(fingerprint["tolerance"])
    for _ in range(4):
        if assignments:
            coefficients = _fit_offset_model(assignments, degree=model_degree)
        updated = _assign_for_model(peaks, candidates, coefficients, final_tolerance)
        if not updated:
            break
        old_pairs = {(match["peak_index"], match["candidate_index"]) for match in assignments}
        new_pairs = {(match["peak_index"], match["candidate_index"]) for match in updated}
        assignments = updated
        if new_pairs == old_pairs:
            break

    if assignments:
        coefficients = _fit_offset_model(assignments, degree=model_degree)
        initial_shift = float(np.nanmedian([match["offset"] for match in assignments]))
    else:
        initial_shift = float(best_shift)
    for match in assignments:
        match["used_for_initial_shift"] = True
    return assignments, {
        "observed_peak_count": len(candidates),
        "initial_shift": initial_shift,
        "model_coefficients": [float(value) for value in np.atleast_1d(coefficients)],
        "fingerprint_model": {
            "kind": fingerprint["kind"],
            "coefficients": [float(value) for value in fingerprint["coefficients"]],
            "score": float(fingerprint["score"]),
            "tolerance": float(fingerprint["tolerance"]),
            "assigned_peak_count": len(fingerprint["assignments"]),
        },
        "assignment_tolerance": float(final_tolerance),
        "matches": assignments,
    }


def estimate_initial_twotheta_shift(two_theta, intensity, peaks: Sequence[PredictedPeak], search_window=2.0):
    _assignments, summary = assign_calibration_peaks(
        two_theta,
        intensity,
        peaks,
        search_window=search_window,
        model_degree=0,
    )
    return float(summary["initial_shift"]), summary["matches"]


def fit_calibration_peaks(
    two_theta,
    intensity,
    peaks: Sequence[PredictedPeak],
    fit_window=0.35,
    initial_shift=None,
):
    fitted = []
    assignment_window = max(float(fit_window), 2.0)
    if initial_shift is None:
        assignments, initial_shift_summary = assign_calibration_peaks(
            two_theta,
            intensity,
            peaks,
            search_window=assignment_window,
            model_degree=1,
        )
        initial_shift = initial_shift_summary["initial_shift"]
    else:
        initial_shift = float(initial_shift)
        assignments, initial_shift_summary = assign_calibration_peaks(
            two_theta,
            intensity,
            peaks,
            search_window=assignment_window,
            model_degree=0,
        )
        initial_shift_summary["initial_shift"] = initial_shift
    assignments_by_peak = {match["peak_index"]: match for match in assignments}
    for peak_index, peak in enumerate(peaks):
        try:
            assignment = assignments_by_peak.get(peak_index)
            if assignment is None:
                raise RuntimeError("No unique observed peak assigned")
            search_center = float(assignment["observed_two_theta"])
            refine_window = min(float(fit_window), 0.45)
            refine_window = max(refine_window, 0.08)
            result = fit_expected_peak(
                two_theta,
                intensity,
                peak.two_theta,
                window=refine_window,
                search_center=search_center,
            )
            center_shift = abs(result["center"] - peak.two_theta)
            search_shift = abs(result["center"] - search_center)
            result["center_shift"] = float(center_shift)
            result["search_shift"] = float(search_shift)
            result["assigned_observed_two_theta"] = float(assignment["observed_two_theta"])
            result["assignment_distance"] = float(assignment["assignment_distance"])
            usable = (
                np.isfinite(result["center"])
                and np.isfinite(result["fwhm"])
                and result["amplitude"] > 0
                and result["fwhm"] > 0
                and search_shift <= refine_window * 0.8
            )
            if not usable:
                result["rejection_reason"] = "quality_gate_failed"
        except Exception as exc:
            result = {"error": str(exc)}
            usable = False
        row = {
            "hkl": "".join(str(v) for v in (peak.hkl or ())),
            "label": peak.label,
            "multiplicity": peak.intensity,
            "expected_two_theta": float(peak.two_theta),
            "initial_shift": float(initial_shift),
            "usable": bool(usable),
        }
        row.update(result)
        if usable:
            row["offset"] = float(row["center"] - row["expected_two_theta"])
        fitted.append(row)
    initial_shift_summary["initial_shift"] = float(initial_shift)
    return fitted, initial_shift_summary


def fit_offset_polynomial(fitted_peaks, degree=2):
    used = [peak for peak in fitted_peaks if peak.get("usable") and not peak.get("offset_fit_outlier")]
    degree = int(degree)
    if degree < 0:
        raise ValueError("Polynomial degree must be >= 0")
    if len(used) <= degree:
        raise RuntimeError("Not enough usable calibration peaks for requested polynomial degree")
    x = np.asarray([peak["expected_two_theta"] for peak in used], dtype=float)
    y = np.asarray([peak["offset"] for peak in used], dtype=float)
    weights = None
    errs = np.asarray([peak.get("center_err", np.nan) for peak in used], dtype=float)
    if np.isfinite(errs).all() and np.any(errs > 0):
        weights = 1.0 / np.maximum(errs, 1e-12)
    coeffs = np.polyfit(x, y, degree, w=weights)
    return [float(value) for value in coeffs]


def fit_caglioti(fitted_peaks):
    used = [
        peak
        for peak in fitted_peaks
        if peak.get("usable") and not peak.get("caglioti_fit_outlier") and np.isfinite(peak.get("fwhm", np.nan))
    ]
    if len(used) < 3:
        return None
    two_theta = np.asarray([peak["expected_two_theta"] for peak in used], dtype=float)
    theta = np.radians(two_theta / 2.0)
    tan_theta = np.tan(theta)
    a = np.column_stack([tan_theta**2, tan_theta, np.ones_like(tan_theta)])
    y = np.asarray([peak["fwhm"] for peak in used], dtype=float) ** 2
    coeffs, *_ = np.linalg.lstsq(a, y, rcond=None)
    return {"U": float(coeffs[0]), "V": float(coeffs[1]), "W": float(coeffs[2])}


def _robust_outlier_mask(residuals, sigma=4.0, absolute_floor=0.0):
    residuals = np.asarray(residuals, dtype=float)
    finite = np.isfinite(residuals)
    mask = np.zeros(len(residuals), dtype=bool)
    if finite.sum() < 6:
        return mask
    center = float(np.nanmedian(residuals[finite]))
    mad = float(np.nanmedian(np.abs(residuals[finite] - center)))
    robust_sigma = 1.4826 * mad if np.isfinite(mad) else 0.0
    threshold = max(float(sigma) * robust_sigma, float(absolute_floor))
    if threshold <= 0:
        return mask
    mask[finite] = np.abs(residuals[finite] - center) > threshold
    return mask


def annotate_calibration_fit_outliers(fitted_peaks, polynomial_degree=2, discard_outliers=False):
    for peak in fitted_peaks:
        peak["offset_fit_outlier"] = False
        peak["caglioti_fit_outlier"] = False
        peak.pop("offset_fit_residual", None)
        peak.pop("caglioti_fit_residual", None)
    if not discard_outliers:
        return {"enabled": False, "offset_outliers": 0, "caglioti_outliers": 0}

    usable = [peak for peak in fitted_peaks if peak.get("usable")]
    degree = int(polynomial_degree)
    if len(usable) > degree + 4:
        x = np.asarray([peak["expected_two_theta"] for peak in usable], dtype=float)
        y = np.asarray([peak["offset"] for peak in usable], dtype=float)
        coeffs = np.polyfit(x, y, min(degree, len(usable) - 1))
        residuals = y - np.polyval(coeffs, x)
        outliers = _robust_outlier_mask(residuals, sigma=4.5, absolute_floor=0.08)
        if outliers.sum() <= max(1, len(usable) // 4):
            for peak, residual, is_outlier in zip(usable, residuals, outliers):
                peak["offset_fit_residual"] = float(residual)
                peak["offset_fit_outlier"] = bool(is_outlier)

    caglioti_candidates = [peak for peak in usable if np.isfinite(peak.get("fwhm", np.nan))]
    if len(caglioti_candidates) >= 7:
        fwhm = np.asarray([peak["fwhm"] for peak in caglioti_candidates], dtype=float)
        median_fwhm = float(np.nanmedian(fwhm))
        fwhm_mad = float(np.nanmedian(np.abs(fwhm - median_fwhm)))
        fwhm_sigma = 1.4826 * fwhm_mad if np.isfinite(fwhm_mad) else 0.0
        gross_threshold = max(4.5 * fwhm_sigma, 0.08, 1.5 * median_fwhm)
        gross_fwhm_outliers = (fwhm - median_fwhm) > gross_threshold
        initial = fit_caglioti(caglioti_candidates)
        if initial:
            x = np.asarray([peak["expected_two_theta"] for peak in caglioti_candidates], dtype=float)
            theta = np.radians(x / 2.0)
            predicted_sq = initial["U"] * np.tan(theta) ** 2 + initial["V"] * np.tan(theta) + initial["W"]
            predicted = np.sqrt(np.maximum(predicted_sq, 0.0))
            residuals = fwhm - predicted
            outliers = _robust_outlier_mask(residuals, sigma=4.0, absolute_floor=0.03) | gross_fwhm_outliers
            if outliers.sum() <= max(2, len(caglioti_candidates) // 4):
                for peak, residual, is_outlier in zip(caglioti_candidates, residuals, outliers):
                    peak["caglioti_fit_residual"] = float(residual)
                    peak["caglioti_fit_outlier"] = bool(is_outlier)

    return {
        "enabled": True,
        "offset_outliers": sum(1 for peak in fitted_peaks if peak.get("offset_fit_outlier")),
        "caglioti_outliers": sum(1 for peak in fitted_peaks if peak.get("caglioti_fit_outlier")),
    }


def evaluate_calibration(calibration, two_theta):
    coeffs = calibration.get("offset_polynomial_coefficients") or calibration.get("coefficients")
    if coeffs is None:
        raise ValueError("Calibration JSON does not contain offset polynomial coefficients")
    return np.polyval([float(value) for value in coeffs], np.asarray(two_theta, dtype=float))


def has_twotheta_correction(metadata):
    if not metadata:
        return False
    for key, value in metadata.items():
        if _metadata_key(key) == _metadata_key(CORRECTION_METADATA_KEY) and str(value).strip():
            return True
    return False


def apply_calibration_to_profile(two_theta, intensity, calibration, allow_recorrect=False, metadata=None):
    if has_twotheta_correction(metadata) and not allow_recorrect:
        return np.asarray(two_theta, dtype=float), np.asarray(intensity, dtype=float), False
    offsets = evaluate_calibration(calibration, two_theta)
    return np.asarray(two_theta, dtype=float) - offsets, np.asarray(intensity, dtype=float), True


def load_calibration(path):
    with open(path, "r", encoding="utf-8") as fh:
        calibration = json.load(fh)
    if not isinstance(calibration, dict) or calibration.get("type") != "twotheta_axis_calibration":
        raise ValueError(f"Not a quixrd 2theta calibration JSON: {path}")
    calibration.setdefault("path", str(path))
    return calibration


def _write_combined_txt(path, combined, metadata, calibration_path=None):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Format: quixrd combined I_vs_2th calibration profile\n")
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                continue
            if calibration_path and _metadata_key(key) == _metadata_key(CORRECTION_METADATA_KEY):
                continue
            fh.write(f"# {key}: {value}\n")
        if calibration_path:
            fh.write(f"# {CORRECTION_METADATA_KEY}: {calibration_path}\n")
        fh.write("# 2theta intensity\n")
        for x, y in zip(combined["two_theta"], combined["intensity"]):
            fh.write(f"{x:.8f} {y:.8g}\n")


def apply_calibration_to_txt_file(path, calibration_json, output_path=None):
    calibration = load_calibration(calibration_json)
    profile = read_txt_profile(path)
    corrected_tth, intensity, applied = apply_calibration_to_profile(
        profile["two_theta"],
        profile["intensity"],
        calibration,
        metadata=profile.get("metadata", {}),
    )
    output_path = Path(output_path or path)
    if not applied:
        return {"path": str(output_path), "applied": False}
    metadata = dict(profile.get("metadata", {}))
    metadata[CORRECTION_METADATA_KEY] = str(calibration_json)
    combined = {"two_theta": corrected_tth, "intensity": intensity}
    _write_combined_txt(output_path, combined, metadata, calibration_path=str(calibration_json))
    return {"path": str(output_path), "applied": True}


def apply_calibration_to_csv_file(path, calibration_json, output_path=None):
    calibration = load_calibration(calibration_json)
    df = pd.read_csv(path)
    correction_columns = [column for column in df.columns if _metadata_key(column) == _metadata_key(CORRECTION_METADATA_KEY)]
    if correction_columns and df[correction_columns[0]].astype(str).str.strip().replace("nan", "").any():
        return {"path": str(output_path or path), "applied": False}
    lower = {str(column).lower(): column for column in df.columns}
    tth_col = lower.get("2theta") or lower.get("two_theta") or lower.get("twotheta")
    if tth_col is None:
        raise ValueError(f"CSV file has no 2theta column: {path}")
    corrected = df.copy()
    corrected[tth_col] = corrected[tth_col].to_numpy(dtype=float) - evaluate_calibration(
        calibration,
        corrected[tth_col].to_numpy(dtype=float),
    )
    corrected[CORRECTION_METADATA_KEY] = str(calibration_json)
    output_path = Path(output_path or path)
    corrected.to_csv(output_path, index=False)
    return {"path": str(output_path), "applied": True}


def apply_calibration_to_exported_files(directory, calibration_json, scans=None):
    root = Path(directory)
    scan_set = {int(scan) for scan in scans} if scans is not None else None
    results = []
    for path in sorted(root.rglob("I_vs_2th_*.txt")):
        scan_match = re.match(r"I_vs_2th_(\d+)_", path.name)
        if scan_set is not None and (not scan_match or int(scan_match.group(1)) not in scan_set):
            continue
        results.append(apply_calibration_to_txt_file(path, calibration_json))
    for path in sorted(root.rglob("scan_*.csv")):
        scan_match = re.match(r"scan_(\d+)_", path.name)
        if scan_set is not None and (not scan_match or int(scan_match.group(1)) not in scan_set):
            continue
        results.append(apply_calibration_to_csv_file(path, calibration_json))
    return {
        "directory": str(root),
        "calibration": str(calibration_json),
        "applied": sum(1 for result in results if result.get("applied")),
        "skipped": sum(1 for result in results if not result.get("applied")),
        "files": results,
    }


def _save_calibration_figure(fig, path):
    path = Path(path)
    fig.savefig(path, dpi=150)
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path)
    return svg_path


def _plot_combined_profile(path, combined, peaks, fitted_peaks, title, show=False):
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.plot(combined["two_theta"], combined["intensity"], "-", linewidth=0.6, label="combined profile")
    x_min = float(np.nanmin(combined["two_theta"]))
    x_max = float(np.nanmax(combined["two_theta"]))
    major_tick, minor_tick = _tick_spacing_for_range(x_min, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(major_tick))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
    ax.grid(which="major", axis="both", linewidth=0.45, alpha=0.28)
    ax.grid(which="minor", axis="both", linewidth=0.3, alpha=0.16)
    ax.set_xlim(x_min, x_max)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)
    overlay_predicted_peaks(ax, peaks, min_label_spacing=0.7)
    model_label_added = False
    for peak in fitted_peaks:
        if not peak.get("usable"):
            continue
        required = ["amplitude", "center", "fwhm", "eta", "background_slope", "background_intercept"]
        if any(not np.isfinite(float(peak.get(key, np.nan))) for key in required):
            continue
        center = float(peak["center"])
        fwhm = max(float(peak["fwhm"]), 1e-6)
        half_width = max(4.0 * fwhm, 0.08)
        local_x = np.linspace(center - half_width, center + half_width, 120)
        data_min = float(np.nanmin(combined["two_theta"]))
        data_max = float(np.nanmax(combined["two_theta"]))
        local_x = local_x[(local_x >= data_min) & (local_x <= data_max)]
        if len(local_x) < 3:
            continue
        local_y = _pseudo_voigt_with_linear_bg(
            local_x,
            float(peak["amplitude"]),
            center,
            fwhm,
            float(peak["eta"]),
            float(peak["background_slope"]),
            float(peak["background_intercept"]),
        )
        ax.plot(
            local_x,
            local_y,
            "-",
            color="tab:orange",
            linewidth=0.45,
            alpha=0.9,
            label="modelled peaks" if not model_label_added else None,
        )
        model_label_added = True
    matched_x = [peak["center"] for peak in fitted_peaks if peak.get("usable")]
    if matched_x:
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            matched_x,
            ymin,
            ymax,
            colors="tab:orange",
            linewidth=0.55,
            alpha=0.8,
            label="fitted centers",
        )
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("2theta (deg)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_calibration_figure(fig, path)
    if not show:
        plt.close(fig)
    return fig


def _plot_fit_curves(path, fitted_peaks, offset_coeffs, caglioti, title, show=False):
    used = [peak for peak in fitted_peaks if peak.get("usable")]
    offset_used = [peak for peak in used if not peak.get("offset_fit_outlier")]
    offset_excluded = [peak for peak in used if peak.get("offset_fit_outlier")]
    caglioti_used = [peak for peak in used if not peak.get("caglioti_fit_outlier")]
    caglioti_excluded = [peak for peak in used if peak.get("caglioti_fit_outlier")]
    fig, (ax_offset, ax_fwhm) = plt.subplots(2, 1, figsize=(11.25, 12), sharex=True)
    if offset_used:
        x = np.asarray([peak["expected_two_theta"] for peak in offset_used], dtype=float)
        offsets = np.asarray([peak["offset"] for peak in offset_used], dtype=float)
        offset_err = np.asarray([peak.get("center_err", np.nan) for peak in offset_used], dtype=float)
        if np.isfinite(offset_err).any():
            ax_offset.errorbar(
                x,
                offsets,
                yerr=offset_err,
                fmt=".",
                capsize=3,
                color="tab:blue",
                ecolor="tab:blue",
                label="peak offsets",
            )
        else:
            ax_offset.plot(x, offsets, ".", color="tab:blue", label="peak offsets")
        x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 300)
        ax_offset.plot(x_fit, np.polyval(offset_coeffs, x_fit), "-", linewidth=0.8, label="polynomial")
    if offset_excluded:
        x_bad = np.asarray([peak["expected_two_theta"] for peak in offset_excluded], dtype=float)
        y_bad = np.asarray([peak["offset"] for peak in offset_excluded], dtype=float)
        err_bad = np.asarray([peak.get("center_err", np.nan) for peak in offset_excluded], dtype=float)
        if np.isfinite(err_bad).any():
            ax_offset.errorbar(
                x_bad,
                y_bad,
                yerr=err_bad,
                fmt=".",
                capsize=3,
                color="tab:cyan",
                ecolor="tab:cyan",
                label="excluded offsets",
            )
        else:
            ax_offset.plot(x_bad, y_bad, ".", color="tab:cyan", label="excluded offsets")
    if caglioti_used:
        x = np.asarray([peak["expected_two_theta"] for peak in caglioti_used], dtype=float)
        fwhm = np.asarray([peak["fwhm"] for peak in caglioti_used], dtype=float)
        fwhm_err = np.asarray([peak.get("fwhm_err", np.nan) for peak in caglioti_used], dtype=float)
        if np.isfinite(fwhm_err).any():
            ax_fwhm.errorbar(
                x,
                fwhm,
                yerr=fwhm_err,
                fmt=".",
                capsize=3,
                color="tab:blue",
                ecolor="tab:blue",
                label="FWHM",
            )
        else:
            ax_fwhm.plot(x, fwhm, ".", color="tab:blue", label="FWHM")
        if caglioti:
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 300)
            theta = np.radians(x_fit / 2.0)
            y = caglioti["U"] * np.tan(theta) ** 2 + caglioti["V"] * np.tan(theta) + caglioti["W"]
            ax_fwhm.plot(x_fit, np.sqrt(np.maximum(y, 0.0)), "-", linewidth=0.8, label="Caglioti")
    if caglioti_excluded:
        x_bad = np.asarray([peak["expected_two_theta"] for peak in caglioti_excluded], dtype=float)
        y_bad = np.asarray([peak["fwhm"] for peak in caglioti_excluded], dtype=float)
        err_bad = np.asarray([peak.get("fwhm_err", np.nan) for peak in caglioti_excluded], dtype=float)
        if np.isfinite(err_bad).any():
            ax_fwhm.errorbar(
                x_bad,
                y_bad,
                yerr=err_bad,
                fmt=".",
                capsize=3,
                color="tab:cyan",
                ecolor="tab:cyan",
                label="excluded FWHM",
            )
        else:
            ax_fwhm.plot(x_bad, y_bad, ".", color="tab:cyan", label="excluded FWHM")
    for ax in (ax_offset, ax_fwhm):
        x_min, x_max = ax.get_xlim()
        major_tick, minor_tick = _tick_spacing_for_range(x_min, x_max)
        ax.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which="major", axis="both", linewidth=0.45, alpha=0.28)
        ax.grid(which="minor", axis="both", linewidth=0.3, alpha=0.16)
    fwhm_ymax = ax_fwhm.get_ylim()[1]
    ax_fwhm.set_ylim(0, fwhm_ymax)
    ax_offset.set_ylabel("Observed - expected 2theta (deg)")
    ax_offset.legend(fontsize=8)
    ax_fwhm.set_xlabel("2theta (deg)")
    ax_fwhm.set_ylabel("FWHM (deg)")
    ax_fwhm.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    _save_calibration_figure(fig, path)
    if not show:
        plt.close(fig)
    return fig


def _default_output_dir(input_paths, output_dir):
    if output_dir:
        path = Path(output_dir)
    else:
        first = Path(input_paths[0])
        path = (first if first.is_dir() else first.parent) / "calibration"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_nxs_calibration_source(nxs_path, output_dir, flat_file_directory="./flat/", flat_file_numbers=None):
    from quixrd.nxs_export.XPAD_XRD_nxs_export import S140XRD, _scan_no_from_nxs_name

    nxs_path = Path(nxs_path)
    scan_number = _scan_no_from_nxs_name(nxs_path.name)
    if scan_number is None:
        raise ValueError(f"Could not determine scan number from NXS filename: {nxs_path}")
    try:
        import h5py

        with h5py.File(nxs_path, "r") as fh:
            root = next(iter(fh.keys()))
            scan_type = fh[f"{root}/scan_config/name"][()]
            if isinstance(scan_type, bytes):
                scan_type = scan_type.decode("utf-8")
            if "delta" not in str(scan_type).lower():
                raise ValueError(f"LaB6 NXS calibration input must be a delta scan, found '{scan_type}'")
    except ValueError:
        raise
    except Exception:
        pass

    exporter = S140XRD(
        nxs_file_directory=str(nxs_path.parent),
        export_directory=str(output_dir),
        flat_file_directory=flat_file_directory,
        flat_file_numbers=flat_file_numbers or [],
    )
    exporter.extract_S140XRD_chidelta(scan_number, showGraph=False, saveGraph=False)
    exporter.extract_S140XRD(scan_number, showGraph=False, saveGraph=False)
    txt_paths = sorted(Path(output_dir).glob(f"I_vs_2th_{scan_number}_delta_*.txt"))
    csv_paths = sorted(Path(output_dir).glob(f"scan_{scan_number}_*.csv"))
    return {"scan_number": scan_number, "txt_paths": txt_paths, "csv_paths": csv_paths}


def build_twotheta_calibration(
    input_paths,
    source_type="txt",
    output_dir=None,
    material="LaB6 (cubic, Pm-3m)",
    lattice_type="cubic",
    a=None,
    b=None,
    c=None,
    energy=None,
    wavelength=None,
    polynomial_degree=2,
    max_index=8,
    fit_window=0.35,
    discard_outliers=False,
    show_plots=False,
    overlap="blend",
    flat_file_directory="./flat/",
    flat_file_numbers=None,
):
    input_paths = [Path(path) for path in (input_paths if isinstance(input_paths, (list, tuple)) else [input_paths])]
    output_dir = _default_output_dir(input_paths, output_dir)
    source_type = str(source_type or "txt").lower()
    copied_sources = []

    if source_type == "nxs":
        exported = export_nxs_calibration_source(
            input_paths[0],
            output_dir,
            flat_file_directory=flat_file_directory,
            flat_file_numbers=flat_file_numbers,
        )
        profiles = read_txt_profiles(exported["txt_paths"])
        copied_sources = [str(path) for path in exported["txt_paths"] + exported["csv_paths"]]
    elif source_type == "csv":
        profiles = read_csv_profiles(input_paths[0])
        target = output_dir / input_paths[0].name
        if input_paths[0].resolve() != target.resolve():
            shutil.copy2(input_paths[0], target)
        copied_sources = [str(target)]
    else:
        profiles = read_txt_profiles(input_paths)
        for path in input_paths:
            target = output_dir / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
            copied_sources.append(str(target))

    combined = combine_profiles(profiles, overlap=overlap)
    metadata = combined["metadata"]
    energy = energy if energy not in (None, "") else metadata.get("energy") or metadata.get("Energy")
    wavelength = wavelength if wavelength not in (None, "") else energy_to_wavelength(energy)
    if wavelength is None:
        raise ValueError("Calibration requires energy metadata or a supplied energy/wavelength")
    energy = energy if energy not in (None, "") else wavelength_to_energy(wavelength)

    lattice = _normalise_material(material, lattice_type=lattice_type, a=a, b=b, c=c)
    peaks = lattice_predicted_peaks(
        lattice["lattice_type"],
        lattice["a"],
        b=lattice["b"],
        c=lattice["c"],
        wavelength=wavelength,
        max_index=max_index,
        min_two_theta=float(np.nanmin(combined["two_theta"])),
        max_two_theta=float(np.nanmax(combined["two_theta"])),
        phase_name="LaB6" if lattice["material"].startswith("LaB6") else lattice["material"],
    )
    peaks = [
        PredictedPeak(
            peak.two_theta,
            f"{peak.label} m={int(peak.intensity)}",
            peak.intensity,
            peak.hkl,
        )
        for peak in peaks
    ]
    fitted_peaks, initial_shift_summary = fit_calibration_peaks(
        combined["two_theta"],
        combined["intensity"],
        peaks,
        fit_window=fit_window,
    )
    outlier_summary = annotate_calibration_fit_outliers(
        fitted_peaks,
        polynomial_degree=polynomial_degree,
        discard_outliers=discard_outliers,
    )
    offset_coeffs = fit_offset_polynomial(fitted_peaks, degree=polynomial_degree)
    caglioti = fit_caglioti(fitted_peaks)

    stamp = _timestamp()
    stem = f"twotheta_calibration_{_safe_suffix(lattice['material'])}_{stamp}"
    combined_txt = output_dir / f"{stem}_combined.txt"
    combined_csv = output_dir / f"{stem}_combined.csv"
    profile_plot = output_dir / f"{stem}_profile.png"
    fit_plot = output_dir / f"{stem}_fits.png"
    profile_plot_svg = profile_plot.with_suffix(".svg")
    fit_plot_svg = fit_plot.with_suffix(".svg")
    json_path = output_dir / f"{stem}.json"

    out_metadata = {
        "Generated By": "quixrd",
        "Calibration Material": lattice["material"],
        "Energy": energy,
        "Wavelength": wavelength,
        "Overlap Policy": overlap,
    }
    _write_combined_txt(combined_txt, combined, {**metadata, **out_metadata})
    pd.DataFrame({"2theta": combined["two_theta"], "intensity": combined["intensity"]}).to_csv(combined_csv, index=False)
    profile_fig = _plot_combined_profile(
        profile_plot,
        combined,
        peaks,
        fitted_peaks,
        f"{lattice['material']} calibration profile",
        show=show_plots,
    )
    fit_fig = _plot_fit_curves(
        fit_plot,
        fitted_peaks,
        offset_coeffs,
        caglioti,
        "2theta calibration fits",
        show=show_plots,
    )
    if show_plots:
        plt.show()

    calibration = {
        "type": "twotheta_axis_calibration",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "material": lattice["material"],
        "lattice": {key: lattice[key] for key in ("lattice_type", "a", "b", "c")},
        "energy": float(energy),
        "wavelength": float(wavelength),
        "source_type": source_type,
        "source_files": [str(path) for path in input_paths],
        "copied_or_exported_sources": copied_sources,
        "overlap_policy": overlap,
        "initial_shift": initial_shift_summary,
        "discard_outliers": bool(discard_outliers),
        "outlier_summary": outlier_summary,
        "polynomial_degree": int(polynomial_degree),
        "offset_polynomial_coefficients": offset_coeffs,
        "caglioti": caglioti,
        "peaks": fitted_peaks,
        "combined_txt": str(combined_txt),
        "combined_csv": str(combined_csv),
        "profile_plot": str(profile_plot),
        "profile_plot_svg": str(profile_plot_svg),
        "fit_plot": str(fit_plot),
        "fit_plot_svg": str(fit_plot_svg),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(calibration, fh, indent=2)
    return {
        "path": str(json_path),
        "calibration": calibration,
        "combined_txt": str(combined_txt),
        "combined_csv": str(combined_csv),
        "profile_plot": str(profile_plot),
        "profile_plot_svg": str(profile_plot_svg),
        "fit_plot": str(fit_plot),
        "fit_plot_svg": str(fit_plot_svg),
    }
