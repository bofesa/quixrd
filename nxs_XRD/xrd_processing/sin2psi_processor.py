"""Sin2psi processing helpers.

The sin2psi analysis workflow in this module is based on the approach from
materialsguy/Bessy-II-KMC-II-insitu-sin2psi:
https://github.com/materialsguy/Bessy-II-KMC-II-insitu-sin2psi
Archive DOI: https://doi.org/10.5281/zenodo.17349576
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

if not os.environ.get("NXS_XRD_GUI_INTERACTIVE"):
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Model
from scipy.signal import find_peaks

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - optional dependency
    sm = None

logger = logging.getLogger(__name__)

ENERGY_TO_WAVELENGTH_KEV_A = 12.3984193


def gauss_func(x, max_intensity, two_theta, fwhm):
    return max_intensity * np.exp(-np.log(2) * ((x - two_theta) / (fwhm / 2)) ** 2)


def lorentz_func(x, max_intensity, two_theta, fwhm):
    return max_intensity / (1 + ((x - two_theta) / (fwhm / 2)) ** 2)


def psv_func(x, max_intensity, two_theta, fwhm, nu):
    return nu * gauss_func(x, max_intensity, two_theta, fwhm) + (1 - nu) * lorentz_func(
        x, max_intensity, two_theta, fwhm
    )


class PSVModel(Model):
    def __init__(self):
        super(PSVModel, self).__init__(psv_func, missing="drop")
        self.name = "pseudo-Voigt"

    def guess(self, data=None, **kws):
        assert "x" in kws
        x = kws["x"]
        max_intensity = max(data)
        try:
            two_theta = x[data.idxmax()]
        except Exception:
            two_theta = x[np.argmax(data)]
        try:
            fwhm = x[data > max_intensity / 2].ptp()
        except Exception:
            fwhm = 0.1
        if np.isnan(fwhm) or fwhm == 0:
            fwhm = (max(x) - min(x)) * 0.1
        self.set_param_hint("nu", min=0, max=1)
        self.set_param_hint("max_intensity", min=0)
        self.set_param_hint("two_theta", min=min(x), max=max(x))
        self.set_param_hint("fwhm", min=0)
        return self.make_params(
            max_intensity=max_intensity, two_theta=two_theta, fwhm=fwhm, nu=0.5
        )


def fit(x, y, background_lower, peak, background_upper, correction_order=1):
    x_bg = np.concatenate((x[background_lower], x[background_upper]))
    y_bg = np.concatenate((y[background_lower], y[background_upper]))

    if correction_order == 2:
        m, n, c = np.polyfit(x_bg, y_bg, 2)

        def bg_correction(xs):
            return m * xs**2 + n * xs + c

    elif correction_order == 1:
        n, c = np.polyfit(x_bg, y_bg, 1)
        m = 0.0

        def bg_correction(xs):
            return n * xs + c

    else:
        raise ValueError("correction_order must be 1 or 2")

    xs = x[peak]
    ys = y[peak]
    y_corr = ys - bg_correction(xs)

    model = PSVModel()
    pars = model.guess(y_corr, x=xs)
    fit_res = model.fit(y_corr, pars, x=xs)

    two_theta = (fit_res.params["two_theta"].value, fit_res.params["two_theta"].stderr)
    max_intensity = (
        fit_res.params["max_intensity"].value,
        fit_res.params["max_intensity"].stderr,
    )
    nu = (fit_res.params["nu"].value, fit_res.params["nu"].stderr)
    fwhm = (fit_res.params["fwhm"].value, fit_res.params["fwhm"].stderr)
    return two_theta, max_intensity, nu, fwhm, m, n, c


def _choose_fit_window(tth, intensity):
    peaks, _ = find_peaks(
        intensity, prominence=(np.max(intensity) - np.min(intensity)) * 0.05
    )
    if len(peaks) == 0:
        peak_idx = int(np.argmax(intensity))
    else:
        peak_idx = int(peaks[np.argmax(intensity[peaks])])
    window = max(5, int(len(tth) * 0.02))
    left = max(0, peak_idx - window)
    right = min(len(tth) - 1, peak_idx + window)
    return left, right, peak_idx


def _estimate_step(tth_arr):
    diffs = np.diff(np.asarray(tth_arr, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 0.01
    return float(np.median(diffs))


def _auto_fit_windows(tth_arr, intensity_arr):
    left, right, peak_idx = _choose_fit_window(tth_arr, intensity_arr)
    width = max(3, right - left)
    return {
        "background_lower": float(tth_arr[max(0, left - width)]),
        "peak_lower": float(tth_arr[left]),
        "peak_upper": float(tth_arr[right]),
        "background_upper": float(tth_arr[min(len(tth_arr) - 1, right + width)]),
    }, left, right, peak_idx


def _seeded_fit_windows(tth_arr, intensity_arr, seed_center, track_window):
    center = float(seed_center)
    step = _estimate_step(tth_arr)
    half_width = float(track_window) if track_window is not None else max(step * 20, np.ptp(tth_arr) * 0.05)
    half_width = max(half_width, step * 3)
    margin = max(half_width * 0.5, step * 5)

    peak_lower = center - half_width
    peak_upper = center + half_width
    background_lower = peak_lower - margin
    background_upper = peak_upper + margin

    left = int(np.searchsorted(tth_arr, peak_lower, side="left"))
    right = int(np.searchsorted(tth_arr, peak_upper, side="right")) - 1
    left = max(0, min(left, len(tth_arr) - 1))
    right = max(left, min(right, len(tth_arr) - 1))
    peak_idx = int(left + np.argmax(intensity_arr[left : right + 1]))

    return {
        "background_lower": float(background_lower),
        "peak_lower": float(peak_lower),
        "peak_upper": float(peak_upper),
        "background_upper": float(background_upper),
    }, left, right, peak_idx


def fit_peak(
    tth,
    intensity,
    background_lower,
    peak_lower,
    peak_upper,
    background_upper,
    plot=True,
    correction_order=1,
    plot_path: Optional[str] = None,
    chi=None,
):
    tth_arr = np.asarray(tth, dtype=float)
    intensity_arr = np.asarray(intensity, dtype=float)

    background_lower = np.intersect1d(
        np.argwhere(np.array(tth_arr) > background_lower),
        np.argwhere(np.array(tth_arr) < peak_lower),
    ).astype(int).ravel()
    background_upper = np.intersect1d(
        np.argwhere(np.array(tth_arr) > peak_upper),
        np.argwhere(np.array(tth_arr) < background_upper),
    ).astype(int).ravel()

    if len(background_lower) == 0 or len(background_upper) == 0:
        raise RuntimeError("Insufficient background points")

    background_lower_intensity = intensity_arr[background_lower]
    background_upper_intensity = intensity_arr[background_upper]
    median_background_level = np.median(
        np.concatenate((background_lower_intensity, background_upper_intensity))
    )
    mean_background_level = np.mean(
        np.concatenate((background_lower_intensity, background_upper_intensity))
    )

    peak = np.intersect1d(
        np.argwhere(np.array(tth_arr) > peak_lower),
        np.argwhere(np.array(tth_arr) < peak_upper),
    ).astype(int).ravel()
    if len(peak) == 0:
        raise RuntimeError("Insufficient peak points")

    two_theta, max_intensity, nu, fwhm, m, n, c = fit(
        tth_arr,
        intensity_arr,
        background_lower,
        peak,
        background_upper,
        correction_order=correction_order,
    )

    if plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(tth_arr, intensity_arr, "b-", label="data", linewidth=0.5)
        ax.plot(tth_arr[background_lower], intensity_arr[background_lower], "r.", linewidth=0.5)
        ax.plot(tth_arr[background_upper], intensity_arr[background_upper], "r.", linewidth=0.5)
        x_bg = np.arange(
            np.min(tth_arr[background_lower]),
            np.max(tth_arr[background_upper]) + 0.1,
            0.1,
        )
        y_bg = m * x_bg**2 + n * x_bg + c
        ax.plot(x_bg, y_bg, "black", linestyle="--", label="Background", linewidth=0.5)
        ax.plot(tth_arr[peak], intensity_arr[peak], "b.")
        x_fit = tth_arr[peak]
        y_fit = psv_func(x_fit, max_intensity[0], two_theta[0], fwhm[0], nu[0])
        y_fit_linear = y_fit + m * x_fit**2 + n * x_fit + c
        # ax.plot(x_fit, y_fit, "g--", linewidth=0.5)     # plot the fitted peak without background
        ax.plot(x_fit, y_fit_linear, "g-", linewidth=0.5, label=f"Fitted peak")   # plot the fitted peak with background
        ax.set_xlabel("2$\\theta$ (deg)")
        ax.set_ylabel("Intensity")
        ax.set_xlim(np.min(tth_arr[background_lower]), np.max(tth_arr[background_upper]))
        ax.legend()
        if plot_path:
            scan_no = plot_path.split("\\")[-3]
            frame_no = plot_path.split("\\")[-1].replace("_fit.png", "")
            psi_text = f" psi={90.0 - float(chi):.3f}" if chi is not None and pd.notna(chi) else ""
            title = f"{scan_no} {frame_no}{psi_text}\ncenter={two_theta[0]:.4f}, fwhm={fwhm[0]:.4f}, nu={nu[0]:.4f}"
        else:
            title = f"center={two_theta[0]:.4f}, fwhm={fwhm[0]:.4f}, nu={nu[0]:.4f}"
        ax.set_title(title)
        # set font size for all text in the figure
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
        fig.tight_layout()
        if plot_path:
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()
    return two_theta, max_intensity, nu, fwhm, mean_background_level, median_background_level, m, n, c


def parse_txt_scan(filepath):
    metadata = {
        "filename": os.path.basename(filepath),
        "scan_type": None,
        "chi": None,
        "temperature": None,
        "energy": None,
        "start_time": None,
        "frame_time": None,
        "metadata": {},
    }
    rows = []
    header_lines = []
    in_data = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                if header_lines:
                    in_data = True
                continue
            if line.startswith("#") and not in_data:
                header_lines.append(line[1:].strip())
                continue
            in_data = True
            if line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append([float(p) for p in parts])
            except Exception:
                continue

    for line in header_lines:
        norm = re.sub(r"\s+", " ", line).strip()
        m = re.match(r"([^:=]+)\s*[:=]\s*(.+)$", norm)
        if m:
            key = _metadata_key(m.group(1))
            metadata["metadata"][key] = _coerce_metadata_value(m.group(2).strip())
        m = re.search(r"scan\s*type\s*:\s*(.+)$", norm, flags=re.IGNORECASE)
        if m:
            metadata["scan_type"] = m.group(1).strip()
        m = re.search(r"\bchi\b\s*[:=]\s*([+\-0-9.eE]+)", norm, flags=re.IGNORECASE)
        if m:
            try:
                metadata["chi"] = float(m.group(1))
            except Exception:
                pass
        m = re.search(r"\btemperature\b\s*[:=]\s*([+\-0-9.eE]+)", norm, flags=re.IGNORECASE)
        if m:
            try:
                metadata["temperature"] = float(m.group(1))
            except Exception:
                pass
        m = re.search(r"\benergy\b\s*[:=]\s*([+\-0-9.eE]+)", norm, flags=re.IGNORECASE)
        if m:
            try:
                metadata["energy"] = float(m.group(1))
            except Exception:
                pass
        m = re.search(r"\bstart\s*time\b\s*[:=]\s*(.+)$", norm, flags=re.IGNORECASE)
        if m:
            metadata["start_time"] = m.group(1).strip()
        m = re.search(r"\bframe\s*time\b\s*[:=]\s*(.+)$", norm, flags=re.IGNORECASE)
        if m:
            metadata["frame_time"] = m.group(1).strip()

    if metadata["scan_type"] is None:
        logger.warning("Missing Scan Type in %s", filepath)
    if metadata["chi"] is None:
        logger.warning("Missing Chi in %s", filepath)
    if metadata["temperature"] is None:
        logger.warning("Missing Temperature in %s", filepath)
    if metadata["energy"] is None:
        logger.warning("Missing Energy in %s", filepath)

    if not rows:
        raise RuntimeError(f"No numeric data found in {filepath}")
    arr = np.asarray(rows, dtype=float)
    if arr.shape[1] < 2:
        raise RuntimeError(f"Need at least two numeric columns in {filepath}")

    metadata["tth"] = arr[:, 0]
    metadata["intensity"] = arr[:, 1]
    if arr.shape[1] >= 3:
        metadata["q"] = arr[:, 2]
    return metadata


def _metadata_key(key):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(key).strip().lower())).strip("_")


def _coerce_metadata_value(value):
    text = str(value).strip()
    try:
        return float(text)
    except Exception:
        return text


def _scan_name_parts(filename):
    stem = Path(filename).stem
    if not stem.startswith("I_vs_2th_"):
        return []
    return stem.split("_")[3:]


def _extract_scan_number(parts):
    if not parts:
        return None
    try:
        return int(parts[0])
    except Exception:
        return None


def _extract_frame_index(parts):
    for token in reversed(parts):
        if token.isdigit():
            return int(token)
    return 0


def discover_scan_files(data_dir, scan_number):
    data_path = Path(data_dir)
    matched = []
    for path in sorted(data_path.glob("I_vs_2th_*.txt")):
        parts = _scan_name_parts(path.name)
        if _extract_scan_number(parts) != scan_number:
            continue
        accept = False
        if len(parts) == 2:
            try:
                meta = parse_txt_scan(str(path))
                accept = (meta.get("scan_type") or "").strip().lower() in {
                    "ascan_chi",
                    "dscan_chi",
                }
            except Exception as exc:
                logger.warning("Skipping %s: %s", path.name, exc)
        elif len(parts) >= 3:
            accept = parts[1].strip().lower() == "chi"
        if accept:
            matched.append((_extract_frame_index(parts), str(path)))
    return [p for _, p in sorted(matched, key=lambda item: (item[0], item[1]))]


def discover_scan_numbers(data_dir, include_raw=True, include_processed=True):
    """Return available scan numbers from exported TXT files and/or processed scan folders."""
    data_path = Path(data_dir)
    scan_numbers = set()
    if include_raw:
        for path in sorted(data_path.glob("I_vs_2th_*.txt")):
            scan_number = _extract_scan_number(_scan_name_parts(path.name))
            if scan_number is not None:
                scan_numbers.add(scan_number)
    if include_processed:
        export_root = data_path / "sin2psi_export"
        for scan_dir in sorted(export_root.glob("scan_*")):
            scan_number = _scan_number_from_dir(scan_dir)
            if scan_number is not None:
                scan_numbers.add(scan_number)
    return sorted(scan_numbers)


def _frame_csv_columns():
    return [
        "frame_index",
        "filename",
        "scan_type",
        "chi",
        "psi_deg",
        "sin2psi",
        "temperature",
        "energy",
        "start_time",
        "frame_time",
        "metadata_json",
        "peak_center",
        "peak_center_err",
        "amplitude",
        "amplitude_err",
        "fwhm",
        "fwhm_err",
        "nu",
        "nu_err",
        "background_mean",
        "background_median",
        "bg_coef_0",
        "bg_coef_1",
        "bg_coef_2",
        "left_idx",
        "right_idx",
        "peak_idx",
        "window_mode",
        "seed_center",
        "background_lower",
        "peak_lower",
        "peak_upper",
        "background_upper",
        "fit_success",
        "excluded_from_sin2psi",
    ]


def _ensure_clean_scan_dir(scan_dir, backup):
    scan_dir = Path(scan_dir)
    if scan_dir.exists():
        if backup:
            stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = scan_dir.with_name(f"{scan_dir.name}_backup_{stamp}")
            shutil.move(str(scan_dir), str(backup_dir))
    scan_dir.mkdir(parents=True, exist_ok=True)


def fit_frame(tth, intensity, plot=False, **fit_kwargs):
    tth_arr = np.asarray(tth, dtype=float)
    intensity_arr = np.asarray(intensity, dtype=float)
    explicit_windows = "background_lower" in fit_kwargs
    seed_center = fit_kwargs.get("seed_center")
    track_window = fit_kwargs.get("track_window")
    fallback_to_auto = bool(fit_kwargs.get("fallback_to_auto", True))

    if explicit_windows:
        windows = {
            "background_lower": float(fit_kwargs["background_lower"]),
            "peak_lower": float(fit_kwargs["peak_lower"]),
            "peak_upper": float(fit_kwargs["peak_upper"]),
            "background_upper": float(fit_kwargs["background_upper"]),
        }
        left = int(np.searchsorted(tth_arr, windows["peak_lower"], side="left"))
        right = int(np.searchsorted(tth_arr, windows["peak_upper"], side="right")) - 1
        left = max(0, min(left, len(tth_arr) - 1))
        right = max(left, min(right, len(tth_arr) - 1))
        peak_idx = int(left + np.argmax(intensity_arr[left : right + 1]))
        window_mode = "explicit"
    else:
        if seed_center is not None:
            windows, left, right, peak_idx = _seeded_fit_windows(
                tth_arr, intensity_arr, seed_center=seed_center, track_window=track_window
            )
            window_mode = "seeded"
        else:
            windows, left, right, peak_idx = _auto_fit_windows(tth_arr, intensity_arr)
            window_mode = "auto"

    correction_order = int(fit_kwargs.get("correction_order", 1))
    plot_path = fit_kwargs.get("plot_path")
    chi = fit_kwargs.get("chi")

    try:
        two_theta, max_intensity, nu, fwhm, mean_bg, median_bg, m, n, c = fit_peak(
            tth_arr,
            intensity_arr,
            windows["background_lower"],
            windows["peak_lower"],
            windows["peak_upper"],
            windows["background_upper"],
            plot=plot,
            correction_order=correction_order,
            plot_path=plot_path,
            chi=chi,
        )
    except Exception:
        if explicit_windows or seed_center is None or not fallback_to_auto:
            raise
        windows, left, right, peak_idx = _auto_fit_windows(tth_arr, intensity_arr)
        window_mode = "auto-fallback"
        two_theta, max_intensity, nu, fwhm, mean_bg, median_bg, m, n, c = fit_peak(
            tth_arr,
            intensity_arr,
            windows["background_lower"],
            windows["peak_lower"],
            windows["peak_upper"],
            windows["background_upper"],
            plot=plot,
            correction_order=correction_order,
            plot_path=plot_path,
            chi=chi,
        )

    peak_mask = np.where((tth_arr > windows["peak_lower"]) & (tth_arr < windows["peak_upper"]))[0]
    x_fit = tth_arr[peak_mask]
    y_peak = psv_func(x_fit, max_intensity[0], two_theta[0], fwhm[0], nu[0])
    y_bg = m * x_fit**2 + n * x_fit + c
    y_combined = y_peak + y_bg

    return {
        "center": float(two_theta[0]),
        "center_err": float(two_theta[1]) if two_theta[1] is not None else np.nan,
        "amplitude": float(max_intensity[0]),
        "amplitude_err": float(max_intensity[1]) if max_intensity[1] is not None else np.nan,
        "fwhm": float(fwhm[0]),
        "fwhm_err": float(fwhm[1]) if fwhm[1] is not None else np.nan,
        "nu": float(nu[0]),
        "nu_err": float(nu[1]) if nu[1] is not None else np.nan,
        "background_mean": float(mean_bg),
        "background_median": float(median_bg),
        "bg_coef_0": float(c),
        "bg_coef_1": float(n),
        "bg_coef_2": float(m),
        "x_fit": x_fit.tolist(),
        "y_peak_fit": y_peak.tolist(),
        "y_bg_fit": y_bg.tolist(),
        "y_combined_fit": y_combined.tolist(),
        "left_idx": int(left),
        "right_idx": int(right),
        "peak_idx": int(peak_idx),
        "background_lower": float(windows["background_lower"]),
        "peak_lower": float(windows["peak_lower"]),
        "peak_upper": float(windows["peak_upper"]),
        "background_upper": float(windows["background_upper"]),
        "window_mode": window_mode,
        "seed_center": float(seed_center) if seed_center is not None else np.nan,
        "fit_success": True,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def _scan_metadata_from_df(df):
    if df.empty:
        return {}
    first = df.sort_values("frame_index").iloc[0]
    keys = [
        "filename",
        "scan_type",
        "chi",
        "psi_deg",
        "sin2psi",
        "temperature",
        "energy",
        "start_time",
        "frame_time",
    ]
    metadata = {key: first.get(key) for key in keys if key in df.columns}
    if "metadata_json" in df.columns and pd.notna(first.get("metadata_json")):
        try:
            extra = json.loads(first.get("metadata_json"))
            if isinstance(extra, dict):
                metadata.update(extra)
        except Exception:
            logger.warning("Could not parse metadata_json for scan summary")
    return _json_safe(metadata)


def _normalise_ranges(ranges):
    normalised = []
    for item in ranges or []:
        if item is None:
            continue
        lower, upper = item
        lower = -np.inf if lower is None else float(lower)
        upper = np.inf if upper is None else float(upper)
        if lower > upper:
            lower, upper = upper, lower
        normalised.append((lower, upper))
    return normalised


def _in_any_range(values, ranges):
    ranges = _normalise_ranges(ranges)
    mask = pd.Series(False, index=values.index)
    numeric = pd.to_numeric(values, errors="coerce")
    for lower, upper in ranges:
        mask |= numeric.between(lower, upper, inclusive="both")
    return mask


def _fit_sin2psi_regression(used, y_column="peak_center"):
    x = used["sin2psi"].to_numpy(dtype=float)
    y = used[y_column].to_numpy(dtype=float)
    yerr = used["peak_center_err"].to_numpy(dtype=float)
    weights_used = None

    if np.all(np.isfinite(yerr)) and np.any(yerr > 0):
        weights = 1.0 / np.square(yerr)
        weights_used = weights.tolist()
    else:
        weights = np.ones_like(x)

    if sm is not None:
        if np.all(np.isfinite(yerr)) and np.any(yerr > 0):
            model = sm.WLS(y, sm.add_constant(x), weights=weights).fit()
        else:
            model = sm.OLS(y, sm.add_constant(x)).fit()
        slope = float(model.params[1])
        intercept = float(model.params[0])
        slope_err = float(model.bse[1])
        intercept_err = float(model.bse[0])
        resid = model.resid.astype(float)
    else:
        xw = np.vstack([x, np.ones_like(x)]).T
        w = np.asarray(weights, dtype=float)
        wmat = np.diag(w)
        beta = np.linalg.inv(xw.T @ wmat @ xw) @ (xw.T @ wmat @ y)
        slope = float(beta[0])
        intercept = float(beta[1])
        resid = y - (slope * x + intercept)
        dof = max(len(x) - 2, 1)
        s2 = float(np.sum(w * resid**2) / dof)
        cov = np.linalg.inv(xw.T @ wmat @ xw) * s2
        slope_err = float(np.sqrt(max(cov[0, 0], 0.0)))
        intercept_err = float(np.sqrt(max(cov[1, 1], 0.0)))

    return {
        "slope": slope,
        "slope_err": slope_err,
        "intercept": intercept,
        "intercept_err": intercept_err,
        "chi2": float(np.sum(np.square(resid))),
        "rms": float(np.sqrt(np.mean(np.square(resid)))),
        "n_points": int(len(used)),
        "weights_used": weights_used,
        "residuals": np.asarray(resid, dtype=float).tolist(),
    }, np.asarray(resid, dtype=float)


def _energy_to_wavelength(energy):
    if energy in (None, ""):
        return None
    energy = float(energy)
    if energy <= 0:
        raise ValueError("Energy must be positive")
    if energy > 1000:
        energy = energy / 1000.0
    return ENERGY_TO_WAVELENGTH_KEV_A / energy


def _resolve_wavelength(wavelength=None, energy=None, df=None):
    if wavelength not in (None, ""):
        return float(wavelength)
    if energy not in (None, ""):
        return _energy_to_wavelength(energy)
    if df is not None and "energy" in df.columns:
        values = pd.to_numeric(df["energy"], errors="coerce").dropna()
        if not values.empty:
            return _energy_to_wavelength(float(values.iloc[0]))
    raise ValueError("Stress calculation requires wavelength, energy, or energy metadata")


def _two_theta_to_d(two_theta, wavelength):
    theta = np.radians(pd.to_numeric(two_theta, errors="coerce") / 2.0)
    return float(wavelength) / (2.0 * np.sin(theta))


def _reference_d0(reference_d0=None, reference_two_theta=None, wavelength=None):
    if reference_d0 not in (None, ""):
        return float(reference_d0), None
    if reference_two_theta not in (None, ""):
        d0 = _two_theta_to_d(float(reference_two_theta), wavelength)
        return float(d0), float(reference_two_theta)
    return None, None


def _weighted_line_fit(x, y, yerr=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if yerr is None:
        w = np.ones_like(x)
    else:
        yerr = np.asarray(yerr, dtype=float)
        if np.all(np.isfinite(yerr)) and np.any(yerr > 0):
            w = 1.0 / np.square(np.where(yerr > 0, yerr, np.nanmedian(yerr[yerr > 0])))
        else:
            w = np.ones_like(x)
    design = np.vstack([x, np.ones_like(x)]).T
    wmat = np.diag(w)
    beta = np.linalg.inv(design.T @ wmat @ design) @ (design.T @ wmat @ y)
    resid = y - (beta[0] * x + beta[1])
    dof = max(len(x) - 2, 1)
    s2 = float(np.sum(w * resid**2) / dof)
    cov = np.linalg.inv(design.T @ wmat @ design) * s2
    return float(beta[0]), float(beta[1]), cov, resid


def calculate_sin2psi_stress(
    df,
    y_column="peak_center",
    elastic_E=None,
    elastic_nu=None,
    elastic_E_units=None,
    reference_two_theta=None,
    reference_d0=None,
    wavelength=None,
    energy=None,
):
    if elastic_E in (None, "") or elastic_nu in (None, ""):
        return None
    elastic_E = float(elastic_E)
    elastic_nu = float(elastic_nu)
    stress_units = str(elastic_E_units).strip() if elastic_E_units not in (None, "") else "same_as_E"
    used = df.loc[~df["excluded_from_sin2psi"]].dropna(subset=[y_column, "sin2psi"]).copy()
    if len(used) < 3:
        raise RuntimeError("Need at least three usable points for stress calculation")
    wavelength = _resolve_wavelength(wavelength=wavelength, energy=energy, df=used)
    d_values = _two_theta_to_d(used[y_column], wavelength)
    twotheta_err = pd.to_numeric(used.get("peak_center_err"), errors="coerce")
    d_err = None
    if twotheta_err.notna().all():
        theta = np.radians(pd.to_numeric(used[y_column], errors="coerce") / 2.0)
        twotheta_err_rad = np.radians(twotheta_err.to_numpy(dtype=float))
        d_err = np.abs(-0.5 * d_values / np.tan(theta) * twotheta_err_rad)

    d0, resolved_two_theta = _reference_d0(
        reference_d0=reference_d0,
        reference_two_theta=reference_two_theta,
        wavelength=wavelength,
    )
    x = used["sin2psi"].to_numpy(dtype=float)
    if d0 is not None:
        strain = (d_values - d0) / d0
        strain_err = None if d_err is None else d_err / d0
        slope, intercept, cov, resid = _weighted_line_fit(x, strain, strain_err)
        stress = elastic_E / (1.0 + elastic_nu) * slope
        stress_err = elastic_E / (1.0 + elastic_nu) * float(np.sqrt(max(cov[0, 0], 0.0)))
        method = "reference_d0" if reference_d0 not in (None, "") else "reference_two_theta"
        return {
            "stress": float(stress),
            "stress_err": float(abs(stress_err)),
            "stress_units": stress_units,
            "elastic_E": elastic_E,
            "elastic_E_units": stress_units,
            "elastic_nu": elastic_nu,
            "stress_method": method,
            "stress_reference_two_theta": resolved_two_theta,
            "stress_reference_d0": float(d0),
            "stress_inferred_d0": None,
            "stress_wavelength": float(wavelength),
            "stress_slope": float(slope),
            "stress_intercept": float(intercept),
            "stress_residuals": np.asarray(resid, dtype=float).tolist(),
        }

    slope, intercept, cov, resid = _weighted_line_fit(x, d_values, d_err)
    if abs(intercept) < 1e-15:
        raise RuntimeError("Cannot infer stress from near-zero d intercept")
    ratio = slope / intercept
    denominator = 1.0 + elastic_nu + 2.0 * elastic_nu * ratio
    if abs(denominator) < 1e-15:
        raise RuntimeError("Equibiaxial stress denominator is too close to zero")
    stress_over_E = ratio / denominator
    stress = elastic_E * stress_over_E
    dr_da = 1.0 / intercept
    dr_db = -slope / (intercept * intercept)
    ratio_var = dr_da**2 * cov[0, 0] + dr_db**2 * cov[1, 1] + 2.0 * dr_da * dr_db * cov[0, 1]
    ratio_err = math.sqrt(max(float(ratio_var), 0.0))
    dstress_dr = elastic_E * (1.0 + elastic_nu) / (denominator * denominator)
    inferred_d0 = intercept / (1.0 - 2.0 * elastic_nu * stress_over_E)
    return {
        "stress": float(stress),
        "stress_err": float(abs(dstress_dr) * ratio_err),
        "stress_units": stress_units,
        "elastic_E": elastic_E,
        "elastic_E_units": stress_units,
        "elastic_nu": elastic_nu,
        "stress_method": "equibiaxial_inferred_d0",
        "stress_reference_two_theta": None,
        "stress_reference_d0": None,
        "stress_inferred_d0": float(inferred_d0),
        "stress_wavelength": float(wavelength),
        "stress_slope": float(slope),
        "stress_intercept": float(intercept),
        "stress_residuals": np.asarray(resid, dtype=float).tolist(),
    }


def _sin2psi_exclusion_mask(
    df,
    excluded_frames=None,
    exclude_chi_ranges=None,
    exclude_sin2psi_ranges=None,
):
    mask = pd.Series(False, index=df.index)
    excluded_frames = sorted(set(int(x) for x in (excluded_frames or [])))
    if excluded_frames:
        mask |= pd.to_numeric(df["frame_index"], errors="coerce").isin(excluded_frames)
    if exclude_chi_ranges:
        mask |= _in_any_range(df["chi"], exclude_chi_ranges)
    if exclude_sin2psi_ranges:
        mask |= _in_any_range(df["sin2psi"], exclude_sin2psi_ranges)
    return mask, excluded_frames


def _auto_exclude_sin2psi_outliers(df, base_mask, sigma=3.0, max_iter=1, y_column="peak_center"):
    sigma = float(sigma)
    max_iter = int(max_iter)
    if sigma <= 0 or max_iter <= 0:
        return base_mask.copy(), []

    mask = base_mask.copy()
    auto_frames = []
    for _ in range(max_iter):
        used = df.loc[~mask].dropna(subset=[y_column, "sin2psi"])
        if len(used) <= 3:
            break
        _, resid = _fit_sin2psi_regression(used, y_column=y_column)
        resid_std = float(np.nanstd(resid, ddof=1)) if len(resid) > 1 else 0.0
        if not np.isfinite(resid_std) or resid_std <= 0:
            break
        worst_position = int(np.nanargmax(np.abs(resid)))
        if abs(float(resid[worst_position])) <= sigma * resid_std:
            break
        used_indices = used.index.to_numpy()
        new_index = used_indices[worst_position]
        if mask.loc[new_index]:
            break
        mask.loc[new_index] = True
        auto_frames.append(int(df.loc[new_index, "frame_index"]))
    return mask, sorted(set(auto_frames))


def _theta_scale(sample_two_theta, reference_two_theta):
    # Currently unused: corrections are applied as absolute 2theta offsets.
    # Keep this helper for a possible future return to theta-dependent scaling.
    sample_theta = np.radians(pd.to_numeric(sample_two_theta, errors="coerce") / 2.0)
    reference_theta = math.radians(float(reference_two_theta) / 2.0)
    ref_tan = math.tan(reference_theta)
    if not np.isfinite(ref_tan) or abs(ref_tan) < 1e-12:
        raise ValueError(f"Invalid reference two-theta for correction scaling: {reference_two_theta}")
    return np.tan(sample_theta) / ref_tan


def load_sin2psi_correction(correction_json):
    with open(correction_json, "r", encoding="utf-8") as fh:
        correction = json.load(fh)
    if not isinstance(correction, dict):
        raise ValueError(f"Sin2psi correction must be a JSON object: {correction_json}")
    method = correction.get("method", "polynomial")
    if method == "polynomial":
        required = ["coefficients", "reference_two_theta"]
    elif method == "gaussian_process":
        required = ["training_x", "training_y", "reference_two_theta", "gp_length_scale", "gp_signal_variance"]
    else:
        raise ValueError(f"Unknown sin2psi correction method '{method}' in {correction_json}")
    for key in required:
        if key not in correction:
            raise ValueError(f"Missing '{key}' in sin2psi correction: {correction_json}")
    correction.setdefault("source_file", str(correction_json))
    return correction


def _correction_paths(correction_json):
    if correction_json is None:
        return []
    if isinstance(correction_json, (str, os.PathLike)):
        text = os.fspath(correction_json)
        if ";" in text:
            return [part.strip() for part in text.split(";") if part.strip()]
        return [correction_json]
    if isinstance(correction_json, Sequence) and not isinstance(correction_json, (bytes, bytearray)):
        paths = []
        for item in correction_json:
            paths.extend(_correction_paths(item))
        return paths
    return [correction_json]


def load_sin2psi_corrections(correction_json):
    return [load_sin2psi_correction(path) for path in _correction_paths(correction_json)]


def _representative_peak_center_for_correction(df):
    if "peak_center" not in df.columns:
        return None, None

    work = df.copy()
    work["peak_center"] = pd.to_numeric(work["peak_center"], errors="coerce")
    if "fit_success" in work.columns:
        success = work["fit_success"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        success = pd.Series(True, index=work.index)
    work = work.loc[work["peak_center"].notna()]
    if work.empty:
        return None, None

    sort_columns = ["frame_index"] if "frame_index" in work.columns else None
    ordered = work.sort_values(sort_columns) if sort_columns else work
    first = ordered.iloc[0]
    first_success = bool(success.loc[first.name]) if first.name in success.index else True
    if first_success and np.isfinite(float(first["peak_center"])):
        return float(first["peak_center"]), "first_frame_peak_center"

    successful = work.loc[success.reindex(work.index, fill_value=False)]
    if successful.empty:
        return None, None
    return float(np.nanmedian(successful["peak_center"].to_numpy(dtype=float))), "median_successful_peak_center"


def select_sin2psi_correction(correction_json, df):
    corrections = load_sin2psi_corrections(correction_json)
    if not corrections:
        return None, None
    scan_two_theta, selection_rule = _representative_peak_center_for_correction(df)
    if scan_two_theta is None:
        return None, None

    selectable = []
    for correction in corrections:
        try:
            reference = float(correction["reference_two_theta"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(reference):
            selectable.append((abs(reference - scan_two_theta), correction, reference))
    if not selectable:
        raise ValueError("No supplied sin2psi correction JSON contains a usable reference_two_theta")

    _, selected, reference = min(selectable, key=lambda item: item[0])
    selection = {
        "selected_correction_file": selected.get("source_file"),
        "selected_correction_method": selected.get("method", "polynomial"),
        "selected_correction_reference_two_theta": float(reference),
        "scan_representative_two_theta": float(scan_two_theta),
        "selection_rule": selection_rule,
        "candidate_correction_files": [correction.get("source_file") for correction in corrections],
    }
    return selected, selection


def _gp_kernel(x1, x2, length_scale, signal_variance):
    x1 = np.asarray(x1, dtype=float).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=float).reshape(1, -1)
    return float(signal_variance) * np.exp(-0.5 * np.square((x1 - x2) / float(length_scale)))


def _gp_predict(x_pred, x_train, y_train, noise, length_scale, signal_variance, return_std=False):
    x_pred = np.asarray(x_pred, dtype=float)
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    noise = np.asarray(noise, dtype=float)
    k_train = _gp_kernel(x_train, x_train, length_scale, signal_variance)
    k_train += np.diag(np.square(noise) + 1e-10)
    k_pred = _gp_kernel(x_pred, x_train, length_scale, signal_variance)
    try:
        alpha = np.linalg.solve(k_train, y_train)
        inv_k_kpred_t = np.linalg.solve(k_train, k_pred.T)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(k_train)
        alpha = pinv @ y_train
        inv_k_kpred_t = pinv @ k_pred.T
    mean = k_pred @ alpha
    if not return_std:
        return mean
    k_self = np.diag(_gp_kernel(x_pred, x_pred, length_scale, signal_variance))
    variance = np.maximum(k_self - np.sum(k_pred * inv_k_kpred_t.T, axis=1), 0.0)
    return mean, np.sqrt(variance)


def _evaluate_sin2psi_correction(correction, sin2psi_values):
    method = correction.get("method", "polynomial")
    if method == "polynomial":
        coeffs = [float(value) for value in correction["coefficients"]]
        return np.polyval(coeffs, pd.to_numeric(sin2psi_values, errors="coerce"))
    if method == "gaussian_process":
        return _gp_predict(
            pd.to_numeric(sin2psi_values, errors="coerce"),
            correction["training_x"],
            correction["training_y"],
            correction.get("training_noise", [1e-6] * len(correction["training_x"])),
            correction["gp_length_scale"],
            correction["gp_signal_variance"],
        )
    raise ValueError(f"Unknown sin2psi correction method: {method}")


def _apply_sin2psi_correction(df, correction):
    corrected = df.copy()
    reference_two_theta = float(correction["reference_two_theta"])
    raw_correction = _evaluate_sin2psi_correction(correction, corrected["sin2psi"])
    if not correction.get("reference_two_theta_provided", True):
        raw_correction = raw_correction - reference_two_theta
    corrected["sin2psi_correction_reference"] = raw_correction
    corrected["sin2psi_correction_scale"] = 1.0
    corrected["sin2psi_correction"] = raw_correction
    corrected["peak_center_uncorrected"] = corrected["peak_center"]
    corrected["peak_center_corrected"] = corrected["peak_center"] - corrected["sin2psi_correction"]
    return corrected


def generate_sin2psi_correction(
    data_dir,
    scan_number,
    degree=2,
    method="polynomial",
    reference_two_theta=None,
    gp_length_scale=0.25,
    gp_signal_variance=None,
    output_path=None,
    excluded_frames=None,
    exclude_chi_ranges=None,
    exclude_sin2psi_ranges=None,
):
    """
    Generate a sin2psi correction polynomial from a reference scan.
    args:
        data_dir: str or Path, path to the data directory containing the reference scan CSV
        scan_number: int, the scan number of the reference scan
        degree: int, degree of the polynomial correction (default: 2)
        method: "polynomial" or "gaussian_process"
        reference_two_theta: optional 2theta angle to fit offsets from; if None, fit true peak angle
        gp_length_scale: Gaussian-process squared-exponential length scale in sin2psi units
        gp_signal_variance: optional Gaussian-process signal variance; if None, estimated from targets
        output_path: str or Path, optional path to save the correction JSON (default: None)
        excluded_frames: list of int, optional list of frame indices to exclude from the fit
        exclude_chi_ranges: list of tuple, optional list of (lower, upper) chi ranges to exclude
        exclude_sin2psi_ranges: list of tuple, optional list of (lower, upper) sin2psi ranges to exclude
        """
    scan_dir = Path(data_dir) / "sin2psi_export" / f"scan_{int(scan_number)}"
    csv_path = scan_dir / f"scan_{int(scan_number)}_fits.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing reference scan fit CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df["psi_deg"] = df["chi"].apply(lambda c: 90.0 - float(c) if pd.notna(c) else np.nan)
    df["sin2psi"] = df["psi_deg"].apply(
        lambda p: math.sin(math.radians(float(p))) ** 2 if pd.notna(p) else np.nan
    )
    exclusion_mask, excluded_frames = _sin2psi_exclusion_mask(
        df,
        excluded_frames=excluded_frames,
        exclude_chi_ranges=exclude_chi_ranges,
        exclude_sin2psi_ranges=exclude_sin2psi_ranges,
    )
    used = df.loc[~exclusion_mask].dropna(subset=["sin2psi", "peak_center"])
    if len(used) <= int(degree):
        raise RuntimeError("Not enough usable reference points for requested correction polynomial degree")

    yerr = pd.to_numeric(used.get("peak_center_err"), errors="coerce")
    weights = None
    if yerr.notna().all() and (yerr > 0).any():
        weights = 1.0 / yerr.to_numpy(dtype=float)
    peak_values = used["peak_center"].to_numpy(dtype=float)
    if reference_two_theta is None:
        correction_target = peak_values
        correction_y = "peak_center"
        correction_reference_two_theta = float(np.average(peak_values, weights=weights))
        reference_two_theta_provided = False
    else:
        correction_reference_two_theta = float(reference_two_theta)
        correction_target = peak_values - correction_reference_two_theta
        correction_y = "peak_center_offset_from_reference_two_theta"
        reference_two_theta_provided = True

    x_train = used["sin2psi"].to_numpy(dtype=float)
    method = str(method).lower().strip()
    if method in {"gp", "gaussian_process", "gaussian-process"}:
        method = "gaussian_process"
    if method not in {"polynomial", "gaussian_process"}:
        raise ValueError("Correction method must be 'polynomial' or 'gaussian_process'")
    if method == "polynomial":
        coefficients = np.polyfit(x_train, correction_target, int(degree), w=weights)
    else:
        coefficients = None
        if gp_signal_variance is None:
            gp_signal_variance = float(np.nanvar(correction_target))
            if not np.isfinite(gp_signal_variance) or gp_signal_variance <= 0:
                gp_signal_variance = 1.0
    gp_noise = yerr.to_numpy(dtype=float) if yerr.notna().all() else np.full(len(used), 1e-6)

    correction = {
        "type": "sin2psi_chi_correction",
        "method": method,
        "created_at": _output_timestamp(),
        "source_scan": int(scan_number),
        "source_csv": str(csv_path),
        "x": "sin2psi",
        "y": correction_y,
        "reference_two_theta": correction_reference_two_theta,
        "reference_two_theta_provided": reference_two_theta_provided,
        "n_points": int(len(used)),
        "excluded_frames": excluded_frames,
        "exclude_chi_ranges": _normalise_ranges(exclude_chi_ranges),
        "exclude_sin2psi_ranges": _normalise_ranges(exclude_sin2psi_ranges),
    }
    if method == "polynomial":
        correction.update(
            {
                "degree": int(degree),
                "coefficients": [float(value) for value in coefficients],
            }
        )
    else:
        correction.update(
            {
                "training_x": [float(value) for value in x_train],
                "training_y": [float(value) for value in correction_target],
                "training_noise": [float(value) for value in gp_noise],
                "gp_length_scale": float(gp_length_scale),
                "gp_signal_variance": float(gp_signal_variance),
            }
        )

    if output_path is None:
        output_dir = Path(data_dir) / "sin2psi_export" / "calibrations"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = _unique_path(
            output_dir / f"sin2psi_correction_scan_{int(scan_number)}_{_output_timestamp()}.json"
        )
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_path = output_path.with_suffix(".png")
    x_fit = np.linspace(float(used["sin2psi"].min()), float(used["sin2psi"].max()), 200)
    y_fit_std = None
    if method == "gaussian_process":
        y_fit, y_fit_std = _gp_predict(
            x_fit,
            correction["training_x"],
            correction["training_y"],
            correction["training_noise"],
            correction["gp_length_scale"],
            correction["gp_signal_variance"],
            return_std=True,
        )
    else:
        y_fit = _evaluate_sin2psi_correction(correction, x_fit)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_yerr = pd.to_numeric(used.get("peak_center_err"), errors="coerce")
    if plot_yerr.notna().any():
        ax.errorbar(
            used["sin2psi"],
            correction_target,
            yerr=plot_yerr.to_numpy(dtype=float),
            fmt=".",
            capsize=3,
            label="reference points",
        )
    else:
        ax.plot(used["sin2psi"], correction_target, ".", label="reference points")
    excluded = df.loc[exclusion_mask].dropna(subset=["sin2psi", "peak_center"])
    if not excluded.empty:
        excluded_targets = excluded["peak_center"].to_numpy(dtype=float)
        if reference_two_theta_provided:
            excluded_targets = excluded_targets - correction_reference_two_theta
        excluded_yerr = pd.to_numeric(excluded.get("peak_center_err"), errors="coerce")
        if excluded_yerr.notna().any():
            ax.errorbar(
                excluded["sin2psi"],
                excluded_targets,
                yerr=excluded_yerr.to_numpy(dtype=float),
                fmt=".",
                capsize=3,
                label="excluded",
            )
        else:
            ax.plot(excluded["sin2psi"], excluded_targets, ".", label="excluded")
    fit_label = f"degree {int(degree)} fit" if method == "polynomial" else "Gaussian process fit"
    ax.plot(x_fit, y_fit, "-", linewidth=0.8, label=fit_label)
    if y_fit_std is not None:
        ax.fill_between(
            x_fit,
            y_fit - 2.0 * y_fit_std,
            y_fit + 2.0 * y_fit_std,
            alpha=0.2,
            label="GP +/- 2 sigma",
        )
    ax.set_xlabel("sin2psi")
    ax.set_ylabel("2theta offset from reference" if reference_two_theta_provided else "2theta")
    ax.set_title(f"Correction scan {int(scan_number)} - {method.replace('_', ' ')}")
    ylim_values = list(correction_target)
    if not excluded.empty:
        ylim_values.extend(excluded_targets)
    _set_ylim_from_points(ax, ylim_values)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(correction), fh, indent=2)
    return {"correction": correction, "path": str(output_path), "plot_path": str(plot_path)}


def perform_sin2psi_fit(
    df,
    scan_dir,
    excluded_frames=None,
    exclude_chi_ranges=None,
    exclude_sin2psi_ranges=None,
    auto_exclude=False,
    auto_exclude_sigma=3.0,
    auto_exclude_max_iter=1,
    correction_json=None,
    elastic_E=None,
    elastic_nu=None,
    elastic_E_units=None,
    stress_reference_two_theta=None,
    stress_reference_d0=None,
    stress_wavelength=None,
    stress_energy=None,
):
    scan_path = Path(scan_dir)
    df = df.copy()
    df["psi_deg"] = df["chi"].apply(lambda c: 90.0 - float(c) if pd.notna(c) else np.nan)
    df["sin2psi"] = df["psi_deg"].apply(
        lambda p: math.sin(math.radians(float(p))) ** 2 if pd.notna(p) else np.nan
    )
    correction, correction_selection = select_sin2psi_correction(correction_json, df)
    y_column = "peak_center"
    if correction is not None:
        df = _apply_sin2psi_correction(df, correction)
        y_column = "peak_center_corrected"
    exclusion_mask, excluded_frames = _sin2psi_exclusion_mask(
        df,
        excluded_frames=excluded_frames,
        exclude_chi_ranges=exclude_chi_ranges,
        exclude_sin2psi_ranges=exclude_sin2psi_ranges,
    )
    auto_excluded_frames = []
    if auto_exclude:
        exclusion_mask, auto_excluded_frames = _auto_exclude_sin2psi_outliers(
            df,
            exclusion_mask,
            sigma=auto_exclude_sigma,
            max_iter=auto_exclude_max_iter,
            y_column=y_column,
        )
    df["excluded_from_sin2psi"] = exclusion_mask
    used = df.loc[~df["excluded_from_sin2psi"]].dropna(subset=[y_column])
    if used.empty:
        raise RuntimeError("No usable points for sin2psi regression")

    summary, resid = _fit_sin2psi_regression(used, y_column=y_column)
    stress_summary = calculate_sin2psi_stress(
        df,
        y_column=y_column,
        elastic_E=elastic_E,
        elastic_nu=elastic_nu,
        elastic_E_units=elastic_E_units,
        reference_two_theta=stress_reference_two_theta,
        reference_d0=stress_reference_d0,
        wavelength=stress_wavelength,
        energy=stress_energy,
    )
    if stress_summary is not None:
        summary.update(stress_summary)
    summary.update(
        {
        "excluded_frames": excluded_frames,
        "exclude_chi_ranges": _normalise_ranges(exclude_chi_ranges),
        "exclude_sin2psi_ranges": _normalise_ranges(exclude_sin2psi_ranges),
        "auto_exclude": bool(auto_exclude),
        "auto_exclude_sigma": float(auto_exclude_sigma),
        "auto_exclude_max_iter": int(auto_exclude_max_iter),
        "auto_excluded_frames": auto_excluded_frames,
        "correction_applied": correction is not None,
        "correction_file": correction_selection.get("selected_correction_file") if correction_selection else None,
        "correction_files": [str(path) for path in _correction_paths(correction_json)],
        "correction_selection": correction_selection,
        "correction": correction,
        "fit_y_column": y_column,
        "metadata": _scan_metadata_from_df(df),
        }
    )
    _save_sin2psi_outputs(df, summary, scan_path)
    return summary


def _save_sin2psi_outputs(df, summary, scan_dir):
    scan_number = _scan_number_from_dir(scan_dir)
    if scan_number is None:
        plot_path = Path(scan_dir) / "sin2psi_plot.png"
    else:
        plot_path = Path(scan_dir) / f"scan_{scan_number}_sin2psi_plot.png"
    json_path = Path(scan_dir) / "sin2psi_fit_params.json"

    y_column = summary.get("fit_y_column", "peak_center")
    used = df.loc[~df["excluded_from_sin2psi"]].dropna(subset=[y_column])
    excluded = df.loc[df["excluded_from_sin2psi"]].dropna(subset=[y_column])

    was_interactive = plt.isinteractive()
    plt.ioff()
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        if summary.get("correction_applied") and "peak_center_uncorrected" in df.columns:
            ax.plot(
                df["sin2psi"],
                df["peak_center_uncorrected"],
                ".",
                alpha=0.2,
                label="uncorrected",
            )
        ax.plot(used["sin2psi"], used[y_column], ".", label="used")
        if not excluded.empty:
            ax.plot(excluded["sin2psi"], excluded[y_column], ".", label="excluded")

        xline = np.linspace(float(df["sin2psi"].min()), float(df["sin2psi"].max()), 200)
        yline = summary["slope"] * xline + summary["intercept"]
        ax.plot(xline, yline, "-", label="fit", linewidth=0.5, color="black")
        ax.set_xlabel("sin2psi")
        ax.set_ylabel(y_column)
        title = f"scan {scan_number} sin2psi fit" if scan_number is not None else "sin2psi fit"
        if summary.get("correction_applied"):
            title += " (corrected)"
        ax.set_title(title)
        ylim_values = list(used[y_column])
        if not excluded.empty:
            ylim_values.extend(excluded[y_column])
        if summary.get("correction_applied") and "peak_center_uncorrected" in df.columns:
            ylim_values.extend(df["peak_center_uncorrected"].dropna())
        _set_ylim_from_points(ax, ylim_values)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
    finally:
        if fig is not None:
            plt.close(fig)
        if was_interactive:
            plt.ion()

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary), fh, indent=2)


def refit_sin2psi_from_csv(
    data_dir,
    scan_number,
    excluded_frames=None,
    exclude_chi_ranges=None,
    exclude_sin2psi_ranges=None,
    auto_exclude=False,
    auto_exclude_sigma=3.0,
    auto_exclude_max_iter=1,
    correction_json=None,
    elastic_E=None,
    elastic_nu=None,
    elastic_E_units=None,
    stress_reference_two_theta=None,
    stress_reference_d0=None,
    stress_wavelength=None,
    stress_energy=None,
):
    scan_dir = Path(data_dir) / "sin2psi_export" / f"scan_{int(scan_number)}"
    csv_path = scan_dir / f"scan_{int(scan_number)}_fits.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing scan fit CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    summary = perform_sin2psi_fit(
        df,
        str(scan_dir),
        excluded_frames=excluded_frames,
        exclude_chi_ranges=exclude_chi_ranges,
        exclude_sin2psi_ranges=exclude_sin2psi_ranges,
        auto_exclude=auto_exclude,
        auto_exclude_sigma=auto_exclude_sigma,
        auto_exclude_max_iter=auto_exclude_max_iter,
        correction_json=correction_json,
        elastic_E=elastic_E,
        elastic_nu=elastic_nu,
        elastic_E_units=elastic_E_units,
        stress_reference_two_theta=stress_reference_two_theta,
        stress_reference_d0=stress_reference_d0,
        stress_wavelength=stress_wavelength,
        stress_energy=stress_energy,
    )
    df["psi_deg"] = df["chi"].apply(lambda c: 90.0 - float(c) if pd.notna(c) else np.nan)
    df["sin2psi"] = df["psi_deg"].apply(
        lambda p: math.sin(math.radians(float(p))) ** 2 if pd.notna(p) else np.nan
    )
    y_column = "peak_center"
    correction = summary.get("correction") if correction_json else None
    if correction:
        df = _apply_sin2psi_correction(df, correction)
        y_column = "peak_center_corrected"
    exclusion_mask, _ = _sin2psi_exclusion_mask(
        df,
        excluded_frames=excluded_frames,
        exclude_chi_ranges=exclude_chi_ranges,
        exclude_sin2psi_ranges=exclude_sin2psi_ranges,
    )
    if auto_exclude:
        exclusion_mask, _ = _auto_exclude_sin2psi_outliers(
            df,
            exclusion_mask,
            sigma=auto_exclude_sigma,
            max_iter=auto_exclude_max_iter,
            y_column=y_column,
        )
    df["excluded_from_sin2psi"] = exclusion_mask
    df.to_csv(csv_path, index=False)
    return {
        "scan_number": int(scan_number),
        "scan_dir": str(scan_dir),
        "csv_path": str(csv_path),
        "summary": summary,
    }


def _scan_number_from_dir(scan_dir):
    match = re.search(r"scan_(\d+)$", Path(scan_dir).name)
    return int(match.group(1)) if match else None


def _summary_dirs(data_dir, scans=None):
    export_root = Path(data_dir) / "sin2psi_export"
    if scans is not None:
        scans = [scans] if isinstance(scans, int) else scans
        return [export_root / f"scan_{int(scan)}" for scan in scans]
    return sorted(export_root.glob("scan_*"), key=lambda p: (_scan_number_from_dir(p) is None, _scan_number_from_dir(p) or p.name))


def _metadata_from_scan_csv(scan_dir, scan_number):
    csv_path = Path(scan_dir) / f"scan_{scan_number}_fits.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Could not read %s: %s", csv_path, exc)
        return {}
    return _scan_metadata_from_df(df)


def collect_sin2psi_summaries(data_dir, scans=None, save_csv=True, output_path=None):
    rows = []
    for scan_dir in _summary_dirs(data_dir, scans=scans):
        scan_number = _scan_number_from_dir(scan_dir)
        if scan_number is None:
            continue
        json_path = scan_dir / "sin2psi_fit_params.json"
        if not json_path.exists():
            logger.warning("Missing sin2psi fit JSON for scan %s: %s", scan_number, json_path)
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception as exc:
            logger.warning("Could not read %s: %s", json_path, exc)
            continue

        metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
        fallback_metadata = _metadata_from_scan_csv(scan_dir, scan_number)
        for key, value in fallback_metadata.items():
            metadata.setdefault(key, value)

        row = {"scan_number": scan_number}
        for key in [
            "slope",
            "slope_err",
            "intercept",
            "intercept_err",
            "chi2",
            "rms",
            "n_points",
            "stress",
            "stress_err",
            "stress_units",
            "elastic_E",
            "elastic_E_units",
            "elastic_nu",
            "stress_method",
            "stress_reference_two_theta",
            "stress_reference_d0",
            "stress_inferred_d0",
        ]:
            row[key] = summary.get(key)
        row.update(metadata)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("scan_number").reset_index(drop=True)
    if save_csv:
        csv_path = Path(output_path) if output_path else _summary_output_dir(data_dir) / "sin2psi_scan_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def _safe_plot_suffix(value):
    suffix = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return suffix or "x"


def _compact_scan_title(scans):
    scans = [scans] if isinstance(scans, int) else list(scans)
    if not scans:
        return "no scans"
    if len(scans) == 1:
        return f"scan {int(scans[0])}"
    sorted_scans = sorted({int(scan) for scan in scans})
    if sorted_scans == list(range(sorted_scans[0], sorted_scans[-1] + 1)):
        return f"scans {sorted_scans[0]}-{sorted_scans[-1]}"
    if len(sorted_scans) <= 8:
        return "scans " + ", ".join(str(scan) for scan in sorted_scans)
    return f"scans {sorted_scans[0]}-{sorted_scans[-1]}"


def _scan_title(scans, plotted_scans=None):
    if scans is None:
        if plotted_scans is None:
            return "all scans"
        values = pd.to_numeric(pd.Series(plotted_scans), errors="coerce").dropna()
        if values.empty:
            return "all scans"
        return _compact_scan_title(values.astype(int).tolist())
    return _compact_scan_title(scans)


def _selector_title(selector):
    text = str(selector)
    if text.startswith("frame_"):
        return f"frame {text.split('_', 1)[1]}"
    if text.startswith("chi_"):
        return f"chi={text.split('_', 1)[1].replace('_', '.')}\u00b0"
    return text.replace("_", " ")


def _output_timestamp():
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def _sin2psi_export_root(data_dir):
    return Path(data_dir) / "sin2psi_export"


def _summary_output_dir(data_dir):
    return _sin2psi_export_root(data_dir) / "summaries"


def _plot_output_dir(data_dir):
    return _sin2psi_export_root(data_dir) / "plots"


def _unique_path(path):
    path = Path(path)
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{idx:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique output path for {path}")


def _latest_matching_file(root, pattern):
    files = [path for path in Path(root).glob(pattern) if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: (path.stat().st_mtime, path.name))


def _matching_files_newest_first(root, pattern):
    files = [path for path in Path(root).glob(pattern) if path.is_file()]
    return sorted(files, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def _normalised_compare_df(df):
    out = df.copy()
    if "scan_number" in out.columns:
        out = out.sort_values("scan_number")
    out = out.reindex(sorted(out.columns), axis=1).reset_index(drop=True)
    out = out.where(pd.notna(out), np.nan)
    return out


def _dataframes_match(left, right):
    try:
        pd.testing.assert_frame_equal(
            _normalised_compare_df(left),
            _normalised_compare_df(right),
            check_dtype=False,
            check_like=True,
            atol=1e-9,
            rtol=1e-9,
        )
        return True
    except AssertionError:
        return False


def _write_or_reuse_summary(df, output_dir, filename_template, legacy_dirs=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_dirs = [output_dir]
    for legacy_dir in legacy_dirs or []:
        legacy_dir = Path(legacy_dir)
        if legacy_dir not in search_dirs:
            search_dirs.append(legacy_dir)
    existing_files = []
    for directory in search_dirs:
        if directory.exists():
            existing_files.extend(_matching_files_newest_first(directory, filename_template.format(timestamp="*")))
    existing_files = sorted(existing_files, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    for existing in existing_files:
        try:
            if _dataframes_match(df, pd.read_csv(existing)):
                return existing
        except Exception as exc:
            logger.warning("Could not compare existing summary %s: %s", existing, exc)
    output_path = _unique_path(output_dir / filename_template.format(timestamp=_output_timestamp()))
    df.to_csv(output_path, index=False)
    return output_path


def build_processing_params(**params):
    return _json_safe(dict(params))


def load_processing_params(params_json):
    with open(params_json, "r", encoding="utf-8") as fh:
        params = json.load(fh)
    if not isinstance(params, dict):
        raise ValueError(f"Processing params must be a JSON object: {params_json}")
    return params


def save_processing_params(params, data_dir, timestamp=None):
    export_root = Path(data_dir) / "sin2psi_export"
    logs_dir = export_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or _output_timestamp()
    output_path = _unique_path(logs_dir / f"sin2psi_processing_params_{stamp}.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(params), fh, indent=2)
    return str(output_path)


def _prepare_plot_frame(df, x, y):
    if df.empty:
        raise RuntimeError("No summary rows found")
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not found in summary")
    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        raise RuntimeError(f"No usable points for {y} vs {x}")

    x_values = plot_df[x]
    x_label = x
    if "time" in str(x).lower():
        parsed = pd.to_datetime(x_values, errors="coerce")
        if parsed.notna().any():
            plot_df = plot_df.loc[parsed.notna()].copy()
            x_values = parsed.loc[parsed.notna()]
            x_label = str(x).replace("_", " ")
    return plot_df, x_values, x_label


def _load_sin2psi_summary_csv(summary_csv, scans=None):
    summary_path = Path(summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    df = pd.read_csv(summary_path)
    if scans is not None and "scan_number" in df.columns:
        scan_values = [int(scan) for scan in ([scans] if isinstance(scans, int) else scans)]
        scan_series = pd.to_numeric(df["scan_number"], errors="coerce")
        df = df.loc[scan_series.isin(scan_values)].copy()
    if not df.empty and "scan_number" in df.columns:
        df = df.sort_values("scan_number").reset_index(drop=True)
    return df, summary_path


def _set_ylim_from_points(ax, values, pad_fraction=0.08):
    y_values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size == 0:
        return
    ymin = float(np.min(y_values))
    ymax = float(np.max(y_values))
    if ymin == ymax:
        pad = max(abs(ymin) * pad_fraction, 1.0)
    else:
        pad = (ymax - ymin) * pad_fraction
    ax.set_ylim(ymin - pad, ymax + pad)


def plot_sin2psi_gradients(data_dir, x="scan_number", scans=None, save=True, show=False, summary_csv=None):
    """
    Plot sin2psi gradient (slope) vs a specified x-axis variable (default: scan_number).
    args:
        data_dir: directory containing the scan data
        x: column name to use for the x-axis (default: "scan_number"). Options include "scan_number", "temperature", "energy", "start_time", etc.
        scans: list of scan numbers to include (default: all scans)
        save: whether to save the plot and summary (default: True)
        show: whether to display the plot (default: False)
        summary_csv: optional existing summary CSV to plot instead of collecting current JSON outputs
    """
    if summary_csv:
        df, selected_summary_path = _load_sin2psi_summary_csv(summary_csv, scans=scans)
    else:
        df = collect_sin2psi_summaries(data_dir, scans=scans, save_csv=False)
        selected_summary_path = None
    plot_df, x_values, x_label = _prepare_plot_frame(df, x, "slope")

    yerr = None
    if "slope_err" in plot_df.columns:
        slope_err = pd.to_numeric(plot_df["slope_err"], errors="coerce")
        if slope_err.notna().any():
            yerr = slope_err.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        x_values,
        pd.to_numeric(plot_df["slope"], errors="coerce"),
        yerr=yerr,
        fmt=".-",
        capsize=3,
        linewidth=0.8,
        markersize=4,
    )
    _set_ylim_from_points(ax, plot_df["slope"])
    ax.set_xlabel(str(x_label).replace("_", " "))
    ax.set_ylabel("sin2psi gradient (slope)")
    ax.set_title(
        f"sin2psi gradient vs {str(x_label).replace('_', ' ')} - "
        f"{_scan_title(scans, plot_df.get('scan_number'))}"
    )
    ax.grid(True, linewidth=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = None
    summary_path = selected_summary_path
    if save:
        export_root = _sin2psi_export_root(data_dir)
        summary_dir = _summary_output_dir(data_dir)
        plot_dir = _plot_output_dir(data_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        x_suffix = _safe_plot_suffix(x)
        stamp = _output_timestamp()
        if summary_path is None:
            summary_path = _write_or_reuse_summary(
                df,
                summary_dir,
                "sin2psi_scan_summary_{timestamp}.csv",
                legacy_dirs=[export_root],
            )
        output_path = _unique_path(plot_dir / f"sin2psi_gradient_vs_{x_suffix}_{stamp}.png")
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "summary": df,
        "summary_path": str(summary_path) if summary_path else None,
        "plot_path": str(output_path) if output_path else None,
    }


def plot_sin2psi_stress(data_dir, x="scan_number", scans=None, save=True, show=False, summary_csv=None):
    if summary_csv:
        df, selected_summary_path = _load_sin2psi_summary_csv(summary_csv, scans=scans)
    else:
        df = collect_sin2psi_summaries(data_dir, scans=scans, save_csv=False)
        selected_summary_path = None
    plot_df, x_values, x_label = _prepare_plot_frame(df, x, "stress")
    yerr = None
    if "stress_err" in plot_df.columns:
        stress_err = pd.to_numeric(plot_df["stress_err"], errors="coerce")
        if stress_err.notna().any():
            yerr = stress_err.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        x_values,
        pd.to_numeric(plot_df["stress"], errors="coerce"),
        yerr=yerr,
        fmt=".-",
        capsize=3,
        linewidth=0.8,
        markersize=4,
    )
    _set_ylim_from_points(ax, plot_df["stress"])
    ax.set_xlabel(str(x_label).replace("_", " "))
    units = ""
    if "stress_units" in plot_df.columns and plot_df["stress_units"].notna().any():
        units = f" ({plot_df['stress_units'].dropna().iloc[0]})"
    ax.set_ylabel(f"stress{units}")
    ax.set_title(
        f"sin2psi stress vs {str(x_label).replace('_', ' ')} - "
        f"{_scan_title(scans, plot_df.get('scan_number'))}"
    )
    ax.grid(True, linewidth=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = None
    summary_path = selected_summary_path
    if save:
        export_root = _sin2psi_export_root(data_dir)
        summary_dir = _summary_output_dir(data_dir)
        plot_dir = _plot_output_dir(data_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        stamp = _output_timestamp()
        x_suffix = _safe_plot_suffix(x)
        if summary_path is None:
            summary_path = _write_or_reuse_summary(
                df,
                summary_dir,
                "sin2psi_scan_summary_{timestamp}.csv",
                legacy_dirs=[export_root],
            )
        output_path = _unique_path(plot_dir / f"sin2psi_stress_vs_{x_suffix}_{stamp}.png")
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "summary": df,
        "summary_path": str(summary_path) if summary_path else None,
        "plot_path": str(output_path) if output_path else None,
    }


def _scan_fit_csv_path(scan_dir, scan_number):
    return Path(scan_dir) / f"scan_{scan_number}_fits.csv"


def _selector_values(value):
    if isinstance(value, (str, bytes)):
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _selected_fit_rows(df, scan_number, frame_index=None, chi=None, label="fit"):
    if (frame_index is None) == (chi is None):
        raise ValueError("Specify exactly one of frame_index or chi")
    matches = []
    if frame_index is not None:
        frame_values = pd.to_numeric(df["frame_index"], errors="coerce")
        for frame_value in _selector_values(frame_index):
            selected = df.loc[frame_values == int(frame_value)]
            selector = f"frame_{int(frame_value)}"
            if selected.empty:
                logger.warning("No %s row matched %s for scan %s", label, selector, scan_number)
                continue
            matches.append((selected.sort_values("frame_index").iloc[0], selector))
    else:
        chi_values = pd.to_numeric(df["chi"], errors="coerce")
        for chi_value in _selector_values(chi):
            selected = df.loc[(chi_values - float(chi_value)).abs() <= 0.1]
            selector = f"chi_{_safe_plot_suffix(chi_value)}"
            if selected.empty:
                logger.warning("No %s row matched %s for scan %s", label, selector, scan_number)
                continue
            matches.append((selected.sort_values("frame_index").iloc[0], selector))
    return matches


def _selected_fwhm_row(df, scan_number, frame_index=None, chi=None):
    matches = _selected_fit_rows(df, scan_number, frame_index=frame_index, chi=chi, label="FWHM")
    if not matches:
        selector = (
            f"frame_{_safe_plot_suffix(frame_index)}"
            if frame_index is not None
            else f"chi_{_safe_plot_suffix(chi)}"
        )
        return None, selector
    return matches[0]


def collect_fwhm_summaries(data_dir, scans=None, frame_index=None, chi=None):
    return collect_fit_value_summaries(
        data_dir,
        scans=scans,
        frame_index=frame_index,
        chi=chi,
        value_column="fwhm",
        error_column="fwhm_err",
        label="FWHM",
    )


def collect_peak_position_summaries(data_dir, scans=None, frame_index=None, chi=None):
    return collect_fit_value_summaries(
        data_dir,
        scans=scans,
        frame_index=frame_index,
        chi=chi,
        value_column="peak_center",
        error_column="peak_center_err",
        label="peak position",
    )


def collect_fit_value_summaries(
    data_dir,
    scans=None,
    frame_index=None,
    chi=None,
    value_column="fwhm",
    error_column=None,
    label="fit",
):
    rows = []
    for scan_dir in _summary_dirs(data_dir, scans=scans):
        scan_number = _scan_number_from_dir(scan_dir)
        if scan_number is None:
            continue
        csv_path = _scan_fit_csv_path(scan_dir, scan_number)
        if not csv_path.exists():
            logger.warning("Missing scan fit CSV for scan %s: %s", scan_number, csv_path)
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", csv_path, exc)
            continue
        if value_column not in df.columns:
            logger.warning("Missing %s column for scan %s: %s", value_column, scan_number, csv_path)
            continue
        for row, selector in _selected_fit_rows(df, scan_number, frame_index=frame_index, chi=chi, label=label):
            out = {"scan_number": scan_number, "selector": selector, "trend_value": row.get(value_column)}
            if error_column and error_column in df.columns:
                out["trend_err"] = row.get(error_column)
            for key in [
                "frame_index",
                "filename",
                "scan_type",
                "chi",
                "psi_deg",
                "sin2psi",
                "temperature",
                "energy",
                "start_time",
                "frame_time",
                "fwhm",
                "fwhm_err",
                "peak_center",
                "peak_center_err",
                "fit_success",
            ]:
                if key in df.columns:
                    out[key] = row.get(key)
            if "metadata_json" in df.columns and pd.notna(row.get("metadata_json")):
                try:
                    extra = json.loads(row.get("metadata_json"))
                    if isinstance(extra, dict):
                        out.update(extra)
                except Exception:
                    logger.warning("Could not parse metadata_json for %s summary", label)
            rows.append(_json_safe(out))

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["selector", "scan_number"]).reset_index(drop=True)
    return result


def _plot_fit_value_trends(
    df,
    data_dir,
    x,
    y_column,
    yerr_column,
    ylabel,
    title_label,
    file_prefix,
    scans=None,
    save=True,
    show=False,
):
    plot_df, _x_values, x_label = _prepare_plot_frame(df, x, y_column)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    selectors = list(plot_df["selector"].dropna().unique()) if "selector" in plot_df.columns else ["selected"]
    for selector in selectors:
        series = plot_df.loc[plot_df["selector"] == selector].copy() if "selector" in plot_df.columns else plot_df
        series, x_values, _series_x_label = _prepare_plot_frame(series, x, y_column)
        yerr = None
        if yerr_column and yerr_column in series.columns:
            err_values = pd.to_numeric(series[yerr_column], errors="coerce")
            if err_values.notna().any():
                yerr = err_values.to_numpy(dtype=float)
        ax.errorbar(
            x_values,
            pd.to_numeric(series[y_column], errors="coerce"),
            yerr=yerr,
            fmt=".-",
            capsize=3,
            linewidth=0.8,
            markersize=4,
            label=_selector_title(selector),
        )
    _set_ylim_from_points(ax, plot_df[y_column])
    ax.set_xlabel(str(x_label).replace("_", " "))
    ax.set_ylabel(ylabel)
    selector = "multi" if len(selectors) > 1 else str(selectors[0])
    selector_title = "multiple chi values" if selector == "multi" else _selector_title(selector)
    ax.set_title(
        f"{title_label} {selector_title} vs {str(x_label).replace('_', ' ')} - "
        f"{_scan_title(scans, plot_df.get('scan_number'))}"
    )
    ax.grid(True, linewidth=0.3)
    if len(selectors) > 1:
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = None
    summary_path = None
    if save:
        summary_dir = _summary_output_dir(data_dir)
        plot_dir = _plot_output_dir(data_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        stamp = _output_timestamp()
        selector_suffix = _safe_plot_suffix(selector)
        x_suffix = _safe_plot_suffix(x)
        summary_path = _unique_path(summary_dir / f"sin2psi_{file_prefix}_summary_{stamp}.csv")
        output_path = _unique_path(plot_dir / f"sin2psi_{file_prefix}_vs_{x_suffix}_{selector_suffix}_{stamp}.png")
        df.to_csv(summary_path, index=False)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {
        "summary": df,
        "summary_path": str(summary_path) if summary_path else None,
        "plot_path": str(output_path) if output_path else None,
    }


def plot_fwhm_trends(data_dir, x="scan_number", scans=None, frame_index=None, chi=None, save=True, show=False):
    df = collect_fwhm_summaries(data_dir, scans=scans, frame_index=frame_index, chi=chi)
    return _plot_fit_value_trends(
        df,
        data_dir,
        x,
        y_column="fwhm",
        yerr_column="fwhm_err",
        ylabel="FWHM",
        title_label="FWHM",
        file_prefix="fwhm",
        scans=scans,
        save=save,
        show=show,
    )


def plot_peak_position_trends(data_dir, x="scan_number", scans=None, frame_index=None, chi=None, save=True, show=False):
    df = collect_peak_position_summaries(data_dir, scans=scans, frame_index=frame_index, chi=chi)
    return _plot_fit_value_trends(
        df,
        data_dir,
        x,
        y_column="peak_center",
        yerr_column="peak_center_err",
        ylabel="Peak position (2theta)",
        title_label="Peak position",
        file_prefix="peak_position",
        scans=scans,
        save=save,
        show=show,
    )


def process_scan(
    data_dir,
    scan_number,
    files=None,
    exclude_frames=None,
    exclude_chi_ranges=None,
    exclude_sin2psi_ranges=None,
    auto_exclude=False,
    auto_exclude_sigma=3.0,
    auto_exclude_max_iter=1,
    correction_json=None,
    plot_frames=True,
    force=True,
    backup=False,
    peak_center=None,
    track_peak=True,
    track_window=0.4,
    fallback_to_auto=True,
    elastic_E=None,
    elastic_nu=None,
    elastic_E_units=None,
    stress_reference_two_theta=None,
    stress_reference_d0=None,
    stress_wavelength=None,
    stress_energy=None,
):
    """
    Process a single scan: fit each frame, save per-frame results, and perform sin2psi regression.
    args:
        data_dir: str, path to the directory containing scan files
        scan_number: int, the scan number to process
        files: Optional[List[str]], list of file paths to process; if None, discover files
        exclude_frames: Optional[List[int]], list of frame indices to exclude from sin2psi fit
        exclude_chi_ranges: Optional[List[Tuple[float, float]]], chi ranges excluded from sin2psi fit
        exclude_sin2psi_ranges: Optional[List[Tuple[float, float]]], sin2psi ranges excluded from sin2psi fit
        auto_exclude: bool, whether to exclude large residual outliers from the sin2psi fit
        correction_json: Optional[str], sin2psi correction JSON generated from a stress-free reference scan
        plot_frames: bool, whether to save per-frame fit plots
        force: bool, whether to overwrite existing outputs
        backup: bool, whether to backup existing scan output directory before overwrite
        peak_center: Optional[float], first peak center guess in 2theta degrees
        track_peak: bool, whether to seed each frame from the previous successful fit
        track_window: float, half-width in 2theta degrees around a seeded center
        fallback_to_auto: bool, whether seeded fits retry with automatic peak detection
        elastic_E: Optional[float], Young's modulus for stress calculation
        elastic_E_units: Optional[str], label for E/stress units
        elastic_nu: Optional[float], Poisson ratio for stress calculation
    returns: dict with keys:
        'scan_number': int, the scan number processed
        'scan_dir': str, path to the scan output directory
        'csv_path': str, path to the CSV file with per-frame results
        'frame_count': int, number of frames processed
        'summary': dict, summary of the sin2psi fit results
    """
    data_dir_path = Path(data_dir)
    scan_dir = data_dir_path / "sin2psi_export" / f"scan_{scan_number}"
    _ensure_clean_scan_dir(scan_dir, backup=backup if force else False)

    for stale in scan_dir.glob("sin2psi_plot.png"):
        stale.unlink(missing_ok=True)
    for stale in scan_dir.glob(f"scan_{scan_number}_sin2psi_plot.png"):
        stale.unlink(missing_ok=True)
    for stale in scan_dir.glob("sin2psi_fit_params.json"):
        stale.unlink(missing_ok=True)
    for stale in scan_dir.glob(f"scan_{scan_number}_fits.csv"):
        stale.unlink(missing_ok=True)
    for stale in scan_dir.glob("frame_*_fit.png"):
        stale.unlink(missing_ok=True)
    old_frames_dir = scan_dir / "frames"
    for stale in old_frames_dir.glob("frame_*_fit.png"):
        stale.unlink(missing_ok=True)
    if old_frames_dir.exists():
        try:
            old_frames_dir.rmdir()
        except OSError:
            logger.warning("Leaving non-empty legacy frames directory: %s", old_frames_dir)

    if files is None:
        files = discover_scan_files(data_dir, scan_number)
    if not files:
        raise RuntimeError(f"No matching scan files found for scan {scan_number}")

    rows = []
    current_seed = float(peak_center) if peak_center is not None else None
    for idx, filepath in enumerate(files):
        parsed = parse_txt_scan(filepath)
        seed_center = current_seed if (idx == 0 or track_peak) else None
        try:
            fit_result = fit_frame(
                parsed["tth"],
                parsed["intensity"],
                plot=plot_frames,
                plot_path=str(scan_dir / f"frame_{idx:03d}_fit.png") if plot_frames else None,
                chi=parsed.get("chi"),
                seed_center=seed_center,
                track_window=track_window,
                fallback_to_auto=fallback_to_auto,
            )
            fit_success = True
            if track_peak and np.isfinite(fit_result["center"]):
                current_seed = fit_result["center"]
        except Exception as exc:
            logger.warning("Fit failed for %s: %s", filepath, exc)
            if plot_frames:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(parsed["tth"], parsed["intensity"], "b-", label="data")
                ax.set_xlabel("2theta")
                ax.set_ylabel("Intensity")
                psi_text = ""
                chi = parsed.get("chi")
                if chi is not None and pd.notna(chi):
                    psi_text = f", psi={90.0 - float(chi):.3f}"
                ax.set_title(f"scan {scan_number} frame {idx:03d}{psi_text} - fit failed")
                ax.legend()
                fig.tight_layout()
                fig.savefig(scan_dir / f"frame_{idx:03d}_fit.png", dpi=150)
                plt.close(fig)
            fit_result = {
                "center": np.nan,
                "center_err": np.nan,
                "amplitude": np.nan,
                "amplitude_err": np.nan,
                "fwhm": np.nan,
                "fwhm_err": np.nan,
                "nu": np.nan,
                "nu_err": np.nan,
                "background_mean": np.nan,
                "background_median": np.nan,
                "bg_coef_0": np.nan,
                "bg_coef_1": np.nan,
                "bg_coef_2": np.nan,
                "left_idx": np.nan,
                "right_idx": np.nan,
                "peak_idx": np.nan,
                "window_mode": "failed",
                "seed_center": float(seed_center) if seed_center is not None else np.nan,
                "background_lower": np.nan,
                "peak_lower": np.nan,
                "peak_upper": np.nan,
                "background_upper": np.nan,
                "fit_success": False,
                "x_fit": [],
                "y_peak_fit": [],
                "y_bg_fit": [],
                "y_combined_fit": [],
            }
            fit_success = False

        chi = parsed.get("chi")
        psi = 90.0 - float(chi) if pd.notna(chi) else np.nan
        sin2psi = math.sin(math.radians(psi)) ** 2 if pd.notna(psi) else np.nan

        rows.append(
            {
                "frame_index": idx,
                "filename": parsed["filename"],
                "scan_type": parsed.get("scan_type"),
                "chi": chi,
                "psi_deg": psi,
                "sin2psi": sin2psi,
                "temperature": parsed.get("temperature"),
                "energy": parsed.get("energy"),
                "start_time": parsed.get("start_time"),
                "frame_time": parsed.get("frame_time"),
                "metadata_json": json.dumps(_json_safe(parsed.get("metadata", {}))),
                "peak_center": fit_result["center"],
                "peak_center_err": fit_result["center_err"],
                "amplitude": fit_result["amplitude"],
                "amplitude_err": fit_result["amplitude_err"],
                "fwhm": fit_result["fwhm"],
                "fwhm_err": fit_result["fwhm_err"],
                "nu": fit_result["nu"],
                "nu_err": fit_result["nu_err"],
                "background_mean": fit_result["background_mean"],
                "background_median": fit_result["background_median"],
                "bg_coef_0": fit_result["bg_coef_0"],
                "bg_coef_1": fit_result["bg_coef_1"],
                "bg_coef_2": fit_result["bg_coef_2"],
                "left_idx": fit_result["left_idx"],
                "right_idx": fit_result["right_idx"],
                "peak_idx": fit_result["peak_idx"],
                "window_mode": fit_result["window_mode"],
                "seed_center": fit_result["seed_center"],
                "background_lower": fit_result["background_lower"],
                "peak_lower": fit_result["peak_lower"],
                "peak_upper": fit_result["peak_upper"],
                "background_upper": fit_result["background_upper"],
                "fit_success": fit_success,
                "excluded_from_sin2psi": False,
            }
        )

    df = pd.DataFrame(rows, columns=_frame_csv_columns())
    exclude_frames = sorted(set(int(x) for x in (exclude_frames or [])))
    exclusion_mask, _ = _sin2psi_exclusion_mask(
        df,
        excluded_frames=exclude_frames,
        exclude_chi_ranges=exclude_chi_ranges,
        exclude_sin2psi_ranges=exclude_sin2psi_ranges,
    )
    df["excluded_from_sin2psi"] = exclusion_mask

    csv_path = scan_dir / f"scan_{scan_number}_fits.csv"
    df.to_csv(csv_path, index=False)

    summary = perform_sin2psi_fit(
        df,
        str(scan_dir),
        excluded_frames=exclude_frames,
        exclude_chi_ranges=exclude_chi_ranges,
        exclude_sin2psi_ranges=exclude_sin2psi_ranges,
        auto_exclude=auto_exclude,
        auto_exclude_sigma=auto_exclude_sigma,
        auto_exclude_max_iter=auto_exclude_max_iter,
        correction_json=correction_json,
        elastic_E=elastic_E,
        elastic_nu=elastic_nu,
        elastic_E_units=elastic_E_units,
        stress_reference_two_theta=stress_reference_two_theta,
        stress_reference_d0=stress_reference_d0,
        stress_wavelength=stress_wavelength,
        stress_energy=stress_energy,
    )

    df["psi_deg"] = df["chi"].apply(lambda c: 90.0 - float(c) if pd.notna(c) else np.nan)
    df["sin2psi"] = df["psi_deg"].apply(
        lambda p: math.sin(math.radians(float(p))) ** 2 if pd.notna(p) else np.nan
    )
    correction = summary.get("correction") if correction_json else None
    if correction:
        df = _apply_sin2psi_correction(df, correction)
    exclusion_mask, _ = _sin2psi_exclusion_mask(
        df,
        excluded_frames=exclude_frames,
        exclude_chi_ranges=exclude_chi_ranges,
        exclude_sin2psi_ranges=exclude_sin2psi_ranges,
    )
    if auto_exclude:
        exclusion_mask, _ = _auto_exclude_sin2psi_outliers(
            df,
            exclusion_mask,
            sigma=auto_exclude_sigma,
            max_iter=auto_exclude_max_iter,
            y_column="peak_center_corrected" if correction else "peak_center",
        )
    df["excluded_from_sin2psi"] = exclusion_mask
    df.to_csv(csv_path, index=False)

    return {
        "scan_number": scan_number,
        "scan_dir": str(scan_dir),
        "csv_path": str(csv_path),
        "frame_count": int(len(df)),
        "summary": summary,
    }
