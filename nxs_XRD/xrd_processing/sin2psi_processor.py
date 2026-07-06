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
    for key in ["coefficients", "reference_two_theta"]:
        if key not in correction:
            raise ValueError(f"Missing '{key}' in sin2psi correction: {correction_json}")
    return correction


def _apply_sin2psi_correction(df, correction):
    corrected = df.copy()
    coeffs = [float(value) for value in correction["coefficients"]]
    reference_two_theta = float(correction["reference_two_theta"])
    raw_correction = np.polyval(coeffs, pd.to_numeric(corrected["sin2psi"], errors="coerce"))
    scale = _theta_scale(corrected["peak_center"], reference_two_theta)
    corrected["sin2psi_correction_reference"] = raw_correction
    corrected["sin2psi_correction_scale"] = scale
    corrected["sin2psi_correction"] = raw_correction * scale
    corrected["peak_center_uncorrected"] = corrected["peak_center"]
    corrected["peak_center_corrected"] = corrected["peak_center"] - corrected["sin2psi_correction"]
    return corrected


def generate_sin2psi_correction(
    data_dir,
    scan_number,
    degree=2,
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
    reference_two_theta = float(np.average(used["peak_center"].to_numpy(dtype=float), weights=weights))
    offsets = used["peak_center"].to_numpy(dtype=float) - reference_two_theta
    coefficients = np.polyfit(
        used["sin2psi"].to_numpy(dtype=float),
        offsets,
        int(degree),
        w=weights,
    )

    correction = {
        "type": "sin2psi_chi_polynomial_correction",
        "created_at": _output_timestamp(),
        "source_scan": int(scan_number),
        "source_csv": str(csv_path),
        "x": "sin2psi",
        "y": "peak_center_offset_from_reference_two_theta",
        "degree": int(degree),
        "coefficients": [float(value) for value in coefficients],
        "reference_two_theta": reference_two_theta,
        "n_points": int(len(used)),
        "excluded_frames": excluded_frames,
        "exclude_chi_ranges": _normalise_ranges(exclude_chi_ranges),
        "exclude_sin2psi_ranges": _normalise_ranges(exclude_sin2psi_ranges),
    }

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
    y_fit = np.polyval(coefficients, x_fit)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_yerr = pd.to_numeric(used.get("peak_center_err"), errors="coerce")
    if plot_yerr.notna().any():
        ax.errorbar(
            used["sin2psi"],
            offsets,
            yerr=plot_yerr.to_numpy(dtype=float),
            fmt="x",
            capsize=3,
            label="reference points",
        )
    else:
        ax.plot(used["sin2psi"], offsets, "x", label="reference points")
    excluded = df.loc[exclusion_mask].dropna(subset=["sin2psi", "peak_center"])
    if not excluded.empty:
        excluded_offsets = excluded["peak_center"].to_numpy(dtype=float) - reference_two_theta
        excluded_yerr = pd.to_numeric(excluded.get("peak_center_err"), errors="coerce")
        if excluded_yerr.notna().any():
            ax.errorbar(
                excluded["sin2psi"],
                excluded_offsets,
                yerr=excluded_yerr.to_numpy(dtype=float),
                fmt=".",
                capsize=3,
                label="excluded",
            )
        else:
            ax.plot(excluded["sin2psi"], excluded_offsets, ".", label="excluded")
    ax.plot(x_fit, y_fit, "-", linewidth=0.8, label=f"degree {int(degree)} fit")
    ax.set_xlabel("sin2psi")
    ax.set_ylabel("2theta offset from reference")
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
):
    scan_path = Path(scan_dir)
    df = df.copy()
    df["psi_deg"] = df["chi"].apply(lambda c: 90.0 - float(c) if pd.notna(c) else np.nan)
    df["sin2psi"] = df["psi_deg"].apply(
        lambda p: math.sin(math.radians(float(p))) ** 2 if pd.notna(p) else np.nan
    )
    correction = load_sin2psi_correction(correction_json) if correction_json else None
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
        "correction_file": str(correction_json) if correction_json else None,
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

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if summary.get("correction_applied") and "peak_center_uncorrected" in df.columns:
        ax.plot(
            df["sin2psi"],
            df["peak_center_uncorrected"],
            "x",
            alpha=0.2,
            label="uncorrected",
        )
    ax.plot(used["sin2psi"], used[y_column], "x", label="used")
    if not excluded.empty:
        ax.plot(excluded["sin2psi"], excluded[y_column], ".", label="excluded")

    xline = np.linspace(float(df["sin2psi"].min()), float(df["sin2psi"].max()), 200)
    yline = summary["slope"] * xline + summary["intercept"]
    ax.plot(xline, yline, "-", label="fit", linewidth=0.5, color="black")
    ax.set_xlabel("sin2psi")
    ax.set_ylabel(y_column)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

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
    )
    df["psi_deg"] = df["chi"].apply(lambda c: 90.0 - float(c) if pd.notna(c) else np.nan)
    df["sin2psi"] = df["psi_deg"].apply(
        lambda p: math.sin(math.radians(float(p))) ** 2 if pd.notna(p) else np.nan
    )
    y_column = "peak_center"
    if correction_json:
        df = _apply_sin2psi_correction(df, load_sin2psi_correction(correction_json))
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
        ]:
            row[key] = summary.get(key)
        row.update(metadata)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("scan_number").reset_index(drop=True)
    if save_csv:
        export_root = Path(data_dir) / "sin2psi_export"
        export_root.mkdir(parents=True, exist_ok=True)
        csv_path = Path(output_path) if output_path else export_root / "sin2psi_scan_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def _safe_plot_suffix(value):
    suffix = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return suffix or "x"


def _output_timestamp():
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


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


def _write_or_reuse_summary(df, export_root, filename_template):
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    latest = _latest_matching_file(export_root, filename_template.format(timestamp="*"))
    if latest is not None:
        try:
            if _dataframes_match(df, pd.read_csv(latest)):
                return latest
        except Exception as exc:
            logger.warning("Could not compare existing summary %s: %s", latest, exc)
    output_path = _unique_path(export_root / filename_template.format(timestamp=_output_timestamp()))
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


def plot_sin2psi_gradients(data_dir, x="scan_number", scans=None, save=True, show=False):
    """
    Plot sin2psi gradient (slope) vs a specified x-axis variable (default: scan_number).
    args:
        data_dir: directory containing the scan data
        x: column name to use for the x-axis (default: "scan_number"). Options include "scan_number", "temperature", "energy", "start_time", etc.
        scans: list of scan numbers to include (default: all scans)
        save: whether to save the plot and summary (default: True)
        show: whether to display the plot (default: False)
    """
    df = collect_sin2psi_summaries(data_dir, scans=scans, save_csv=False)
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
        fmt="o-",
        capsize=3,
        linewidth=0.8,
        markersize=4,
    )
    ax.set_xlabel(str(x_label).replace("_", " "))
    ax.set_ylabel("sin2psi gradient (slope)")
    ax.grid(True, linewidth=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = None
    summary_path = None
    if save:
        export_root = Path(data_dir) / "sin2psi_export"
        export_root.mkdir(parents=True, exist_ok=True)
        x_suffix = _safe_plot_suffix(x)
        stamp = _output_timestamp()
        summary_path = _write_or_reuse_summary(df, export_root, "sin2psi_scan_summary_{timestamp}.csv")
        output_path = _unique_path(export_root / f"sin2psi_gradient_vs_{x_suffix}_{stamp}.png")
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


def _selected_fwhm_row(df, scan_number, frame_index=None, chi=None):
    if (frame_index is None) == (chi is None):
        raise ValueError("Specify exactly one of frame_index or chi")
    if frame_index is not None:
        selected = df.loc[pd.to_numeric(df["frame_index"], errors="coerce") == int(frame_index)]
        selector = f"frame_{int(frame_index)}"
    else:
        selected = df.loc[pd.to_numeric(df["chi"], errors="coerce") == float(chi)]
        selector = f"chi_{_safe_plot_suffix(chi)}"
    if selected.empty:
        logger.warning("No FWHM row matched %s for scan %s", selector, scan_number)
        return None, selector
    return selected.sort_values("frame_index").iloc[0], selector


def collect_fwhm_summaries(data_dir, scans=None, frame_index=None, chi=None):
    rows = []
    selector = None
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
        row, selector = _selected_fwhm_row(df, scan_number, frame_index=frame_index, chi=chi)
        if row is None:
            continue
        out = {"scan_number": scan_number, "selector": selector}
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
                logger.warning("Could not parse metadata_json for FWHM summary")
        rows.append(_json_safe(out))

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("scan_number").reset_index(drop=True)
    return result


def plot_fwhm_trends(data_dir, x="scan_number", scans=None, frame_index=None, chi=None, save=True, show=False):
    df = collect_fwhm_summaries(data_dir, scans=scans, frame_index=frame_index, chi=chi)
    plot_df, x_values, x_label = _prepare_plot_frame(df, x, "fwhm")

    yerr = None
    if "fwhm_err" in plot_df.columns:
        fwhm_err = pd.to_numeric(plot_df["fwhm_err"], errors="coerce")
        if fwhm_err.notna().any():
            yerr = fwhm_err.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        x_values,
        pd.to_numeric(plot_df["fwhm"], errors="coerce"),
        yerr=yerr,
        fmt="o-",
        capsize=3,
        linewidth=0.8,
        markersize=4,
    )
    ax.set_xlabel(str(x_label).replace("_", " "))
    ax.set_ylabel("FWHM")
    ax.grid(True, linewidth=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = None
    summary_path = None
    if save:
        export_root = Path(data_dir) / "sin2psi_export"
        export_root.mkdir(parents=True, exist_ok=True)
        stamp = _output_timestamp()
        selector = str(df["selector"].iloc[0]) if "selector" in df.columns and not df.empty else "selected"
        selector_suffix = _safe_plot_suffix(selector)
        x_suffix = _safe_plot_suffix(x)
        summary_path = _unique_path(export_root / f"sin2psi_fwhm_summary_{stamp}.csv")
        output_path = _unique_path(export_root / f"sin2psi_fwhm_vs_{x_suffix}_{selector_suffix}_{stamp}.png")
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
    )

    df["psi_deg"] = df["chi"].apply(lambda c: 90.0 - float(c) if pd.notna(c) else np.nan)
    df["sin2psi"] = df["psi_deg"].apply(
        lambda p: math.sin(math.radians(float(p))) ** 2 if pd.notna(p) else np.nan
    )
    if correction_json:
        df = _apply_sin2psi_correction(df, load_sin2psi_correction(correction_json))
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
            y_column="peak_center_corrected" if correction_json else "peak_center",
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
