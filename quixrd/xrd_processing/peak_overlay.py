from __future__ import annotations

import math
import re
import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np


ENERGY_TO_WAVELENGTH_KEV_A = 12.3984193


@dataclass(frozen=True)
class PredictedPeak:
    two_theta: float
    label: str = ""
    intensity: float = 1.0
    hkl: tuple[int, int, int] | None = None


def energy_to_wavelength(energy):
    if energy in (None, ""):
        return None
    energy = float(energy)
    if energy <= 0:
        raise ValueError("Energy must be positive")
    if energy > 1000:
        energy = energy / 1000.0
    return ENERGY_TO_WAVELENGTH_KEV_A / energy


def parse_two_theta_peaks(text):
    peaks = []
    for idx, part in enumerate(re.split(r"[,;\n]+", str(text or ""))):
        part = part.strip()
        if not part:
            continue
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", part)
        if not match:
            raise ValueError(f"Could not parse predicted peak position: {part}")
        value = float(match.group(0))
        label = part.replace(match.group(0), "").strip(" :=()[]")
        peaks.append(PredictedPeak(value, label or f"{value:g}", 1.0, None))
    return peaks


def _allowed_reflection(lattice_type, h, k, l):
    if h == k == l == 0:
        return False
    if lattice_type == "bcc":
        return (h + k + l) % 2 == 0
    if lattice_type == "fcc":
        return len({h % 2, k % 2, l % 2}) == 1
    return True


def _d_spacing(lattice_type, h, k, l, a, b=None, c=None):
    if h == k == l == 0:
        return None
    if lattice_type in {"cubic", "simple", "fcc", "bcc"}:
        value = h * h + k * k + l * l
        return None if value <= 0 else float(a) / math.sqrt(value)
    if lattice_type == "tetragonal":
        c = float(c if c not in (None, "") else a)
        value = (h * h + k * k) / float(a) ** 2 + (l * l) / c**2
        return None if value <= 0 else 1.0 / math.sqrt(value)
    if lattice_type == "hcp":
        c = float(c if c not in (None, "") else a)
        value = (4.0 / 3.0) * (h * h + h * k + k * k) / float(a) ** 2 + (l * l) / c**2
        return None if value <= 0 else 1.0 / math.sqrt(value)
    if lattice_type == "orthorhombic":
        if b in (None, "") or c in (None, ""):
            raise ValueError("Orthorhombic lattice requires a, b, and c")
        value = (h * h) / float(a) ** 2 + (k * k) / float(b) ** 2 + (l * l) / float(c) ** 2
        return None if value <= 0 else 1.0 / math.sqrt(value)
    raise ValueError(f"Unsupported lattice type: {lattice_type}")


def _multiplicity(h, k, l):
    reflections = set()
    values = (int(h), int(k), int(l))
    for perm in set(itertools.permutations(values)):
        sign_options = []
        for value in perm:
            sign_options.append((0,) if value == 0 else (-abs(value), abs(value)))
        for signed in itertools.product(*sign_options):
            reflections.add(signed)
    return len(reflections)


def lattice_predicted_peaks(
    lattice_type,
    a,
    b=None,
    c=None,
    wavelength=None,
    energy=None,
    max_index=8,
    min_two_theta=None,
    max_two_theta=None,
    phase_name="",
):
    lattice_type = str(lattice_type or "cubic").lower().strip()
    if lattice_type == "cubic/simple":
        lattice_type = "cubic"
    wavelength = float(wavelength) if wavelength not in (None, "") else energy_to_wavelength(energy)
    if wavelength is None:
        raise ValueError("Lattice peak prediction requires wavelength or energy")
    max_index = int(max_index or 8)
    peaks = []
    seen = set()
    for h in range(0, max_index + 1):
        for k in range(0, max_index + 1):
            for l in range(0, max_index + 1):
                if not _allowed_reflection(lattice_type, h, k, l):
                    continue
                spacing = _d_spacing(lattice_type, h, k, l, a, b=b, c=c)
                if spacing is None:
                    continue
                sine_theta = wavelength / (2.0 * spacing)
                if sine_theta <= 0 or sine_theta > 1:
                    continue
                two_theta = math.degrees(2.0 * math.asin(sine_theta))
                if min_two_theta is not None and two_theta < float(min_two_theta):
                    continue
                if max_two_theta is not None and two_theta > float(max_two_theta):
                    continue
                rounded = round(two_theta, 5)
                if rounded in seen:
                    continue
                seen.add(rounded)
                hkl = (h, k, l)
                label = f"{phase_name} " if phase_name else ""
                label += f"({h}{k}{l})"
                peaks.append(PredictedPeak(two_theta, label.strip(), _multiplicity(h, k, l), hkl))
    return sorted(peaks, key=lambda peak: peak.two_theta)


def build_predicted_peaks(source="list", two_theta_list="", **kwargs):
    if source in {"list", "2theta", "two_theta"}:
        return parse_two_theta_peaks(two_theta_list)
    if source == "lattice":
        return lattice_predicted_peaks(**kwargs)
    raise ValueError(f"Unknown predicted peak source: {source}")


def thin_labelled_peaks(peaks: Iterable[PredictedPeak], min_spacing=1.0):
    labelled = []
    last_labelled = -np.inf
    for peak in sorted(peaks, key=lambda item: item.two_theta):
        if peak.two_theta - last_labelled >= float(min_spacing):
            labelled.append(peak)
            last_labelled = peak.two_theta
    return labelled


def unique_peak_positions(peaks: Iterable[PredictedPeak], decimals=5):
    unique = []
    seen = set()
    for peak in sorted(peaks or [], key=lambda item: item.two_theta):
        key = round(float(peak.two_theta), int(decimals))
        if key in seen:
            continue
        seen.add(key)
        unique.append(peak)
    return unique


def overlay_predicted_peaks(
    ax,
    peaks,
    color="black",
    alpha=0.55,
    linewidth=0.5,
    min_label_spacing=1.0,
    label_x_offset_points=-3,
):
    peaks = list(peaks or [])
    if not peaks:
        return []
    peaks = unique_peak_positions(peaks)
    ymin, ymax = ax.get_ylim()
    ax.vlines(
        [peak.two_theta for peak in peaks],
        ymin,
        ymax,
        colors=color,
        alpha=alpha,
        linewidth=linewidth,
    )
    labelled = thin_labelled_peaks(peaks, min_spacing=min_label_spacing)
    y_text = ymin + 0.96 * (ymax - ymin)
    for peak in labelled:
        ax.annotate(
            peak.label or f"{peak.two_theta:g}",
            xy=(peak.two_theta, y_text),
            xytext=(label_x_offset_points, 0),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color=color,
            alpha=min(1.0, alpha + 0.25),
        )
    ax.set_ylim(ymin, ymax)
    return labelled
