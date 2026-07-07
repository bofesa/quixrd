"""Simple XRD peak plotter.

This app lets you enter one or more lattice definitions and overlay their
powder diffraction peak positions on a single plot.

Supported lattice types:
- fcc
- bcc
- hcp
- orthorhombic

The program uses Bragg's law to convert d-spacings into 2θ peak positions and
draws a stick pattern for each queued lattice.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PatternSpec:
    name: str
    lattice_type: str
    a: float
    b: float | None
    c: float | None
    wavelength: float
    min_two_theta: float
    max_two_theta: float


@dataclass(frozen=True)
class Peak:
    two_theta: float
    intensity: float
    hkl: tuple[int, int, int]


LATTICE_TYPES = ("fcc", "hcp", "bcc", "orthorhombic")
# would be nice to implement tetragonal


def allowed_cubic_reflection(lattice_type: str, h: int, k: int, l: int) -> bool:
    if h == k == l == 0:
        return False

    if lattice_type == "bcc":
        return (h + k + l) % 2 == 0

    if lattice_type == "fcc":
        parity = {h % 2, k % 2, l % 2}
        return len(parity) == 1

    return True


def d_spacing(spec: PatternSpec, h: int, k: int, l: int) -> float | None:
    if h == k == l == 0:
        return None

    if spec.lattice_type in {"fcc", "bcc"}:
        value = h * h + k * k + l * l
        if value == 0:
            return None
        return spec.a / math.sqrt(value)

    if spec.lattice_type == "hcp":
        if not spec.c:
            return None
        basal = (4.0 / 3.0) * (h * h + h * k + k * k) / (spec.a * spec.a)
        axial = (l * l) / (spec.c * spec.c)
        value = basal + axial
        return None if value <= 0 else 1.0 / math.sqrt(value)

    if spec.lattice_type == "orthorhombic":
        if not spec.b or not spec.c:
            return None
        value = (h * h) / (spec.a * spec.a) + (k * k) / (spec.b * spec.b) + (l * l) / (spec.c * spec.c)
        return None if value <= 0 else 1.0 / math.sqrt(value)

    return None


def multiplicity_cubic(h: int, k: int, l: int) -> int:
    '''Returns the multiplicity of a cubic reflection based on its Miller indices.'''
    h, k, l = sorted((abs(h), abs(k), abs(l)), reverse=True)
    if l == 0:
        if k == 0:
            return 6      # h00
        elif h == k:
            return 12     # hh0
        else:
            return 24     # hk0

    if h == k == l:
        return 8          # hhh
    elif h == k or k == l:
        return 24         # hhl
    else:
        return 48         # hkl


def multiplicity(spec: PatternSpec, h: int, k: int, l: int) -> int:
    '''Returns the multiplicity of a reflection based on its Miller indices and lattice type.'''
    h, k, l = abs(h), abs(k), abs(l)

    if spec.lattice_type in {"fcc", "bcc"}:
        return multiplicity_cubic(h, k, l)

    elif spec.lattice_type == "hcp":
        nonzero = sum(x != 0 for x in (h, k, l))
        if nonzero == 1:
            return 6 if l == 0 else 2
        elif nonzero == 2:
            return 12
        else:
            return 12

    elif spec.lattice_type in {"tetragonal", "orthorhombic"}:
        nonzero = sum(x != 0 for x in (h, k, l))
        if nonzero == 1:
            return 2
        elif nonzero == 2:
            return 4
        else:
            return 8
    return 1


def enumerate_peaks(spec: PatternSpec) -> list[Peak]:
    peaks: list[Peak] = []
    limit = 8

    for h in range(0, limit + 1):
        for k in range(0, limit + 1):
            for l in range(0, limit + 1):
                if h == k == l == 0:
                    continue

                if spec.lattice_type in {"fcc", "bcc"} and not allowed_cubic_reflection(spec.lattice_type, h, k, l):
                    continue

                spacing = d_spacing(spec, h, k, l)
                if spacing is None:
                    continue

                sine_theta = spec.wavelength / (2.0 * spacing)
                if sine_theta > 1.0:
                    continue

                two_theta = math.degrees(2.0 * math.asin(sine_theta))
                if two_theta < spec.min_two_theta or two_theta > spec.max_two_theta:
                    continue

                # Calcuate intensities using structure factor
                intensity = multiplicity(spec, h, k, l)
                peaks.append(Peak(two_theta=two_theta, intensity=intensity, hkl=(h, k, l)))


    peaks.sort(key=lambda peak: peak.two_theta)
    return peaks


def normalize_intensities(peaks: list[Peak]) -> list[Peak]:
    if not peaks:
        return peaks

    max_intensity = max(peak.intensity for peak in peaks)
    if max_intensity <= 0:
        return peaks

    return [Peak(two_theta=peak.two_theta, intensity=peak.intensity / max_intensity, hkl=peak.hkl) for peak in peaks]


class XRDPlotterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("XRD Pattern Overlay Plotter")
        self.root.geometry("1180x720")

        self.patterns: list[PatternSpec] = []

        self.name_var = tk.StringVar(value="Pattern 1")
        self.lattice_var = tk.StringVar(value="fcc")
        self.a_var = tk.StringVar(value="3.6")
        self.b_var = tk.StringVar(value="3.6")
        self.c_var = tk.StringVar(value="5.9")
        self.wavelength_var = tk.StringVar(value="1.5406")
        self.min_two_theta_var = tk.StringVar(value="0")
        self.max_two_theta_var = tk.StringVar(value="90")

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(outer)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        plot_frame = ttk.Frame(outer)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        form = ttk.LabelFrame(controls, text="Pattern Settings", padding=12)
        form.pack(fill=tk.X)

        self._add_entry(form, "Name", self.name_var, 0)
        self._add_dropdown(form, "Lattice", self.lattice_var, LATTICE_TYPES, 1)
        self._add_entry(form, "a (Å)", self.a_var, 2)
        self._add_entry(form, "b (Å)", self.b_var, 3)
        self._add_entry(form, "c (Å)", self.c_var, 4)
        self._add_entry(form, "Wavelength (Å)", self.wavelength_var, 5)
        self._add_entry(form, "Min 2θ (deg)", self.min_two_theta_var, 6)
        self._add_entry(form, "Max 2θ (deg)", self.max_two_theta_var, 7)

        button_row = ttk.Frame(form)
        button_row.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        button_row.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(button_row, text="Add Pattern", command=self.add_pattern).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(button_row, text="Save Pattern", command=self.save_pattern).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(button_row, text="Load Pattern", command=self.load_pattern).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        list_frame = ttk.LabelFrame(controls, text="Queued Patterns", padding=12)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.pattern_list = tk.Listbox(list_frame, height=6)
        self.pattern_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.pattern_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.pattern_list.config(yscrollcommand=scrollbar.set)

        list_buttons = ttk.Frame(controls)
        list_buttons.pack(fill=tk.X, pady=(10, 0))
        list_buttons.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(list_buttons, text="Remove Selected", command=self.remove_selected).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(list_buttons, text="Clear All", command=self.clear_patterns).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(list_buttons, text="Plot Patterns", command=self.plot_patterns).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        save_buttons = ttk.Frame(controls)
        save_buttons.pack(fill=tk.X, pady=(10, 0))
        save_buttons.columnconfigure((0,), weight=1)

        ttk.Button(save_buttons, text="Save Plot", command=self.save_plot).grid(row=0, column=0, sticky="ew")

        help_box = ttk.LabelFrame(controls, text="Notes", padding=12)
        help_box.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(
            help_box,
            text=(
                "For fcc and bcc, only the allowed reflections are shown.\n"
                "hcp uses the hexagonal d-spacing equation.\n"
                "Orthorhombic uses the standard three-axis lattice spacing."
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        self.figure = Figure(figsize=(8.8, 6.0), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.format_axes(ax=self.ax)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def current_pattern_spec(self) -> PatternSpec:
        return self._parse_spec()

    def _add_entry(self, parent: ttk.LabelFrame, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=4, padx=(0, 10))
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky=tk.EW, pady=4)
        parent.columnconfigure(1, weight=1)

    def _add_dropdown(self, parent: ttk.LabelFrame, label: str, variable: tk.StringVar, values: tuple[str, ...], row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=4, padx=(0, 10))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=16)
        combo.grid(row=row, column=1, sticky=tk.EW, pady=4)
        parent.columnconfigure(1, weight=1)

    def _parse_spec(self) -> PatternSpec:
        lattice_type = self.lattice_var.get().strip().lower()
        if lattice_type not in LATTICE_TYPES:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")

        a = float(self.a_var.get())
        b_text = self.b_var.get().strip()
        c_text = self.c_var.get().strip()
        wavelength = float(self.wavelength_var.get())
        min_two_theta = float(self.min_two_theta_var.get())
        max_two_theta = float(self.max_two_theta_var.get())

        if a <= 0 or wavelength <= 0 or max_two_theta <= 0:
            raise ValueError("Lattice parameters, wavelength, and max 2θ must be positive.")
        if min_two_theta < 0:
            raise ValueError("Min 2θ must be zero or positive.")
        if min_two_theta >= max_two_theta:
            raise ValueError("Min 2θ must be smaller than max 2θ.")

        if lattice_type == "hcp":
            c = float(c_text) if c_text else None
            if c is None or c <= 0:
                raise ValueError("hcp requires both a and c.")
            return PatternSpec(
                name=self.name_var.get().strip() or "Pattern",
                lattice_type=lattice_type,
                a=a,
                b=None,
                c=c,
                wavelength=wavelength,
                min_two_theta=min_two_theta,
                max_two_theta=max_two_theta,
            )

        if lattice_type == "orthorhombic":
            b = float(b_text) if b_text else None
            c = float(c_text) if c_text else None
            if b is None or c is None or b <= 0 or c <= 0:
                raise ValueError("orthorhombic requires a, b, and c.")
            return PatternSpec(
                name=self.name_var.get().strip() or "Pattern",
                lattice_type=lattice_type,
                a=a,
                b=b,
                c=c,
                wavelength=wavelength,
                min_two_theta=min_two_theta,
                max_two_theta=max_two_theta,
            )

        return PatternSpec(
            name=self.name_var.get().strip() or "Pattern",
            lattice_type=lattice_type,
            a=a,
            b=None,
            c=None,
            wavelength=wavelength,
            min_two_theta=min_two_theta,
            max_two_theta=max_two_theta,
        )

    def add_pattern(self) -> None:
        try:
            spec = self._parse_spec()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.patterns.append(spec)
        self.pattern_list.insert(
            tk.END,
            f"{spec.name} | {spec.lattice_type} | λ={spec.wavelength:g} Å | {spec.min_two_theta:g}°-{spec.max_two_theta:g}°",
        )

    def save_pattern(self) -> None:
        selected = self.pattern_list.curselection()
        if selected:
            spec = self.patterns[selected[0]]
        else:
            try:
                spec = self.current_pattern_spec()
            except ValueError as exc:
                messagebox.showerror("Invalid input", str(exc))
                return

        default_name = f"{spec.name.replace(' ', '_')}_{spec.lattice_type}.txt"
        path_text = filedialog.asksaveasfilename(
            title="Save pattern parameters",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path_text:
            return

        lines = [
            f"name = {spec.name}",
            f"lattice = {spec.lattice_type}",
            f"a = {spec.a:g}",
        ]
        if spec.lattice_type == "hcp":
            lines.append(f"c = {spec.c:g}")
        elif spec.lattice_type == "orthorhombic":
            lines.append(f"b = {spec.b:g}")
            lines.append(f"c = {spec.c:g}")
        else:
            lines.append(f"b = {spec.a:g}")
            lines.append(f"c = {spec.a:g}")
        lines.extend([
            f"wavelength = {spec.wavelength:g}",
            f"min_two_theta = {spec.min_two_theta:g}",
            f"max_two_theta = {spec.max_two_theta:g}",
        ])

        Path(path_text).write_text("\n".join(lines) + "\n", encoding="utf-8")
        messagebox.showinfo("Saved", f"Saved pattern parameters to\n{path_text}")
    
    def load_pattern(self) -> None:
        path_text = filedialog.askopenfilename(
            title="Load pattern parameters",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],)
        if not path_text:
            return
        try:
            content = Path(path_text).read_text(encoding="utf-8")
            params = {}
            for line in content.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    params[key.strip()] = value.strip()

            spec = PatternSpec(
                name=params["name"],
                lattice_type=params["lattice"],
                a=float(params["a"]),
                b=float(params["b"]),
                c=float(params["c"]),
                wavelength=float(params["wavelength"]),
                min_two_theta=float(params["min_two_theta"]),
                max_two_theta=float(params["max_two_theta"]),
            )
            self.patterns.append(spec)
            self.pattern_list.insert(
                tk.END,
                f"{spec.name} | {spec.lattice_type} | λ={spec.wavelength:g} Å | {spec.min_two_theta:g}°-{spec.max_two_theta:g}°",
            )
        except Exception as exc:
            messagebox.showerror("Error loading pattern", f"Failed to load pattern from file:\n{exc}")


    def remove_selected(self) -> None:
        selected = list(self.pattern_list.curselection())
        if not selected:
            return

        for index in reversed(selected):
            self.pattern_list.delete(index)
            del self.patterns[index]
            
    def format_axes(self, ax) -> None:
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Normalized intensity")
        ax.set_title("Overlayed XRD Patterns")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.25)
        ax.minorticks_on()
        ax.grid(True, which='minor', axis='x', alpha=0.1)

    def clear_patterns(self) -> None:
        self.patterns.clear()
        self.pattern_list.delete(0, tk.END)
        self.ax.clear()
        self.format_axes(self.ax)
        self.canvas.draw()

    def plot_patterns(self) -> None:
        if not self.patterns:
            messagebox.showinfo("No patterns", "Add at least one pattern before plotting.")
            return

        self.ax.clear()
        self.format_axes(self.ax)
        self.ax.set_xlim(min(spec.min_two_theta for spec in self.patterns), max(spec.max_two_theta for spec in self.patterns))

        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        offset_step = 1.2

        for index, spec in enumerate(self.patterns):
            peaks = normalize_intensities(enumerate_peaks(spec))
            if not peaks:
                continue
            # Remove equivalent peaks with same 2θ (e.g. 110 and 101 in orthorhombic) to avoid duplicate labels, and overwrite peaks
            unique_peaks = []
            for peak in peaks:
                if not any(abs(peak.two_theta - other_peak.two_theta) < 0.1 for other_peak in unique_peaks):
                    unique_peaks.append(peak)
            peaks = unique_peaks

            print(f"Material: {spec.name}, \nLattice: {spec.lattice_type}, \nPeaks: \n{chr(10).join(f'{peak.hkl}, {peak.two_theta:.2f}°, {peak.intensity:.2f}' for peak in peaks)}")

            offset = index * offset_step
            color = colors[index % len(colors)]
            label = f"{spec.name} ({spec.lattice_type})"

            visible_peaks = [peak for peak in peaks if peak.two_theta >= spec.min_two_theta]
            for peak in visible_peaks:
                top = offset + peak.intensity
                self.ax.vlines(peak.two_theta, offset, top, color=color, linewidth=2, alpha=0.9)

            self.ax.hlines(offset, spec.min_two_theta, spec.max_two_theta, color=color, linewidth=1, alpha=0.2)
            self.ax.text(self.ax.get_xlim()[0] + 0.5, offset-0.1, label, color=color, fontsize=9, va="bottom", ha="left")

            strongest = max(visible_peaks or peaks, key=lambda peak: peak.intensity)
            self.ax.annotate(
                f"{strongest.hkl[0]}{strongest.hkl[1]}{strongest.hkl[2]}",
                xy=(strongest.two_theta, offset + strongest.intensity),
                xytext=(-6, 2),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
            nearly_strongest = [peak for peak in visible_peaks if peak.intensity >= 0.3 * strongest.intensity and peak != strongest]
            for i, peak in enumerate(nearly_strongest):
                # Place labels slightly differently overlapping labels if there are two very close strong peaks
                if any(abs(peak.two_theta - other_peak.two_theta) < 0.5 for other_peak in nearly_strongest[:i] if other_peak != peak):
                    self.ax.annotate(
                        f"{peak.hkl[0]}{peak.hkl[1]}{peak.hkl[2]}",
                        xy=(peak.two_theta, offset + peak.intensity),
                        xytext=(-6, 8),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                    )
                else:
                    self.ax.annotate(
                        f"{peak.hkl[0]}{peak.hkl[1]}{peak.hkl[2]}",
                        xy=(peak.two_theta, offset + peak.intensity),
                        xytext=(-6, 2),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                    )

        self.ax.set_ylim(-0.2, len(self.patterns) * offset_step + 0.2)
        self.ax.grid(True, which='minor', axis='x', alpha=0.1)
        self.canvas.draw()

    def save_plot(self) -> None:
        if not self.patterns:
            messagebox.showinfo("No patterns", "Add at least one pattern before saving the plot.")
            return

        path_text = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF file", "*.pdf"),
                ("SVG file", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not path_text:
            return

        self.figure.savefig(path_text, bbox_inches="tight")
        messagebox.showinfo("Saved", f"Saved plot to\n{path_text}")


def main() -> None:
    root = tk.Tk()
    XRDPlotterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
