from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
import tkinter as tk
import tkinter.font as tkfont
import hashlib
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


os.environ.setdefault("quixrd_GUI_INTERACTIVE", "1")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from quixrd.xrd_processing import sin2psi_processor as proc
    from quixrd.xrd_processing import spinodal_peak_analysis as peak_analysis
    from quixrd.xrd_processing import twotheta_calibration as tth_cal
    from quixrd.xrd_processing import williamson_hall as wh
else:
    from . import sin2psi_processor as proc
    from . import spinodal_peak_analysis as peak_analysis
    from . import twotheta_calibration as tth_cal
    from . import williamson_hall as wh

NXS_EXPORT_DIR = Path(__file__).resolve().parents[1] / "nxs_export"
if str(NXS_EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(NXS_EXPORT_DIR))


SIN2PSI_LABEL = "sin\u00b2\u03c8"
ENERGY_WAVELENGTH_CONSTANT = 12.3984193
SPECTRA_SCAN_TYPES = ("chi", "delta", "z", "omega")
SPECTRA_LABEL_OPTIONS = (
    ("type", "Scan type", "Add the scan type, such as chi or delta, to legend labels."),
    ("temp", "Temperature", "Add the first-frame temperature metadata to legend labels."),
    ("time", "Start time", "Add the first-frame start time metadata to legend labels."),
)
TAB_NAMES = ["Extraction", "Plotting", "Sorting", "Peak Analysis", "Williamson-Hall", f"{SIN2PSI_LABEL} Analysis"]
TAB_PREFIXES = {
    "Extraction": ("extract.",),
    "Calibration": ("calibration.",),
    "Plotting": ("plot.",),
    "Sorting": ("sort.",),
    "Peak Analysis": ("peak.",),
    "Williamson-Hall": ("wh.",),
    f"{SIN2PSI_LABEL} Analysis": ("sin2psi.",),
}

X_METADATA_OPTIONS = [
    "scan_number",
    "temperature",
    "energy",
    "start_time",
    "frame_time",
    "chi",
    "psi_deg",
    "sin2psi",
]
PEAK_SECONDARY_Y_OPTIONS = [
    "none",
    "delta_bic",
    "minor_major_height_ratio",
    "two_minor_major_height_ratio",
    "single_center_1",
    "two_center_1",
    "two_center_2",
    "single_fwhm_1",
    "two_fwhm_1",
    "two_fwhm_2",
    "temperature",
    "energy",
    "chi",
    "frame_time",
    "start_time",
]
APP_CONFIG_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".config") / "quixrd"
GUI_SETTINGS_PATH = APP_CONFIG_DIR / "gui_settings.json"
DEFAULT_CACHE_ROOT = APP_CONFIG_DIR / "cache"
APP_ICON_PATH = Path(__file__).resolve().parents[1] / "qx.ico"
LARGE_CACHE_COPY_THRESHOLD = 100
LARGE_FIT_SCAN_THRESHOLD = 50


class ToolTip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id = None
        self._window = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._window = tk.Toplevel(self.widget)
        self._window.wm_overrideredirect(True)
        self._window.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            self._window,
            text=self.text,
            padding=(8, 5),
            relief="solid",
            borderwidth=1,
            wraplength=360,
        )
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None


class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, padding=12)
        self.content.columnconfigure(0, weight=1)

        self._window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_content_configure(self, _event=None):
        self._update_scrollregion()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window_id, width=event.width)
        self._update_scrollregion()

    def _update_scrollregion(self):
        bbox = self.canvas.bbox("all")
        if bbox is None:
            return
        content_height = max(bbox[3], self.canvas.winfo_height())
        self.canvas.configure(scrollregion=(bbox[0], bbox[1], bbox[2], content_height))
        if bbox[3] <= self.canvas.winfo_height():
            self.canvas.yview_moveto(0)

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        bbox = self.canvas.bbox("all")
        if bbox is None or bbox[3] <= self.canvas.winfo_height():
            self.canvas.yview_moveto(0)
            return "break"
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"


class GuiLogStream:
    def __init__(self, emit):
        self.emit = emit
        self.capture = StringIO()
        self._pending = ""

    def write(self, text):
        if not text:
            return 0
        self.capture.write(text)
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if line.strip():
                self.emit(line)
        return len(text)

    def flush(self):
        if self._pending.strip():
            self.emit(self._pending)
        self._pending = ""

    def getvalue(self):
        return self.capture.getvalue()


class XRDGuiApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.master = master
        self.variables = {}
        self.widgets = {}
        self.sections = {}
        self.placeholders = {}
        self.placeholder_active = {}
        self.editable_combos = set()
        self.browse_kinds = {}
        self.widget_keys = {}
        self.help_window = None
        self.calibration_window = None
        self.settings_path = GUI_SETTINGS_PATH
        self.settings = self._load_gui_settings()
        self.cache_root = Path(self.settings.get("cache_root") or DEFAULT_CACHE_ROOT)
        self.use_local_cache_var = tk.BooleanVar(value=self._use_local_cache_default())
        self.apply_twotheta_calibration_var = tk.BooleanVar(
            value=bool(self.settings.get("apply_twotheta_calibration", False))
        )
        self.twotheta_calibration_file = tk.StringVar(value=self.settings.get("twotheta_calibration_file", ""))
        self.status_var = tk.StringVar(value="Ready")
        self._syncing_energy_wavelength = False
        self._syncing_outlier_preset = False
        self.cancel_event = threading.Event()
        self.task_running = False
        self._configure_master()
        self._build()

    def _configure_master(self):
        self.master.title("quixrd Workflow")
        self._apply_window_icon(self.master)
        self.master.geometry("1120x780")
        self.master.minsize(900, 620)

    def _apply_window_icon(self, window):
        if not APP_ICON_PATH.exists():
            return False
        try:
            window.iconbitmap(default=str(APP_ICON_PATH))
            return True
        except tk.TclError:
            return False

    def _build(self):
        self.pack(fill="both", expand=True)
        self._configure_styles()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)

        self._build_menu_bar()

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tabs = {}
        self.tab_frames = {}
        for name in TAB_NAMES:
            outer = ttk.Frame(self.notebook)
            outer.columnconfigure(0, weight=1)
            outer.rowconfigure(0, weight=1)
            scroller = ScrollableFrame(outer)
            scroller.grid(row=0, column=0, sticky="nsew")
            self.notebook.add(outer, text=name)
            self.tab_frames[name] = outer
            self.tabs[name] = scroller.content

        self._build_extraction_tab(self.tabs["Extraction"])
        self._build_plotting_tab(self.tabs["Plotting"])
        self._build_sorting_tab(self.tabs["Sorting"])
        self._build_peak_analysis_tab(self.tabs["Peak Analysis"])
        self._build_williamson_hall_tab(self.tabs["Williamson-Hall"])
        self._build_sin2psi_tab(self.tabs[f"{SIN2PSI_LABEL} Analysis"])
        self._build_log_panel()
        self._build_status_bar()
        self.log("GUI ready.")

    def _build_menu_bar(self):
        self.menu_bar = tk.Menu(self.master)

        file_menu = tk.Menu(self.menu_bar, tearoff=False)
        file_menu.add_command(label="Export Current Tab Parameters...", command=lambda: self.export_parameters_dialog("current"))
        file_menu.add_command(label="Export All Parameters...", command=lambda: self.export_parameters_dialog("all"))
        file_menu.add_separator()
        file_menu.add_command(label="Import Into Current Tab...", command=lambda: self.import_parameters_dialog("current"))
        file_menu.add_command(label="Import Into All Tabs...", command=lambda: self.import_parameters_dialog("all"))
        file_menu.add_separator()
        file_menu.add_command(label="Reveal Selected Path in File Explorer", command=self.reveal_selected_path)
        file_menu.add_separator()
        file_menu.add_command(label="Select Local Cache Folder...", command=self.select_local_cache_folder)
        file_menu.add_checkbutton(
            label="Use Local Cache",
            variable=self.use_local_cache_var,
            command=self._sync_use_local_cache_setting,
        )
        file_menu.add_command(label="Clear Local Cache", command=self.clear_local_cache)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.destroy)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        self.file_menu = file_menu

        calibration_menu = tk.Menu(self.menu_bar, tearoff=False)
        calibration_menu.add_command(label="Run 2theta Calibration...", command=self.open_calibration_window)
        calibration_menu.add_separator()
        calibration_menu.add_command(label="Select 2theta Calibration File...", command=self.select_twotheta_calibration_file)
        calibration_menu.add_checkbutton(
            label="Apply 2theta Calibration by Default",
            variable=self.apply_twotheta_calibration_var,
            command=self._sync_twotheta_calibration_setting,
        )
        self.menu_bar.add_cascade(label="Calibration", menu=calibration_menu)
        self.calibration_menu = calibration_menu

        help_menu = tk.Menu(self.menu_bar, tearoff=False)
        help_menu.add_command(label="Workflow Help", command=self.show_help)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        self.help_menu = help_menu

        self.master.configure(menu=self.menu_bar)

    def _configure_styles(self):
        self.style = ttk.Style(self.master)
        optional_font = tkfont.nametofont("TkDefaultFont").copy()
        optional_font.configure(slant="italic")
        self.optional_font = optional_font
        self.style.configure("Optional.TLabel", font=optional_font)
        self.style.configure("Optional.TCheckbutton", font=optional_font)

    def _section(self, parent, title, row):
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        return frame

    def _named_section(self, parent, name, title, row):
        frame = self._section(parent, title, row)
        self.sections[name] = frame
        return frame

    def _set_section_visible(self, name, visible):
        frame = self.sections.get(name)
        if frame is None:
            return
        if visible:
            frame.grid()
        else:
            frame.grid_remove()

    def _entry(self, parent, row, label, key, tooltip="", browse=None, default="", optional=False, placeholder=""):
        label_style = "Optional.TLabel" if optional else "TLabel"
        label_widget = ttk.Label(parent, text=label, style=label_style)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        var = tk.StringVar(value=default)
        self.variables[key] = var
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        self.widget_keys[str(entry)] = key
        self.placeholders[key] = placeholder
        self.placeholder_active[key] = False
        if placeholder and not default:
            self._restore_placeholder(key, entry)
        entry.bind("<FocusIn>", lambda _event, k=key, w=entry: self._clear_placeholder(k, w), add="+")
        entry.bind("<FocusOut>", lambda _event, k=key, w=entry: self._restore_placeholder(k, w), add="+")
        widgets = [label_widget, entry]
        if tooltip:
            ToolTip(entry, tooltip)
        if browse:
            self.browse_kinds[key] = browse
            button = ttk.Button(parent, text="Browse", command=lambda k=key, v=var, b=browse: self._browse(v, b, k))
            button.grid(row=row, column=2, sticky="e", padx=(8, 0), pady=4)
            widgets.append(button)
            self.widget_keys[str(button)] = key
            if tooltip:
                ToolTip(button, tooltip)
        self.widgets[key] = widgets
        return entry

    def _input_widget_for_key(self, key):
        for widget in self.widgets.get(key, []):
            if isinstance(widget, (ttk.Entry, ttk.Combobox)):
                return widget
        return None

    def _clear_placeholder(self, key, widget=None):
        if self.placeholder_active.get(key):
            self.placeholder_active[key] = False
            self.variables[key].set("")
            entry = self._input_widget_for_key(key)
            if entry is not None:
                try:
                    entry.configure(foreground="")
                except tk.TclError:
                    pass

    def _restore_placeholder(self, key, widget=None):
        placeholder = self.placeholders.get(key)
        if not placeholder or self.variables[key].get():
            return
        self.placeholder_active[key] = True
        self.variables[key].set(placeholder)
        target = widget or self._input_widget_for_key(key)
        if target is not None:
            try:
                target.configure(foreground="#777777")
            except tk.TclError:
                pass

    def _checkbox(self, parent, row, label, key, tooltip="", default=False, column=0, optional=False):
        var = tk.BooleanVar(value=default)
        self.variables[key] = var
        check_style = "Optional.TCheckbutton" if optional else "TCheckbutton"
        check = ttk.Checkbutton(parent, text=label, variable=var, style=check_style)
        check.grid(row=row, column=column, sticky="w", pady=4)
        self.widgets[key] = [check]
        if tooltip:
            ToolTip(check, tooltip)
        return check

    def _combo(self, parent, row, label, key, values, tooltip="", default=None, optional=False, readonly=True):
        label_style = "Optional.TLabel" if optional else "TLabel"
        label_widget = ttk.Label(parent, text=label, style=label_style)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        var = tk.StringVar(value=default or values[0])
        self.variables[key] = var
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly" if readonly else "normal")
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        self.widgets[key] = [label_widget, combo]
        if not readonly:
            self.editable_combos.add(key)
        if tooltip:
            ToolTip(combo, tooltip)
        return combo

    def _radio_group(self, parent, row, label, key, options, tooltip="", default=None):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        var = tk.StringVar(value=default or options[0][0])
        self.variables[key] = var
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="w", pady=4)
        radios = []
        for idx, option in enumerate(options):
            if len(option) == 3:
                value, text, option_tooltip = option
            else:
                value, text = option
                option_tooltip = tooltip
            radio = ttk.Radiobutton(frame, text=text, value=value, variable=var)
            radio.grid(row=0, column=idx, sticky="w", padx=(0, 12))
            if option_tooltip:
                ToolTip(radio, option_tooltip)
            radios.append(radio)
        self.widgets[key] = [label_widget, frame] + radios
        return var

    def _action_button(self, parent, row, label, command, tooltip=None):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        button = ttk.Button(frame, text=label, command=command)
        button.grid(row=0, column=0, padx=(0, 8))
        ToolTip(button, tooltip or "Run this workflow with the current tab values.")

    def _plot_buttons(self, parent, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        run_button = ttk.Button(frame, text="Run Selected Plot", command=self.run_plotting)
        run_button.grid(row=0, column=0, padx=(0, 8))
        ToolTip(run_button, "Run the selected plotting workflow with the current values.")
        save_button = ttk.Button(frame, text="Save Current Plot", command=self.save_current_plot)
        save_button.grid(row=0, column=1, padx=(0, 8))
        ToolTip(save_button, "Save the currently open Matplotlib figure to saved_plots.")

    def _set_enabled(self, keys, enabled):
        state = "normal" if enabled else "disabled"
        for key in keys:
            for widget in self.widgets.get(key, []):
                try:
                    if isinstance(widget, ttk.Combobox):
                        widget.configure(state=("normal" if key in self.editable_combos else "readonly") if enabled else "disabled")
                    else:
                        widget.configure(state=state)
                except tk.TclError:
                    pass

    def _set_widgets_visible(self, keys, visible):
        for key in keys:
            for widget in self.widgets.get(key, []):
                if visible:
                    widget.grid()
                else:
                    widget.grid_remove()

    def _scan_type_checkboxes(self, parent, row):
        label_widget = ttk.Label(parent, text="Scan types")
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="w", pady=4)
        widgets = [label_widget, frame]
        for idx, scan_type in enumerate(SPECTRA_SCAN_TYPES):
            key = f"plot.scan_type.{scan_type}"
            var = tk.BooleanVar(value=(scan_type == "chi" or scan_type == "delta"))
            self.variables[key] = var
            check = ttk.Checkbutton(frame, text=scan_type, variable=var)
            check.grid(row=0, column=idx, sticky="w", padx=(0, 12))
            ToolTip(check, f"Include {scan_type} scans in spectra plots.")
            widgets.append(check)
        self.widgets["plot.scan_types"] = widgets
        return frame

    def _spectra_label_checkboxes(self, parent, row):
        label_widget = ttk.Label(parent, text="Legend labels")
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="w", pady=4)
        widgets = [label_widget, frame]
        for idx, (value, text, tooltip) in enumerate(SPECTRA_LABEL_OPTIONS):
            key = f"plot.label.{value}"
            var = tk.BooleanVar(value=False)
            self.variables[key] = var
            check = ttk.Checkbutton(frame, text=text, variable=var)
            check.grid(row=0, column=idx, sticky="w", padx=(0, 12))
            ToolTip(check, tooltip)
            widgets.append(check)
        self.widgets["plot.labels"] = widgets
        return frame

    def _update_extraction_mode(self):
        mode = self.variables["extract.mode"].get()
        self._set_enabled(["extract.mirror"], mode == "batch")
        self.set_status(f"Extraction mode: {mode}")

    def _update_plot_mode(self):
        mode = self.variables["plot.type"].get()
        selector = self.variables["plot.fwhm_selector"].get()
        peak_source = self.variables.get("plot.predicted_source")
        peak_source_value = peak_source.get() if peak_source is not None else "list"
        lattice_type_var = self.variables.get("plot.predicted_lattice_type")
        lattice_type = lattice_type_var.get() if lattice_type_var is not None else "fcc"
        show_predicted = self._bool("plot.show_predicted_peaks") if "plot.show_predicted_peaks" in self.variables else False
        spectra_mode = mode == "spectra"
        trend_mode = mode in {"gradient", "stress", "fwhm", "peak_position"}
        fit_value_mode = mode in {"fwhm", "peak_position"}
        self._set_widgets_visible(["plot.scan_types", "plot.labels", "plot.offset", "plot.show_predicted_peaks"], spectra_mode)
        self._set_enabled(["plot.scan_types", "plot.labels", "plot.offset"], spectra_mode)
        self._set_widgets_visible(["plot.x"], trend_mode)
        self._set_enabled(["plot.x"], trend_mode)
        self._set_widgets_visible(["plot.summary_csv"], mode in {"gradient", "stress"})
        self._set_enabled(["plot.summary_csv"], mode in {"gradient", "stress"})
        self._set_widgets_visible(["plot.fwhm_selector"], fit_value_mode)
        self._set_widgets_visible(["plot.fwhm_frame"], fit_value_mode and selector == "frame")
        self._set_widgets_visible(["plot.fwhm_chi"], fit_value_mode and selector == "chi")
        self._set_enabled(["plot.fwhm_selector"], fit_value_mode)
        self._set_enabled(["plot.fwhm_frame"], fit_value_mode and selector == "frame")
        self._set_enabled(["plot.fwhm_chi"], fit_value_mode and selector == "chi")
        self._set_enabled(["plot.show_predicted_peaks"], spectra_mode)
        predicted_enabled = spectra_mode and show_predicted
        list_enabled = predicted_enabled and peak_source_value == "list"
        lattice_enabled = predicted_enabled and peak_source_value == "lattice"
        self._set_widgets_visible(["plot.predicted_source"], predicted_enabled)
        self._set_widgets_visible(["plot.predicted_twotheta"], list_enabled)
        self._set_widgets_visible(
            [
                "plot.predicted_lattice_type",
                "plot.predicted_a",
                "plot.predicted_b",
                "plot.predicted_c",
                "plot.predicted_wavelength",
                "plot.predicted_energy",
                "plot.predicted_max_index",
                "plot.predicted_phase",
            ],
            lattice_enabled,
        )
        self._set_enabled(["plot.predicted_source"], predicted_enabled)
        self._set_enabled(["plot.predicted_twotheta"], list_enabled)
        cubic_like = lattice_type in {"cubic", "fcc", "bcc"}
        ac_like = lattice_type in {"hcp", "tetragonal"}
        self._set_enabled(
            [
                "plot.predicted_lattice_type",
                "plot.predicted_wavelength",
                "plot.predicted_energy",
                "plot.predicted_max_index",
                "plot.predicted_phase",
            ],
            lattice_enabled,
        )
        self._set_enabled(["plot.predicted_a"], lattice_enabled)
        self._set_enabled(["plot.predicted_b"], lattice_enabled and not cubic_like and not ac_like)
        self._set_enabled(["plot.predicted_c"], lattice_enabled and not cubic_like)
        self.set_status(f"Plot type: {mode}")

    def _update_calibration_mode(self):
        if "calibration.source_type" not in self.variables:
            return
        source_type = self.variables["calibration.source_type"].get()
        material = self.variables["calibration.material"].get()
        lattice_type = self.variables["calibration.lattice_type"].get()
        is_lab6 = material.startswith("LaB6")
        if is_lab6:
            self._set_variable_value("calibration.lattice_type", "cubic")
            self._set_variable_value("calibration.a", str(tth_cal.LAB6_A))
            lattice_type = "cubic"
        self._set_enabled(["calibration.flat_dir", "calibration.flat_scans"], source_type == "nxs")
        self._set_enabled(["calibration.lattice_type", "calibration.a"], not is_lab6)
        cubic_like = lattice_type in {"cubic", "fcc", "bcc"}
        ac_like = lattice_type in {"hcp", "tetragonal"}
        self._set_enabled(["calibration.b"], (not is_lab6) and not cubic_like and not ac_like)
        self._set_enabled(["calibration.c"], (not is_lab6) and not cubic_like)
        outlier_mode = self.variables.get("calibration.outlier_mode")
        mode = outlier_mode.get() if outlier_mode is not None else "off"
        threshold_enabled = mode in {"automatic", "review"}
        manual_enabled = mode == "manual"
        self._set_enabled(["calibration.outlier_sensitivity"], threshold_enabled)
        self._set_enabled(
            [
                "calibration.offset_floor",
                "calibration.caglioti_floor",
                "calibration.max_fwhm_multiplier",
                "calibration.max_outlier_fraction",
            ],
            threshold_enabled,
        )
        self._set_enabled(["calibration.manual_exclusions"], manual_enabled)
        self.set_status(f"Calibration source: {source_type}")

    def _autofill_calibration_outlier_preset(self):
        if self._syncing_outlier_preset or "calibration.outlier_sensitivity" not in self.variables:
            return
        sensitivity = self.variables["calibration.outlier_sensitivity"].get()
        preset = tth_cal.OUTLIER_SENSITIVITY_PRESETS.get(sensitivity)
        if not preset:
            return
        mapping = {
            "calibration.offset_floor": "offset_floor",
            "calibration.caglioti_floor": "caglioti_floor",
            "calibration.max_fwhm_multiplier": "max_fwhm_multiplier",
            "calibration.max_outlier_fraction": "max_outlier_fraction",
        }
        self._syncing_outlier_preset = True
        try:
            for gui_key, preset_key in mapping.items():
                if gui_key in self.variables:
                    self._set_variable_value(gui_key, f"{preset[preset_key]:g}")
        finally:
            self._syncing_outlier_preset = False
        self.set_status(f"Calibration outlier sensitivity: {sensitivity}")

    def _update_sort_mode(self):
        mode = self.variables["sort.mode"].get()
        self._set_enabled(["sort.sample_file", "sort.calibrations"], mode == "nxs")
        self._set_enabled(["sort.export_dir", "sort.move"], mode == "extracted")
        self.set_status(f"Sort mode: {mode}")

    def _update_peak_mode(self):
        if "peak.scan_type" not in self.variables:
            return
        action = self.variables.get("peak.action")
        action_value = action.get() if action is not None else "run"
        scan_type = self.variables["peak.scan_type"].get()
        show_frame = scan_type == "chi"
        if "peak.inputs" in self.sections:
            if action_value == "run":
                self.sections["peak.inputs"].grid()
            else:
                self.sections["peak.inputs"].grid_remove()
        if "peak.replot" in self.sections:
            if action_value == "replot":
                self.sections["peak.replot"].grid()
            else:
                self.sections["peak.replot"].grid_remove()
        if "peak.fit_options" in self.sections:
            if action_value == "run":
                self.sections["peak.fit_options"].grid()
            else:
                self.sections["peak.fit_options"].grid_remove()
        self._set_widgets_visible(["peak.frame_index"], show_frame)
        self._set_enabled(["peak.frame_index"], show_frame)
        if hasattr(self, "peak_action_button"):
            self.peak_action_button.configure(text="Run Peak Analysis" if action_value == "run" else "Replot Existing Results")
        if scan_type == "delta":
            self.set_status("Peak Analysis: delta scans use all frames")
        else:
            self.set_status(f"Peak Analysis {'mode: replot' if action_value == 'replot' else 'scan type: ' + (scan_type or 'all')}")

    def _update_wh_mode(self):
        if "wh.target_source" not in self.variables:
            return
        target_source = self.variables["wh.target_source"].get()
        profile_source = self.variables["wh.profile_source"].get()
        scan_type = self.variables["wh.scan_type"].get()
        lattice_type = self.variables["wh.lattice_type"].get()
        lattice = target_source == "lattice"
        self._set_widgets_visible(["wh.manual_two_theta"], not lattice)
        self._set_widgets_visible(
            [
                "wh.lattice_type",
                "wh.a",
                "wh.b",
                "wh.c",
                "wh.phase_name",
                "wh.max_index",
                "wh.thermal_alpha",
                "wh.reference_temperature",
            ],
            lattice,
        )
        self._set_widgets_visible(["wh.scan_type"], profile_source == "txt")
        show_frame = profile_source == "csv" or (profile_source == "txt" and scan_type == "chi")
        self._set_widgets_visible(["wh.frame_index"], show_frame)
        self._set_enabled(["wh.frame_index"], show_frame)
        cubic_like = lattice_type in {"cubic", "fcc", "bcc"}
        self._set_enabled(["wh.b", "wh.c"], lattice and not cubic_like)
        if cubic_like:
            if "wh.b" in self.variables:
                self.variables["wh.b"].set("")
            if "wh.c" in self.variables:
                self.variables["wh.c"].set("")
        self.set_status("Williamson-Hall: lattice targets" if lattice else "Williamson-Hall: manual 2theta targets")

    def _update_sin2psi_mode(self):
        action = self.variables["sin2psi.action"].get()
        method = self.variables.get("sin2psi.correction_method")
        method_value = method.get() if method is not None else "polynomial"
        peak_keys = ["sin2psi.peak_center", "sin2psi.track_window", "sin2psi.track_peak", "sin2psi.plot_frames"]
        exclusion_keys = [
            "sin2psi.exclude_frames",
            "sin2psi.exclude_chi",
            "sin2psi.exclude_sin2psi",
            "sin2psi.auto_exclude",
        ]
        correction_apply_keys = ["sin2psi.correction_json"]
        stress_keys = [
            "sin2psi.elastic_E",
            "sin2psi.elastic_E_units",
            "sin2psi.elastic_nu",
            "sin2psi.stress_reference_two_theta",
            "sin2psi.stress_reference_d0",
            "sin2psi.stress_wavelength",
            "sin2psi.stress_energy",
        ]
        calibration_keys = [
            "sin2psi.reference_folder",
            "sin2psi.reference_scan",
            "sin2psi.correction_method",
            "sin2psi.correction_degree",
            "sin2psi.reference_two_theta",
        ]
        summary_keys = ["sin2psi.summary_x", "sin2psi.summary_secondary_y"]
        self._set_section_visible("sin2psi.peak_options", action == "process")
        self._set_section_visible("sin2psi.exclusions", action in {"process", "refit", "correction"})
        self._set_section_visible("sin2psi.stress", action in {"process", "refit"})
        self._set_section_visible("sin2psi.calibration", action == "correction")
        self._set_section_visible("sin2psi.summary", action == "summaries")
        self._set_widgets_visible(["sin2psi.correction_json"], action in {"process", "refit"})
        self._set_widgets_visible(["sin2psi.correction_degree"], action == "correction" and method_value == "polynomial")
        self._set_enabled(peak_keys, action == "process")
        self._set_enabled(exclusion_keys, action in {"process", "refit", "correction"})
        self._set_enabled(correction_apply_keys, action in {"process", "refit"})
        self._set_enabled(stress_keys, action in {"process", "refit"})
        self._set_enabled(calibration_keys, action == "correction")
        self._set_enabled(summary_keys, action == "summaries")
        self._set_enabled(["sin2psi.correction_degree"], action == "correction" and method_value == "polynomial")
        self._set_enabled(["sin2psi.preview"], action in {"process", "refit"})
        self.set_status(f"{SIN2PSI_LABEL} action: {action}")

    def _sin2psi_buttons(self, parent, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        buttons = [
            ("Run Selected Action", self.run_sin2psi_action),
            ("Preview First Scan", self.preview_sin2psi),
        ]
        for idx, (label, command) in enumerate(buttons):
            button = ttk.Button(frame, text=label, command=lambda cmd=command: cmd())
            button.grid(row=0, column=idx, padx=(0, 8), pady=(0, 4))
            ToolTip(button, "Run the selected sin2psi workflow.")

    def _browse(self, variable, browse, key=None):
        if browse == "directory":
            value = filedialog.askdirectory()
        elif browse == "json":
            value = filedialog.askopenfilename(
                title="Select JSON file",
                filetypes=[("JSON files", "*.json")],
            )
        elif browse == "json_multi":
            values = filedialog.askopenfilenames(
                title="Select JSON file(s)",
                filetypes=[("JSON files", "*.json")],
            )
            value = "; ".join(values)
        elif browse == "calibration_input":
            source_type = self.variables.get("calibration.source_type")
            source_type = source_type.get() if source_type is not None else "txt"
            if source_type == "txt":
                values = filedialog.askopenfilenames(
                    title="Select calibration TXT frame files",
                    filetypes=[("TXT files", "*.txt"), ("All files", "*.*")],
                )
                value = "; ".join(values)
            else:
                filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
                if source_type == "nxs":
                    filetypes = [("NXS files", "*.nxs"), ("All files", "*.*")]
                value = filedialog.askopenfilename(
                    title="Select calibration input file",
                    filetypes=filetypes,
                )
        else:
            value = filedialog.askopenfilename()
        if value:
            if key is not None:
                self._clear_placeholder(key)
            variable.set(value)
            self.set_status(f"Selected: {value}")

    def _get(self, key):
        if self.placeholder_active.get(key):
            return ""
        value = self.variables[key].get()
        return value.strip() if isinstance(value, str) else value

    def _required_path(self, key, label):
        value = self._get(key)
        if not value:
            raise ValueError(f"{label} is required")
        return value

    def _optional_path(self, key):
        return self._get(key) or None

    def _optional_json_path(self, key, label):
        value = self._optional_path(key)
        if value:
            self._validate_json_paths(value, label)
        return value

    def _validate_json_paths(self, value, label):
        for part in str(value or "").split(";"):
            path_text = part.strip()
            if not path_text:
                continue
            path = Path(path_text)
            if path.suffix.lower() != ".json":
                raise ValueError(f"{label} must be a .json file, not: {path}")
        return True

    def _optional_float(self, key):
        value = self._get(key)
        return None if value in ("", None) else float(value)

    def _optional_int(self, key):
        value = self._get(key)
        return None if value in ("", None) else int(value)

    def _bool(self, key):
        return bool(self.variables[key].get())

    def _parse_every_nth(self, text):
        text = str(text or "").strip()
        if not text.startswith(":") or text.count(":") != 1:
            return None
        step = int(text[1:].strip())
        if step <= 0:
            raise ValueError("Every-nth step must be greater than zero")
        return step

    def _apply_every_nth(self, scans, step):
        if step is None or scans is None:
            return scans
        return list(scans)[::step]

    def _parse_int_list(self, text, required=False):
        text = str(text or "").strip()
        if not text:
            if required:
                raise ValueError("A scan range/list is required")
            return []
        if self._parse_every_nth(text) is not None:
            if required:
                raise ValueError("Use :n only where scans can be discovered automatically")
            return []
        values = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            step = None
            if ":" in part:
                part, step_text = [item.strip() for item in part.split(":", 1)]
                step = abs(int(step_text))
                if step == 0:
                    raise ValueError("Range step must be greater than zero")
            if "-" in part:
                start, end = [int(item.strip()) for item in part.split("-", 1)]
                signed_step = step or 1
                signed_step = signed_step if end >= start else -signed_step
                values.extend(range(start, end + signed_step, signed_step))
            else:
                if step is not None:
                    raise ValueError("Step syntax is only valid with a range, e.g. 440-500:5")
                values.append(int(part))
        return values

    def _parse_float_or_list(self, text):
        text = str(text or "").strip()
        if not text:
            return None
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def _parse_int_or_list(self, text):
        values = self._parse_int_list(text, required=False)
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def _parse_csv_list(self, text):
        return [part.strip() for part in str(text or "").split(",") if part.strip()]

    def _selected_plot_scan_types(self):
        selected = [scan_type for scan_type in SPECTRA_SCAN_TYPES if self._bool(f"plot.scan_type.{scan_type}")]
        if not selected:
            raise ValueError("Select at least one scan type for spectra plotting")
        return selected

    def _selected_spectra_labels(self):
        return [value for value, _text, _tooltip in SPECTRA_LABEL_OPTIONS if self._bool(f"plot.label.{value}")]

    def _parse_ranges(self, text):
        ranges = []
        for part in self._parse_csv_list(text):
            if "-" not in part:
                value = float(part)
                ranges.append((value, value))
                continue
            lower, upper = [float(item.strip()) for item in part.split("-", 1)]
            ranges.append((lower, upper))
        return ranges

    def _parse_date_range(self, text):
        text = str(text or "").strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        compact = text.replace(" ", "")
        if "-" in compact and len(compact.replace("-", "")) == 16:
            return [part.strip() for part in compact.split("-", 1)]
        return [text]

    def _selected_scan_list(self, key):
        scans = self._parse_int_list(self._get(key), required=True)
        if not scans:
            raise ValueError("At least one scan is required")
        return scans

    def _selected_sin2psi_scans(self, action=None):
        text = self._get("sin2psi.scans")
        every_nth = self._parse_every_nth(text)
        if text:
            if every_nth is None:
                scans = self._parse_int_list(text, required=True)
                if not scans:
                    raise ValueError("At least one scan is required")
                return scans

        if action == "correction":
            return []
        if action == "summaries" and every_nth is None:
            return None

        data_dir = self._required_path("sin2psi.data_dir", "Data directory")
        include_raw = action in (None, "process")
        include_processed = action in (None, "refit", "summaries")
        scans = proc.discover_scan_numbers(
            data_dir,
            include_raw=include_raw,
            include_processed=include_processed,
        )
        if not scans:
            scans = proc.discover_scan_numbers(data_dir)
        if not scans:
            raise ValueError(f"No scans found in {data_dir}")
        scans = self._apply_every_nth(scans, every_nth)
        return scans if scans else None if action == "summaries" else scans

    def _selected_plot_scans(self, data_dir, plot_type):
        text = self._get("plot.scans")
        every_nth = self._parse_every_nth(text)
        scans = self._parse_int_list(text, required=False)
        if scans:
            return scans
        if plot_type == "spectra":
            discovered = proc.discover_scan_numbers(data_dir, include_raw=True, include_processed=False)
            if not discovered:
                raise ValueError(f"No exported spectra scans found in {data_dir}")
            return self._apply_every_nth(discovered, every_nth)
        if every_nth is not None:
            discovered = proc.discover_scan_numbers(data_dir, include_raw=False, include_processed=True)
            if not discovered:
                discovered = proc.discover_scan_numbers(data_dir)
            if not discovered:
                raise ValueError(f"No scans found in {data_dir}")
            return self._apply_every_nth(discovered, every_nth)
        return None

    def _selected_wh_scans(self, data_dir, profile_source, scan_type, frame_index):
        text = self._get("wh.scans")
        every_nth = self._parse_every_nth(text)
        scans = self._parse_int_list(text, required=False)
        if scans:
            return scans
        discovered = wh.discover_scan_numbers(
            data_dir,
            profile_source=profile_source,
            scan_type=scan_type,
            frame_index=frame_index,
        )
        if not discovered:
            raise ValueError(f"No matching Williamson-Hall scans found in {data_dir}")
        return self._apply_every_nth(discovered, every_nth)

    def _format_derived_float(self, value):
        return f"{value:.8g}"

    def _sync_energy_wavelength(self, source_key, target_key):
        if self._syncing_energy_wavelength:
            return
        if self._focused_key() != source_key:
            return
        source_text = self._get(source_key)
        if not source_text:
            return
        try:
            source_value = float(source_text)
        except (TypeError, ValueError):
            return
        if source_value <= 0:
            return
        if source_key.endswith("wavelength"):
            target_value = ENERGY_WAVELENGTH_CONSTANT / source_value
        elif source_key.endswith("energy"):
            energy_kev = source_value / 1000.0 if source_value > 1000 else source_value
            if energy_kev <= 0:
                return
            target_value = ENERGY_WAVELENGTH_CONSTANT / energy_kev
        else:
            return
        try:
            self._syncing_energy_wavelength = True
            self._set_variable_value(target_key, self._format_derived_float(target_value))
        finally:
            self._syncing_energy_wavelength = False

    def _link_energy_wavelength_fields(self, wavelength_key, energy_key):
        self.variables[wavelength_key].trace_add(
            "write",
            lambda *_: self._sync_energy_wavelength(wavelength_key, energy_key),
        )
        self.variables[energy_key].trace_add(
            "write",
            lambda *_: self._sync_energy_wavelength(energy_key, wavelength_key),
        )

    def _predicted_peak_options(self):
        if not self._bool("plot.show_predicted_peaks"):
            return None
        source = self._get("plot.predicted_source")
        if source == "list":
            return {
                "source": "list",
                "two_theta_list": self._required_path("plot.predicted_twotheta", "Predicted 2theta list"),
            }
        return {
            "source": "lattice",
            "lattice_type": self._get("plot.predicted_lattice_type"),
            "a": self._optional_float("plot.predicted_a"),
            "b": self._optional_float("plot.predicted_b"),
            "c": self._optional_float("plot.predicted_c"),
            "wavelength": self._optional_float("plot.predicted_wavelength"),
            "energy": self._optional_float("plot.predicted_energy"),
            "max_index": self._optional_int("plot.predicted_max_index") or 8,
            "phase_name": self._get("plot.predicted_phase"),
        }

    def _stress_options(self):
        return {
            "elastic_E": self._optional_float("sin2psi.elastic_E"),
            "elastic_E_units": self._get("sin2psi.elastic_E_units") or None,
            "elastic_nu": self._optional_float("sin2psi.elastic_nu"),
            "stress_reference_two_theta": self._optional_float("sin2psi.stress_reference_two_theta"),
            "stress_reference_d0": self._optional_float("sin2psi.stress_reference_d0"),
            "stress_wavelength": self._optional_float("sin2psi.stress_wavelength"),
            "stress_energy": self._optional_float("sin2psi.stress_energy"),
        }

    def _run_task(self, title, func, on_success=None, run_on_main=False):
        if self.task_running:
            messagebox.showwarning(title, "Another task is already running. Cancel or wait for it to finish.", parent=self.master)
            return
        self.cancel_event.clear()
        self._set_task_running(True)
        self.log(f"{title}: started")
        self.set_status(f"{title}: started")
        if run_on_main:
            stream = GuiLogStream(self._emit_task_log)
            try:
                self._ensure_interactive_matplotlib()
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = func()
                stream.flush()
                if self.cancel_event.is_set():
                    self._task_cancelled(title)
                else:
                    self._task_done(title, result, "", on_success)
            except Exception as exc:
                stream.flush()
                if self.cancel_event.is_set() and "cancel" in str(exc).lower():
                    self._task_cancelled(title)
                else:
                    self._task_failed(title, exc, "", traceback.format_exc())
            return

        def worker():
            stream = GuiLogStream(self._emit_task_log)
            try:
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = func()
                stream.flush()
                self.master.after(0, lambda result=result: self._task_cancelled(title) if self.cancel_event.is_set() else self._task_done(title, result, "", on_success))
            except Exception as exc:
                stream.flush()
                details = traceback.format_exc()
                self.master.after(
                    0,
                    lambda exc=exc, details=details: self._task_cancelled(title) if self.cancel_event.is_set() and "cancel" in str(exc).lower() else self._task_failed(title, exc, "", details),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _emit_task_log(self, message):
        if threading.current_thread() is threading.main_thread():
            self.log(message)
            self.master.update_idletasks()
        else:
            self.master.after(0, lambda msg=message: self.log(msg))

    def _call_on_main_sync(self, func):
        if threading.current_thread() is threading.main_thread():
            return func()
        done = threading.Event()
        result = {"value": None, "error": None}

        def runner():
            try:
                result["value"] = func()
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()

        self.master.after(0, runner)
        done.wait()
        if result["error"] is not None:
            raise result["error"]
        return result["value"]

    def _ensure_interactive_matplotlib(self):
        try:
            import matplotlib

            if matplotlib.get_backend().lower() == "agg":
                matplotlib.use("TkAgg", force=True)
        except Exception as exc:
            self.log(f"Could not switch Matplotlib to an interactive backend: {exc}")

    def _task_done(self, title, result, output, on_success=None):
        for line in output.splitlines():
            if line.strip():
                self.log(line)
        if on_success:
            on_success(result)
        self.log(f"{title}: finished")
        self.set_status(f"{title}: finished")
        self._set_task_running(False)
        self.cancel_event.clear()

    def _task_failed(self, title, exc, output, details):
        for line in output.splitlines():
            if line.strip():
                self.log(line)
        self.log(f"{title}: failed - {exc}")
        self.set_status(f"{title}: failed")
        self._set_task_running(False)
        self.cancel_event.clear()
        messagebox.showerror(title, f"{exc}\n\nSee the command log for details.", parent=self.master)
        for line in details.splitlines()[-8:]:
            self.log(line)

    def _task_cancelled(self, title):
        self.log(f"{title}: cancelled")
        self.set_status(f"{title}: cancelled")
        self._set_task_running(False)
        self.cancel_event.clear()

    def _selected_tab_name(self):
        selected = self.notebook.select()
        if not selected:
            return TAB_NAMES[0]
        return self.notebook.tab(selected, "text")

    def _keys_for_tab(self, tab_name):
        prefixes = TAB_PREFIXES.get(tab_name, ())
        return [key for key in sorted(self.variables) if key.startswith(prefixes)]

    def _variable_value(self, key):
        if self.placeholder_active.get(key):
            return ""
        value = self.variables[key].get()
        if isinstance(value, bool):
            return value
        return "" if value is None else str(value)

    def _set_variable_value(self, key, value):
        variable = self.variables.get(key)
        if variable is None:
            return False
        if isinstance(variable, tk.BooleanVar):
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "on"}
            variable.set(bool(value))
        else:
            self.placeholder_active[key] = False
            variable.set("" if value is None else str(value))
            if self.placeholders.get(key) and not variable.get():
                self._restore_placeholder(key)
            else:
                entry = self._input_widget_for_key(key)
                if entry is not None:
                    try:
                        entry.configure(foreground="")
                    except tk.TclError:
                        pass
        return True

    def collect_parameters(self, scope="all"):
        created_at = datetime.now().isoformat(timespec="seconds")
        if scope == "current":
            tab_name = self._selected_tab_name()
            keys = self._keys_for_tab(tab_name)
            return {
                "format": "quixrd_gui_parameters",
                "version": 1,
                "scope": "current_tab",
                "tab": tab_name,
                "created_at": created_at,
                "parameters": {key: self._variable_value(key) for key in keys},
            }

        return {
            "format": "quixrd_gui_parameters",
            "version": 1,
            "scope": "all_tabs",
            "created_at": created_at,
            "tabs": {
                tab_name: {key: self._variable_value(key) for key in self._keys_for_tab(tab_name)}
                for tab_name in TAB_NAMES
            },
        }

    def export_parameters(self, path, scope="all"):
        path = Path(path)
        payload = self.collect_parameters(scope=scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.log(f"Exported {scope} parameters to {path}")
        return path

    def export_parameters_dialog(self, scope):
        path = filedialog.asksaveasfilename(
            title="Export parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.export_parameters(path, scope=scope)

    def _parameters_to_apply(self, payload, scope):
        selected_tab = self._selected_tab_name()
        if isinstance(payload, dict) and "tabs" in payload and isinstance(payload["tabs"], dict):
            if scope == "current":
                return dict(payload["tabs"].get(selected_tab, {}))
            merged = {}
            for tab_params in payload["tabs"].values():
                if isinstance(tab_params, dict):
                    merged.update(tab_params)
            return merged

        if isinstance(payload, dict) and "parameters" in payload and isinstance(payload["parameters"], dict):
            params = dict(payload["parameters"])
        elif isinstance(payload, dict):
            params = dict(payload)
        else:
            raise ValueError("Parameter file must contain a JSON object")

        if scope == "current":
            allowed = set(self._keys_for_tab(selected_tab))
            return {key: value for key, value in params.items() if key in allowed}
        return params

    def import_parameters(self, path, scope="all"):
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Could not read parameter file as text: {path}. "
                "Check that this path points to a JSON parameter file, not a plot/image file."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid parameter JSON: {path}") from exc
        params = self._parameters_to_apply(payload, scope)
        applied = 0
        for key, value in params.items():
            if self._set_variable_value(key, value):
                applied += 1
        self._refresh_dynamic_states()
        self.log(f"Imported {applied} parameter values from {path}")
        return applied

    def import_parameters_dialog(self, scope):
        path = filedialog.askopenfilename(
            title="Import parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            try:
                self.import_parameters(path, scope=scope)
            except Exception as exc:
                messagebox.showerror("Import parameters", f"Could not import parameters:\n{exc}", parent=self.master)
                self.log(f"Parameter import failed: {exc}")

    def _focused_key(self):
        focused = self.master.focus_get()
        if focused is None:
            return None
        return self.widget_keys.get(str(focused))

    def _path_keys_for_current_tab(self):
        return [key for key in self._keys_for_tab(self._selected_tab_name()) if key in self.browse_kinds]

    def _reveal_path_for_key(self, key):
        value = self._get(key) if key else ""
        if not value:
            return None
        path = Path(value)
        if path.is_file():
            return path.parent
        return path

    def _reveal_candidate_path(self):
        focused_key = self._focused_key()
        if focused_key in self.browse_kinds:
            path = self._reveal_path_for_key(focused_key)
            if path is not None:
                return path
        for key in self._path_keys_for_current_tab():
            path = self._reveal_path_for_key(key)
            if path is not None:
                return path
        return None

    def reveal_selected_path(self):
        path = self._reveal_candidate_path()
        if path is None:
            messagebox.showinfo(
                "Reveal in File Explorer",
                "Select or fill a path field on the current tab first.",
                parent=self.master,
            )
            return
        if not path.exists():
            messagebox.showerror("Reveal in File Explorer", f"Path does not exist:\n{path}", parent=self.master)
            self.log(f"Reveal failed, path does not exist: {path}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            self.log(f"Revealed in file explorer: {path}")
        except Exception as exc:
            messagebox.showerror("Reveal in File Explorer", f"Could not open file explorer:\n{exc}", parent=self.master)
            self.log(f"Reveal failed: {exc}")

    def _load_gui_settings(self):
        try:
            if self.settings_path.exists():
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_gui_settings(self):
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings["cache_root"] = str(self.cache_root)
        self.settings["use_local_cache"] = bool(self.use_local_cache_var.get())
        self.settings["twotheta_calibration_file"] = self.twotheta_calibration_file.get()
        self.settings["apply_twotheta_calibration"] = bool(self.apply_twotheta_calibration_var.get())
        self.settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")

    def _cache_root(self):
        return Path(self.cache_root)

    def set_cache_root(self, path):
        self.cache_root = Path(path)
        self._save_gui_settings()
        self.log(f"Local cache folder set to: {self.cache_root}")
        return self.cache_root

    def select_local_cache_folder(self):
        initial_dir = self._cache_root()
        initial_parent = initial_dir.parent if initial_dir.parent.exists() else Path.home()
        value = filedialog.askdirectory(
            title="Select local cache folder",
            initialdir=str(initial_dir if initial_dir.exists() else initial_parent),
        )
        if value:
            self.set_cache_root(value)

    def _use_local_cache_default(self):
        return bool(self.settings.get("use_local_cache", False))

    def _use_local_cache(self):
        return bool(self.use_local_cache_var.get())

    def _sync_use_local_cache_setting(self):
        self._save_gui_settings()
        state = "enabled" if self._use_local_cache() else "disabled"
        self.log(f"Local cache {state}.")

    def _sync_twotheta_calibration_setting(self):
        self._save_gui_settings()
        state = "enabled" if self.apply_twotheta_calibration_var.get() else "disabled"
        self.log(f"Default 2theta calibration application {state}.")

    def select_twotheta_calibration_file(self):
        initial = self.twotheta_calibration_file.get()
        initial_dir = str(Path(initial).parent) if initial else str(Path.home())
        value = filedialog.askopenfilename(
            title="Select 2theta calibration JSON",
            initialdir=initial_dir,
            filetypes=[("Calibration JSON", "*.json")],
        )
        if value:
            self._validate_json_paths(value, "2theta calibration file")
            self.twotheta_calibration_file.set(value)
            self._save_gui_settings()
            self.log(f"2theta calibration file selected: {value}")

    def _twotheta_calibration_for_processing(self):
        path = self.twotheta_calibration_file.get().strip()
        if self.apply_twotheta_calibration_var.get() and path and Path(path).exists():
            self._validate_json_paths(path, "2theta calibration file")
            return path
        return None

    def _raise_if_cancelled(self):
        if self.cancel_event.is_set():
            raise RuntimeError("Task cancelled by user")

    def _confirm_large_job(self, title, message):
        return messagebox.askokcancel(title, message, parent=self.master)

    def _cache_source_dir(self, source_dir):
        source_path = Path(source_dir)
        try:
            source_text = str(source_path.resolve())
        except Exception:
            source_text = str(source_path.absolute())
        digest = hashlib.sha1(source_text.lower().encode("utf-8")).hexdigest()[:12]
        slug = proc._safe_plot_suffix(source_path.name or "data")
        return self._cache_root() / "exported_txt" / f"{slug}_{digest}"

    def _scan_txt_sources(self, data_dir, scans):
        source_dir = Path(data_dir)
        paths = []
        missing_scans = []
        for scan in sorted({int(scan) for scan in scans}):
            matches = sorted(source_dir.glob(f"I_vs_2th_{scan}_*.txt"))
            if matches:
                paths.extend(matches)
            else:
                missing_scans.append(scan)
        unique = {}
        for path in paths:
            unique[path.name] = path
        return [unique[name] for name in sorted(unique)], missing_scans

    def _ensure_scan_txt_cache(self, data_dir, scans):
        scan_count = len(scans) if scans is not None else 0
        print(f"Local cache: checking source TXT files for {scan_count} scan(s)...", flush=True)
        source_paths, missing_scans = self._scan_txt_sources(data_dir, scans)
        cache_dir = self._cache_source_dir(data_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pending_copy = [source for source in source_paths if not (cache_dir / source.name).exists()]
        pending_reuse = len(source_paths) - len(pending_copy)
        print(
            f"Local cache: started - copying {len(pending_copy)} file(s), reusing {pending_reuse} existing file(s)"
            f" in {cache_dir}",
            flush=True,
        )
        copied = 0
        reused = 0
        for source in source_paths:
            self._raise_if_cancelled()
            target = cache_dir / source.name
            if target.exists():
                reused += 1
                continue
            shutil.copy2(source, target)
            copied += 1
        print(
            f"Local cache: completed - {copied} copied, {reused} reused"
            f" for {len(source_paths)} scan file(s) in {cache_dir}",
            flush=True,
        )
        if missing_scans:
            print(f"Local cache: no source TXT files found for scan(s): {missing_scans}", flush=True)
        return {
            "cache_dir": str(cache_dir),
            "copied": copied,
            "reused": reused,
            "missing_scans": missing_scans,
            "source_count": len(source_paths),
        }

    def _cache_copy_count(self, data_dir, scans):
        source_paths, _missing_scans = self._scan_txt_sources(data_dir, scans)
        cache_dir = self._cache_source_dir(data_dir)
        return sum(1 for source in source_paths if not (cache_dir / source.name).exists()), len(source_paths)

    def _confirm_cache_copy_if_large(self, data_dir, scans):
        copy_count, source_count = self._cache_copy_count(data_dir, scans)
        if copy_count <= LARGE_CACHE_COPY_THRESHOLD:
            return True
        return self._confirm_large_job(
            "Large Local Cache Copy",
            "This task will copy "
            f"{copy_count} TXT file(s) into the local cache before analysis.\n\n"
            f"{source_count} source TXT file(s) are needed in total. Existing cached files will be reused.\n\n"
            "This can take a while on OneDrive or network folders. Continue?",
        )

    def clear_local_cache(self):
        cache_root = self._cache_root()
        if not cache_root.exists():
            self.log(f"Local cache is already clear: {cache_root}")
            return
        proceed = messagebox.askyesno(
            "Clear Local Cache",
            f"Delete the local cache folder?\n\n{cache_root}",
            parent=self.master,
        )
        if not proceed:
            self.log("Clear Local Cache cancelled.")
            return

        def task():
            shutil.rmtree(cache_root)
            print(f"Deleted local cache: {cache_root}")
            return str(cache_root)

        self._run_task("Clear Local Cache", task)

    def _refresh_dynamic_states(self):
        if "extract.mode" in self.variables:
            self._update_extraction_mode()
        if "calibration.source_type" in self.variables:
            self._update_calibration_mode()
        if "plot.type" in self.variables and "plot.fwhm_selector" in self.variables:
            self._update_plot_mode()
        if "sort.mode" in self.variables:
            self._update_sort_mode()
        if "sin2psi.action" in self.variables:
            self._update_sin2psi_mode()

    def open_calibration_window(self):
        if self.calibration_window is not None and self.calibration_window.winfo_exists():
            self.calibration_window.lift()
            self.calibration_window.focus_set()
            self.set_status("2theta Calibration window is already open.")
            return self.calibration_window

        window = tk.Toplevel(self.master)
        window.title("quixrd 2theta Calibration")
        self._apply_window_icon(window)
        window.geometry("900x720")
        window.minsize(760, 560)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        window.protocol("WM_DELETE_WINDOW", self._close_calibration_window)
        self.calibration_window = window

        scroller = ScrollableFrame(window)
        scroller.grid(row=0, column=0, sticky="nsew")
        self._build_calibration_tab(scroller.content)
        self._update_calibration_mode()
        self.log("Opened 2theta Calibration window.")
        return window

    def _close_calibration_window(self):
        if self.calibration_window is not None and self.calibration_window.winfo_exists():
            self.calibration_window.destroy()
        self.calibration_window = None
        self.set_status("2theta Calibration window closed.")

    def _xrd_exporter(self):
        from XPAD_XRD_nxs_export import S140XRD

        flat_scans = self._parse_int_list(self._get("extract.flat_scans"), required=False)
        return S140XRD(
            nxs_file_directory=self._required_path("extract.nxs_dir", "NXS input directory"),
            export_directory=self._required_path("extract.export_dir", "Export directory"),
            flat_file_directory=self._optional_path("extract.flat_dir") or "./flat/",
            flat_file_numbers=flat_scans,
            daterange=self._parse_date_range(self._get("extract.date_range")),
        )

    def run_extraction(self):
        mode = self._get("extract.mode")

        def task():
            exporter = self._xrd_exporter()
            scans = self._selected_scan_list("extract.scans")
            show_graph = self._bool("extract.show_graph")
            save_graph = self._bool("extract.save_graph")
            mirror = self._bool("extract.mirror")
            if mode == "csv":
                if len(scans) == 1:
                    result = exporter.extract_S140XRD(scans[0], showGraph=show_graph, saveGraph=save_graph, mirror_sorted_structure=mirror)
                else:
                    result = exporter.batch_extract_S140XRD(scans, showGraph=show_graph, saveGraph=save_graph, mirror_sorted_structure=mirror)
                calibration = self._twotheta_calibration_for_processing()
                if calibration:
                    correction = tth_cal.apply_calibration_to_exported_files(self._required_path("extract.export_dir", "Export directory"), calibration, scans=scans)
                    return {"extraction": result, "twotheta_correction": correction}
                return result
            if mode == "batch":
                txt = exporter.batch_extract_S140XRD_chidelta(scans, showGraph=show_graph, saveGraph=save_graph, mirror_sorted_structure=mirror)
                csv = exporter.batch_extract_S140XRD(scans, showGraph=show_graph, saveGraph=save_graph, mirror_sorted_structure=mirror)
                result = {"txt_frames": txt, "combined_csv": csv}
                calibration = self._twotheta_calibration_for_processing()
                if calibration:
                    result["twotheta_correction"] = tth_cal.apply_calibration_to_exported_files(self._required_path("extract.export_dir", "Export directory"), calibration, scans=scans)
                return result
            if len(scans) == 1:
                result = exporter.extract_S140XRD_chidelta(scans[0], showGraph=show_graph, saveGraph=save_graph, mirror_sorted_structure=mirror)
            else:
                result = exporter.batch_extract_S140XRD_chidelta(scans, showGraph=show_graph, saveGraph=save_graph, mirror_sorted_structure=mirror)
            calibration = self._twotheta_calibration_for_processing()
            if calibration:
                correction = tth_cal.apply_calibration_to_exported_files(self._required_path("extract.export_dir", "Export directory"), calibration, scans=scans)
                return {"extraction": result, "twotheta_correction": correction}
            return result

        self._run_task("Extraction", task, run_on_main=self._bool("extract.show_graph"))

    def _calibration_input_paths(self):
        text = self._required_path("calibration.input_paths", "Calibration input path(s)")
        if ";" in text:
            paths = [part.strip().strip('"') for part in text.split(";") if part.strip()]
        else:
            paths = [text]
        return paths

    def _calibration_outlier_options(self):
        return {
            "offset_floor": self._optional_float("calibration.offset_floor"),
            "caglioti_floor": self._optional_float("calibration.caglioti_floor"),
            "max_fwhm_multiplier": self._optional_float("calibration.max_fwhm_multiplier"),
            "max_outlier_fraction": self._optional_float("calibration.max_outlier_fraction"),
        }

    def _calibration_manual_exclusions(self):
        text = self._get("calibration.manual_exclusions")
        if not text:
            return None
        return [part.strip() for part in re.split(r"[;,]\s*|\s+", text) if part.strip()]

    def _review_calibration_outliers(self, fitted_peaks, outlier_summary):
        usable = [peak for peak in fitted_peaks if peak.get("usable")]
        if not usable:
            return {"offset": [], "caglioti": []}

        def fmt(value):
            try:
                if value is None:
                    return ""
                value = float(value)
                if not math.isfinite(value):
                    return ""
                return f"{value:.5g}"
            except Exception:
                return str(value or "")

        proposed_offset = {str(peak.get("peak_index")) for peak in usable if peak.get("offset_fit_outlier")}
        proposed_caglioti = {str(peak.get("peak_index")) for peak in usable if peak.get("caglioti_fit_outlier")}

        review_fig = None
        try:
            self._ensure_interactive_matplotlib()
            import matplotlib.pyplot as plt

            offset_coeffs = tth_cal.fit_offset_polynomial(
                fitted_peaks,
                degree=self._optional_int("calibration.polynomial_degree") if self._get("calibration.polynomial_degree") else 2,
            )
            caglioti = tth_cal.fit_caglioti(fitted_peaks)
            review_fig = tth_cal._plot_fit_curves(
                None,
                fitted_peaks,
                offset_coeffs,
                caglioti,
                "Proposed calibration exclusions",
                show=False,
                save=False,
            )
            review_fig.canvas.manager.set_window_title("Proposed calibration exclusions")
            review_fig.show()
            plt.show(block=False)
            self.log("Displayed proposed calibration exclusions plot.")
        except Exception as exc:
            self.log(f"Could not create review plot: {exc}")

        self.log("Opening calibration exclusion review.")
        parent = self.calibration_window if self.calibration_window is not None and self.calibration_window.winfo_exists() else self.master
        dialog = tk.Toplevel(parent)
        dialog.title("Review Calibration Exclusions")
        dialog.transient(parent)
        dialog.geometry("980x520")
        self._apply_window_icon(dialog)
        dialog.attributes("-topmost", True)

        ttk.Label(
            dialog,
            text=(
                "Review fitted calibration peaks. Checked boxes are excluded from the corresponding final fit and "
                "recorded in the calibration JSON."
            ),
            wraplength=940,
        ).pack(anchor="w", padx=12, pady=(12, 6))

        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        canvas = tk.Canvas(frame, highlightthickness=0)
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        table = ttk.Frame(canvas)
        table.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=table, anchor="nw")
        canvas.configure(yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        headers = ["Offset", "Caglioti", "Idx", "HKL", "Expected 2theta", "Observed 2theta", "Offset", "FWHM", "Proposed"]
        widths = [8, 8, 6, 8, 16, 16, 12, 12, 18]
        for col, (header, width) in enumerate(zip(headers, widths)):
            ttk.Label(table, text=header, width=width).grid(row=0, column=col, sticky="w", padx=(0, 6), pady=(0, 4))

        offset_vars = {}
        caglioti_vars = {}
        for row_idx, peak in enumerate(usable, start=1):
            key = str(peak.get("peak_index"))
            proposed = []
            if key in proposed_offset:
                proposed.append("offset")
            if key in proposed_caglioti:
                proposed.append("Caglioti")
            offset_var = tk.BooleanVar(value=key in proposed_offset)
            caglioti_var = tk.BooleanVar(value=key in proposed_caglioti)
            offset_vars[key] = offset_var
            caglioti_vars[key] = caglioti_var
            values = [
                ttk.Checkbutton(table, variable=offset_var),
                ttk.Checkbutton(table, variable=caglioti_var),
                ttk.Label(table, text=key, width=6),
                ttk.Label(table, text=str(peak.get("hkl", "")), width=8),
                ttk.Label(table, text=fmt(peak.get("expected_two_theta")), width=16),
                ttk.Label(table, text=fmt(peak.get("center")), width=16),
                ttk.Label(table, text=fmt(peak.get("offset")), width=12),
                ttk.Label(table, text=fmt(peak.get("fwhm")), width=12),
                ttk.Label(table, text=", ".join(proposed) if proposed else "-", width=18),
            ]
            for col, widget in enumerate(values):
                widget.grid(row=row_idx, column=col, sticky="w", padx=(0, 6), pady=1)

        result = {"accepted": False, "selection": None}

        def close_review_plot():
            if review_fig is None:
                return
            try:
                import matplotlib.pyplot as plt

                plt.close(review_fig)
            except Exception:
                pass

        def accept():
            result["accepted"] = True
            result["selection"] = {
                "offset": [key for key, var in offset_vars.items() if var.get()],
                "caglioti": [key for key, var in caglioti_vars.items() if var.get()],
            }
            close_review_plot()
            dialog.destroy()

        def clear_all():
            for var in list(offset_vars.values()) + list(caglioti_vars.values()):
                var.set(False)

        def cancel():
            close_review_plot()
            dialog.destroy()

        buttons = ttk.Frame(dialog)
        buttons.pack(anchor="e", padx=12, pady=(0, 12))
        ttk.Button(buttons, text="Use Selected Exclusions", command=accept).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Clear All", command=clear_all).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(buttons, text="Cancel Calibration", command=cancel).grid(row=0, column=2)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.update_idletasks()
        dialog.lift()
        dialog.focus_set()
        dialog.after(500, lambda: dialog.attributes("-topmost", False) if dialog.winfo_exists() else None)
        self.log("Waiting for calibration exclusion review.")
        self.master.wait_window(dialog)
        if not result["accepted"]:
            raise RuntimeError("Calibration cancelled during exclusion review")

        self.log(
            f"Calibration exclusion review accepted "
            f"{len(result['selection']['offset'])} offset and {len(result['selection']['caglioti'])} Caglioti exclusions."
        )
        return result["selection"]

    def _show_image_paths(self, paths):
        paths = [Path(path) for path in paths if path and Path(path).exists()]
        if not paths:
            return
        try:
            self._ensure_interactive_matplotlib()
            import matplotlib.pyplot as plt

            for path in paths:
                image = plt.imread(path)
                height, width = image.shape[:2]
                aspect = width / max(height, 1)
                fig_width = min(13.5, max(7.0, 8.5 * aspect))
                fig_height = min(11.5, max(5.0, fig_width / max(aspect, 1e-6)))
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                ax.imshow(image)
                ax.set_axis_off()
                fig.canvas.manager.set_window_title(path.name)
                fig.tight_layout()
            plt.show(block=False)
        except Exception as exc:
            self.log(f"Could not display calibration plots: {exc}")

    def _show_calibration_result_plots(self, result):
        figs = [result.get("profile_fig"), result.get("fit_fig")] if isinstance(result, dict) else []
        figs = [fig for fig in figs if fig is not None]
        if figs:
            try:
                self._ensure_interactive_matplotlib()
                import matplotlib.pyplot as plt

                for fig, title in zip(figs, ["Calibration profile", "Calibration fits"]):
                    try:
                        fig.canvas.manager.set_window_title(title)
                    except Exception:
                        pass
                    fig.show()
                plt.show(block=False)
                return
            except Exception as exc:
                self.log(f"Could not display calibration figures: {exc}")
        paths = [result.get("profile_plot"), result.get("fit_plot")] if isinstance(result, dict) else []
        self._show_image_paths(paths)

    def _apply_generated_calibration(self, result):
        path = result.get("path") if isinstance(result, dict) else None
        if not path:
            return
        self.twotheta_calibration_file.set(path)
        self.apply_twotheta_calibration_var.set(True)
        self._save_gui_settings()
        self.log(f"2theta calibration selected and enabled: {path}")

    def run_twotheta_calibration(self):
        outlier_mode = self._get("calibration.outlier_mode") if "calibration.outlier_mode" in self.variables else "off"
        show_plots_requested = self._bool("calibration.show_plots")
        backend_show_plots = show_plots_requested and outlier_mode != "review"

        def review_callback(fitted_peaks, outlier_summary):
            self._emit_task_log("2theta Calibration: opening exclusion review")
            return self._review_calibration_outliers(fitted_peaks, outlier_summary)

        def task():
            result = tth_cal.build_twotheta_calibration(
                self._calibration_input_paths(),
                source_type=self._get("calibration.source_type"),
                output_dir=self._optional_path("calibration.output_dir"),
                material=self._get("calibration.material"),
                lattice_type=self._get("calibration.lattice_type"),
                a=self._optional_float("calibration.a"),
                b=self._optional_float("calibration.b"),
                c=self._optional_float("calibration.c"),
                energy=self._optional_float("calibration.energy"),
                wavelength=self._optional_float("calibration.wavelength"),
                polynomial_degree=self._optional_int("calibration.polynomial_degree") if self._get("calibration.polynomial_degree") else 2,
                max_index=self._optional_int("calibration.max_index") or 8,
                fit_window=self._optional_float("calibration.fit_window") or 0.35,
                discard_outliers=False,
                outlier_mode=outlier_mode,
                outlier_sensitivity=self._get("calibration.outlier_sensitivity"),
                outlier_options=self._calibration_outlier_options(),
                manual_exclusions=self._calibration_manual_exclusions(),
                outlier_review_callback=review_callback if outlier_mode == "review" else None,
                progress_callback=self._emit_task_log,
                return_figures=show_plots_requested and outlier_mode == "review",
                show_plots=backend_show_plots,
                flat_file_directory=self._optional_path("calibration.flat_dir") or "./flat/",
                flat_file_numbers=self._parse_int_list(self._get("calibration.flat_scans"), required=False),
            )
            print(f"Wrote: {result['path']}")
            print(f"Wrote: {result['combined_txt']}")
            print(f"Wrote: {result['combined_csv']}")
            print(f"Wrote: {result['profile_plot']}")
            print(f"Wrote: {result['fit_plot']}")
            return result

        self._run_task(
            "2theta Calibration",
            task,
            on_success=lambda result: (
                self._apply_generated_calibration(result),
                self._log_result_paths(result),
                self._show_calibration_result_plots(result) if show_plots_requested and outlier_mode == "review" else None,
            ),
            run_on_main=backend_show_plots or outlier_mode == "review",
        )

    def extract_and_apply_twotheta_correction(self):
        self.apply_twotheta_calibration_var.set(True)
        self._save_gui_settings()
        self.notebook.select(self.tab_frames["Extraction"])
        self.log("2theta calibration application enabled. Switched to Extraction tab.")

    def run_sorting(self):
        mode = self._get("sort.mode")

        def task():
            if mode == "extracted":
                from XPAD_XRD_nxs_export import sort_extracted_by_sample

                return sort_extracted_by_sample(
                    source_export_directory=self._required_path("sort.export_dir", "Existing export directory"),
                    sorted_nxs_directory=self._required_path("sort.nxs_dir", "NXS directory"),
                    output_directory=self._required_path("sort.output_dir", "Output directory"),
                    move=self._bool("sort.move"),
                )

            from XRD_funcs import sort_nxs_by_sample

            calibration = self._get("sort.calibrations")
            calibration_map = {"sub": "sub", "copy": True, "skip": False}
            return sort_nxs_by_sample(
                nxs_directory=self._required_path("sort.nxs_dir", "NXS directory"),
                sample_file=self._required_path("sort.sample_file", "Sample spreadsheet"),
                output_directory=self._required_path("sort.output_dir", "Output directory"),
                export_calibrations=calibration_map.get(calibration, "sub"),
            )

        self._run_task("Sorting", task)

    def run_plotting(self):
        plot_type = self._get("plot.type")
        preset_data_dir = None
        preset_scans = None
        if plot_type == "spectra" and self._use_local_cache():
            self.log("Plotting: checking scan files and local cache...")
            self.set_status("Plotting: checking scan files and local cache...")
            self.master.update_idletasks()
            preset_data_dir = self._required_path("plot.data_dir", "Export/data directory")
            preset_scans = self._selected_plot_scans(preset_data_dir, plot_type)
            self.log(f"Plotting: found {len(preset_scans)} scan(s); checking cache contents...")
            self.master.update_idletasks()
            if not self._confirm_cache_copy_if_large(preset_data_dir, preset_scans):
                self.log("Plotting cancelled before local cache copy.")
                self.set_status("Plotting: cancelled")
                return
            self.log("Plotting: preflight complete; starting plot task...")
            self.master.update_idletasks()

        def task():
            show = self._bool("plot.show_final")
            save = self._bool("plot.save_final")
            summary_csv = self._optional_path("plot.summary_csv")
            data_dir = preset_data_dir or self._required_path("plot.data_dir", "Export/data directory")
            scans = preset_scans if preset_scans is not None else self._selected_plot_scans(data_dir, plot_type)
            if plot_type == "spectra":
                from XRD_spectra_anal import Spectrum

                spectrum_dir = data_dir
                if self._use_local_cache():
                    cache_info = self._ensure_scan_txt_cache(data_dir, scans)
                    spectrum_dir = cache_info["cache_dir"]
                spectrum = Spectrum(directory=spectrum_dir)
                return spectrum.plot_Ivs2theta(
                    scanNos=scans,
                    plot_only=self._selected_plot_scan_types(),
                    offset=self._optional_float("plot.offset") or 0.0,
                    label=self._selected_spectra_labels(),
                    predicted_peaks=self._predicted_peak_options(),
                    show_plot=show,
                    save_plot=save,
                    save_directory=str(self._manual_plot_save_dir()),
                    twotheta_calibration_json=self._twotheta_calibration_for_processing(),
                )
            if plot_type == "gradient":
                return proc.plot_sin2psi_gradients(
                    data_dir,
                    x=self._get("plot.x"),
                    scans=scans,
                    save=save,
                    show=show,
                    summary_csv=summary_csv,
                )
            if plot_type == "stress":
                return proc.plot_sin2psi_stress(
                    data_dir,
                    x=self._get("plot.x"),
                    scans=scans,
                    save=save,
                    show=show,
                    summary_csv=summary_csv,
                )

            frame_index = None
            chi = None
            if self._get("plot.fwhm_selector") == "frame":
                frame_index = self._parse_int_or_list(self._get("plot.fwhm_frame"))
            else:
                chi = self._parse_float_or_list(self._get("plot.fwhm_chi"))
            if frame_index is None and chi is None:
                raise ValueError("Specify either frame number(s) or chi value(s)")
            if plot_type == "peak_position":
                return proc.plot_peak_position_trends(data_dir, x=self._get("plot.x"), scans=scans, frame_index=frame_index, chi=chi, save=save, show=show)
            return proc.plot_fwhm_trends(data_dir, x=self._get("plot.x"), scans=scans, frame_index=frame_index, chi=chi, save=save, show=show)

        self._run_task("Plotting", task, on_success=self._log_result_paths, run_on_main=plot_type == "spectra" or self._bool("plot.show_final"))

    def run_peak_action(self):
        if self._get("peak.action") == "replot":
            self.replot_peak_analysis()
        else:
            self.run_peak_analysis()

    def run_peak_analysis(self):
        def progress(message):
            if threading.current_thread() is threading.main_thread():
                self.log(message)
                self.master.update_idletasks()
            else:
                self.master.after(0, lambda msg=message: self.log(msg))

        def resolve_peak_scans():
            data_dir = self._required_path("peak.data_dir", "Data directory")
            scan_type = self._get("peak.scan_type") or None
            frame_index = (self._optional_int("peak.frame_index") if self._get("peak.frame_index") else 0) if scan_type == "chi" else None
            peak_scan_text = self._get("peak.scans")
            every_nth = self._parse_every_nth(peak_scan_text)
            explicit = self._parse_int_list(peak_scan_text, required=False)
            discovered = peak_analysis.discover_scan_numbers(data_dir, scan_type=scan_type, frame_index=frame_index)
            if explicit:
                discovered_set = set(discovered)
                scans = [scan for scan in explicit if scan in discovered_set]
            else:
                scans = self._apply_every_nth(discovered, every_nth)
            if not scans:
                raise ValueError(f"No matching Peak Analysis scans found in {data_dir}")
            return data_dir, scan_type, frame_index, scans

        try:
            resolved_data_dir, resolved_scan_type, resolved_frame_index, resolved_scans = resolve_peak_scans()
        except Exception as exc:
            messagebox.showerror("Peak Analysis", str(exc), parent=self.master)
            self.log(f"Peak Analysis setup failed: {exc}")
            return

        def task():
            return peak_analysis.run_peak_series(
                data_dir=resolved_data_dir,
                scans=resolved_scans,
                scan_type=resolved_scan_type,
                frame_index=resolved_frame_index,
                peak_center=self._optional_float("peak.center"),
                fit_window=self._optional_float("peak.window") or 0.5,
                fit_mode=self._get("peak.fit_mode"),
                split_guess=self._optional_float("peak.split_guess"),
                x=self._get("peak.x") or "scan_number",
                save=True,
                show=False,
                diagnostic_all_fits=self._bool("peak.diagnostic_all_fits"),
                progress_callback=progress,
                cancel_check=self.cancel_event.is_set,
                secondary_y=self._get("peak.secondary_y"),
                plot_mode=self._get("peak.fit_mode"),
            )

        scan_count = len(resolved_scans)
        if scan_count > LARGE_FIT_SCAN_THRESHOLD and not self._confirm_large_job(
            "Large Peak Analysis Job",
            f"Peak Analysis will fit {scan_count} scan(s).\n\n"
            "This can take several minutes. You can use Cancel Task to stop between scans, but the current fit must finish first.\n\n"
            "Continue?",
        ):
            self.log("Peak Analysis cancelled before fitting.")
            return

        def on_success(result):
            self._log_result_paths(result)
            if self._bool("peak.show_final") and isinstance(result, dict) and result.get("csv_path"):
                self._show_peak_trend_from_csv(result["csv_path"])

        self._run_task(
            "Peak Analysis",
            task,
            on_success=on_success,
            run_on_main=False,
        )

    def _show_peak_trend_from_csv(self, csv_path):
        try:
            self._ensure_interactive_matplotlib()
            result = peak_analysis.plot_peak_series_from_csv(
                csv_path,
                x=self._get("peak.x") or "scan_number",
                save=False,
                show=True,
                secondary_y=self._get("peak.secondary_y"),
                plot_mode=self._get("peak.fit_mode"),
            )
            import matplotlib.pyplot as plt

            plt.show(block=False)
            if result.get("plot_path"):
                self.log(f"Displayed Peak Analysis trend plot: {result['plot_path']}")
            else:
                self.log("Displayed Peak Analysis trend plot.")
        except Exception as exc:
            self.log(f"Could not display Peak Analysis trend plot: {exc}")

    def replot_peak_analysis(self):
        csv_path = self._required_path("peak.results_csv", "Peak Analysis results CSV")

        def task():
            return peak_analysis.plot_peak_series_from_csv(
                csv_path,
                x=self._get("peak.x") or "scan_number",
                save=True,
                show=False,
                secondary_y=self._get("peak.secondary_y"),
                plot_mode=self._get("peak.fit_mode"),
            )

        def on_success(result):
            self._log_result_paths(result)
            if self._bool("peak.show_final"):
                self._show_peak_trend_from_csv(csv_path)

        self._run_task(
            "Peak Analysis Replot",
            task,
            on_success=on_success,
            run_on_main=False,
        )

    def run_williamson_hall(self):
        def progress(message):
            if threading.current_thread() is threading.main_thread():
                self.log(message)
                self.master.update_idletasks()
            else:
                self.master.after(0, lambda msg=message: self.log(msg))

        try:
            data_dir = self._required_path("wh.data_dir", "Data directory")
            profile_source = self._get("wh.profile_source") or "txt"
            scan_type = (self._get("wh.scan_type") or None) if profile_source == "txt" else None
            frame_index = None
            if profile_source == "csv" or scan_type == "chi":
                frame_index = self._optional_int("wh.frame_index")
                if frame_index is None and profile_source == "txt":
                    frame_index = 0
            scans = self._selected_wh_scans(data_dir, profile_source, scan_type, frame_index)
        except Exception as exc:
            messagebox.showerror("Williamson-Hall", str(exc), parent=self.master)
            self.log(f"Williamson-Hall setup failed: {exc}")
            return

        target_source = self._get("wh.target_source") or "lattice"
        calibration_json = self._optional_json_path("wh.twotheta_calibration_json", "2theta calibration JSON") or self._twotheta_calibration_for_processing()

        def task():
            return wh.run_williamson_hall_series(
                data_dir=data_dir,
                scans=scans,
                profile_source=profile_source,
                scan_type=scan_type,
                frame_index=frame_index,
                target_source=target_source,
                manual_two_theta=self._get("wh.manual_two_theta"),
                lattice_type=self._get("wh.lattice_type") or "cubic",
                a=self._optional_float("wh.a"),
                b=self._optional_float("wh.b"),
                c=self._optional_float("wh.c"),
                wavelength=self._optional_float("wh.wavelength"),
                energy=self._optional_float("wh.energy"),
                max_index=self._optional_int("wh.max_index") or 8,
                phase_name=self._get("wh.phase_name"),
                thermal_alpha=self._optional_float("wh.thermal_alpha"),
                reference_temperature=self._optional_float("wh.reference_temperature"),
                fit_window=self._optional_float("wh.fit_window") or 0.35,
                shape_factor=self._optional_float("wh.shape_factor") or 0.9,
                twotheta_calibration_json=calibration_json,
                residual_shift_limit=self._optional_float("wh.residual_shift_limit") or 0.15,
                registration_window=self._optional_float("wh.registration_window") or 2.0,
                x=self._get("wh.x") or "scan_number",
                secondary_y=self._get("wh.secondary_y"),
                save=True,
                show=False,
                progress_callback=progress,
                cancel_check=self.cancel_event.is_set,
            )

        scan_count = len(scans)
        if scan_count > LARGE_FIT_SCAN_THRESHOLD and not self._confirm_large_job(
            "Large Williamson-Hall Job",
            f"Williamson-Hall analysis will fit multiple peaks in {scan_count} scan(s).\n\n"
            "This can take several minutes. You can use Cancel Task to stop between scans, but the current peak fit must finish first.\n\n"
            "Continue?",
        ):
            self.log("Williamson-Hall cancelled before fitting.")
            return

        def on_success(result):
            self._log_result_paths(result)
            if self._bool("wh.show_final") and isinstance(result, dict) and result.get("summary_path"):
                self._show_wh_trend_from_csv(result["summary_path"])

        self._run_task("Williamson-Hall", task, on_success=on_success, run_on_main=True)

    def _show_wh_trend_from_csv(self, csv_path):
        try:
            self._ensure_interactive_matplotlib()
            wh.plot_wh_trends_from_csv(
                csv_path,
                x=self._get("wh.x") or "scan_number",
                save=False,
                show=True,
                secondary_y=self._get("wh.secondary_y"),
            )
            import matplotlib.pyplot as plt

            plt.show(block=False)
            self.log("Displayed Williamson-Hall trend plot.")
        except Exception as exc:
            self.log(f"Could not display Williamson-Hall trend plot: {exc}")

    def _manual_plot_save_dir(self):
        summary_csv = self._optional_path("plot.summary_csv")
        if summary_csv:
            return Path(summary_csv).parent / "saved_plots"
        data_dir = self._required_path("plot.data_dir", "Export/data directory")
        return Path(data_dir) / "saved_plots"

    def save_current_plot(self):
        try:
            import matplotlib.pyplot as plt

            fig_numbers = plt.get_fignums()
            if not fig_numbers:
                messagebox.showinfo(
                    "Save Current Plot",
                    "No open Matplotlib figure was found. Run a plot with Show final plot enabled first.",
                    parent=self.master,
                )
                self.log("Save Current Plot: no open figure found.")
                return None
            fig = plt.figure(fig_numbers[-1])
            output_dir = self._manual_plot_save_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"manual_plot_{stamp}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            self.log(f"Saved current plot: {output_path}")
            return output_path
        except Exception as exc:
            messagebox.showerror("Save Current Plot", f"Could not save current plot:\n{exc}", parent=self.master)
            self.log(f"Save Current Plot failed: {exc}")
            return None

    def _sin2psi_common(self):
        action = self._get("sin2psi.action")
        return {
            "data_dir": self._required_path("sin2psi.data_dir", "Data directory"),
            "scans": self._selected_sin2psi_scans(action),
            "use_cache": self._use_local_cache(),
            "exclude_frames": self._parse_int_list(self._get("sin2psi.exclude_frames"), required=False),
            "exclude_chi_ranges": self._parse_ranges(self._get("sin2psi.exclude_chi")),
            "exclude_sin2psi_ranges": self._parse_ranges(self._get("sin2psi.exclude_sin2psi")),
            "auto_exclude": self._bool("sin2psi.auto_exclude"),
            "correction_json": self._optional_json_path("sin2psi.correction_json", f"{SIN2PSI_LABEL} correction JSON"),
            "show": self._bool("sin2psi.show_final"),
        }

    def _process_sin2psi_scans(self, common, scans):
        results = []
        source_dir = common["data_dir"]
        read_dir = source_dir
        if common.get("use_cache"):
            cache_info = self._ensure_scan_txt_cache(source_dir, scans)
            read_dir = cache_info["cache_dir"]
        for scan in scans:
            self._raise_if_cancelled()
            files = proc.discover_scan_files(read_dir, scan)
            if not files:
                raise FileNotFoundError(f"No matching scan files found for scan {scan}")
            result = proc.process_scan(
                data_dir=source_dir,
                scan_number=scan,
                files=files,
                exclude_frames=common["exclude_frames"],
                exclude_chi_ranges=common["exclude_chi_ranges"],
                exclude_sin2psi_ranges=common["exclude_sin2psi_ranges"],
                auto_exclude=common["auto_exclude"],
                correction_json=common["correction_json"],
                plot_frames=self._bool("sin2psi.plot_frames"),
                force=True,
                peak_center=self._optional_float("sin2psi.peak_center"),
                track_peak=self._bool("sin2psi.track_peak"),
                track_window=float(self._get("sin2psi.track_window") or 1.0),
                twotheta_calibration_json=self._twotheta_calibration_for_processing(),
                **self._stress_options(),
            )
            print(f"Wrote: {result['csv_path']}")
            print(f"Wrote: {result['scan_dir']}")
            results.append(result)
        return results

    def _refit_sin2psi_scans(self, common, scans):
        results = []
        for scan in scans:
            self._raise_if_cancelled()
            result = proc.refit_sin2psi_from_csv(
                common["data_dir"],
                scan,
                excluded_frames=common["exclude_frames"],
                exclude_chi_ranges=common["exclude_chi_ranges"],
                exclude_sin2psi_ranges=common["exclude_sin2psi_ranges"],
                auto_exclude=common["auto_exclude"],
                correction_json=common["correction_json"],
                **self._stress_options(),
            )
            print(f"Updated: {result['csv_path']}")
            print(f"Updated: {result['scan_dir']}")
            results.append(result)
        return results

    def run_sin2psi_action(self, scans_override=None):
        action = self._get("sin2psi.action")
        try:
            if scans_override is not None:
                warning_scans = list(scans_override)
            else:
                warning_scans = self._selected_sin2psi_scans(action)
            warning_count = len(warning_scans) if warning_scans is not None else 0
            if action in {"process", "refit"} and warning_count > LARGE_FIT_SCAN_THRESHOLD:
                if not self._confirm_large_job(
                    f"Large {SIN2PSI_LABEL} Job",
                    f"{SIN2PSI_LABEL} {action} will run on {warning_count} discovered scan(s).\n\n"
                    "This can take several minutes. You can use Cancel Task to stop between scans, but the current scan must finish first.\n\n"
                    "Continue?",
                ):
                    self.log(f"{SIN2PSI_LABEL} {action} cancelled before fitting.")
                    return
            if action == "process" and self._use_local_cache() and warning_scans:
                data_dir = self._required_path("sin2psi.data_dir", "Data directory")
                if not self._confirm_cache_copy_if_large(data_dir, warning_scans):
                    self.log(f"{SIN2PSI_LABEL} process cancelled before local cache copy.")
                    return
        except Exception as exc:
            if scans_override is None:
                messagebox.showerror(f"{SIN2PSI_LABEL} {action}", str(exc), parent=self.master)
                self.log(f"{SIN2PSI_LABEL} setup failed: {exc}")
                return

        def task():
            common = self._sin2psi_common()
            scans = list(scans_override) if scans_override is not None else common["scans"]
            if action == "process":
                return self._process_sin2psi_scans(common, scans)
            if action == "refit":
                return self._refit_sin2psi_scans(common, scans)
            if action == "correction":
                result = proc.generate_sin2psi_correction(
                    self._required_path("sin2psi.reference_folder", "Reference folder"),
                    int(self._required_path("sin2psi.reference_scan", "Reference scan")),
                    degree=self._optional_int("sin2psi.correction_degree") or 2,
                    method=self._get("sin2psi.correction_method"),
                    reference_two_theta=self._optional_float("sin2psi.reference_two_theta"),
                    exclude_chi_ranges=common["exclude_chi_ranges"],
                    exclude_sin2psi_ranges=common["exclude_sin2psi_ranges"],
                    excluded_frames=common["exclude_frames"],
                )
                print(f"Wrote: {result['path']}")
                print(f"Wrote: {result['plot_path']}")
                return result

            gradient = proc.plot_sin2psi_gradients(
                common["data_dir"],
                x=self._get("sin2psi.summary_x") or "scan_number",
                scans=scans,
                show=common["show"],
                secondary_y=self._get("sin2psi.summary_secondary_y"),
            )
            print(f"Wrote: {gradient['summary_path']}")
            print(f"Wrote: {gradient['plot_path']}")
            return gradient

        self._run_task(
            f"{SIN2PSI_LABEL} {action}",
            task,
            on_success=self._log_result_paths,
            run_on_main=action in {"process", "refit", "correction", "summaries"},
        )

    def _log_result_paths(self, result):
        if isinstance(result, dict):
            for key in (
                "summary_path",
                "plot_path",
                "diagnostic_plot_path",
                "path",
                "csv_path",
                "peaks_path",
                "params_path",
                "scan_dir",
                "output_dir",
                "combined_txt",
                "combined_csv",
                "profile_plot",
                "fit_plot",
            ):
                if result.get(key):
                    self.log(f"{key}: {result[key]}")
            for key in ("success_count", "failed_count"):
                if key in result:
                    self.log(f"{key}: {result[key]}")
            if result.get("diagnostic_plot_paths"):
                paths = result["diagnostic_plot_paths"]
                self.log(f"diagnostic_plot_files: {len(paths)}")
                for path in paths[:5]:
                    self.log(f"diagnostic_plot_file: {path}")
                if len(paths) > 5:
                    self.log(f"diagnostic_plot_file: ... {len(paths) - 5} more")
            if result.get("wh_plot_paths"):
                paths = result["wh_plot_paths"]
                self.log(f"williamson_hall_plot_files: {len(paths)}")
                for path in paths[:5]:
                    self.log(f"williamson_hall_plot_file: {path}")
                if len(paths) > 5:
                    self.log(f"williamson_hall_plot_file: ... {len(paths) - 5} more")

    def preview_sin2psi(self, command_name="Preview First Scan"):
        action = self._get("sin2psi.action")
        if action not in {"process", "refit"}:
            messagebox.showinfo("Preview", "Preview is available for Process peaks and Refit trends only.", parent=self.master)
            return
        scans = self._selected_sin2psi_scans(action)
        first_scan = scans[0]
        remaining = scans[1:]

        def after_preview(_result):
            if not remaining:
                return
            proceed = messagebox.askyesno(
                "Continue after preview?",
                f"Preview scan {first_scan} finished. Continue with {len(remaining)} remaining scan(s)?",
                parent=self.master,
            )
            if proceed:
                self.run_sin2psi_action(scans_override=remaining)
            else:
                self.log("Preview confirmation: abort selected.")

        self._run_task(f"{SIN2PSI_LABEL} preview scan {first_scan}", lambda: self.run_sin2psi_action_sync([first_scan]), on_success=after_preview)

    def run_sin2psi_action_sync(self, scans):
        action = self._get("sin2psi.action")
        common = self._sin2psi_common()
        if action == "process":
            return self._process_sin2psi_scans(common, scans)
        if action == "refit":
            return self._refit_sin2psi_scans(common, scans)
        raise ValueError("Preview is available for Process peaks and Refit trends only")

    def _build_extraction_tab(self, parent):
        paths = self._section(parent, "Paths", 0)
        self._entry(paths, 0, "NXS input directory", "extract.nxs_dir", "Folder containing .nxs files.", "directory")
        self._entry(paths, 1, "Export directory", "extract.export_dir", "Folder where TXT/CSV/PNG exports will be written.", "directory")
        self._entry(paths, 2, "Flat directory", "extract.flat_dir", "Folder containing flat-field NXS files.", "directory", optional=True)
        self._entry(paths, 3, "Flat scan numbers", "extract.flat_scans", "Comma-separated flat scan numbers, e.g. 39,40.", optional=True)

        scans = self._section(parent, "Scans and Options", 1)
        mode = self._radio_group(
            scans,
            0,
            "Extraction mode",
            "extract.mode",
            [
                ("txt", "TXT frames", "Export one TXT intensity file per detector frame."),
                ("csv", "Combined CSV", "Export one combined CSV for the selected scan."),
                ("batch", "Batch", "Run extraction over multiple scans or a scan range."),
            ],
            "txt",
        )
        self._entry(
            scans,
            1,
            "Scan range/list",
            "extract.scans",
            "Single scan, comma list, range, or stepped range. Use 1500-1520:5 to take every fifth scan.",
            placeholder="e.g. 1515,1517,1520 or 1500-1520:5",
        )
        self._entry(scans, 2, "Date range", "extract.date_range", "Date range used for date-folder NXS lookup, e.g. 20260609-20260615.", optional=True)
        self._checkbox(scans, 3, "Mirror sorted structure", "extract.mirror", "Mirror sample-sorted NXS folders in the export directory.", optional=True)
        self._checkbox(scans, 4, "Save graph", "extract.save_graph", "Save extraction overview PNGs.", optional=True)
        self._checkbox(scans, 5, "Show graph", "extract.show_graph", "Display extraction plots interactively.", optional=True)
        mode.trace_add("write", lambda *_: self._update_extraction_mode())
        self._update_extraction_mode()
        self._action_button(parent, 2, "Run Selected Extraction", command=self.run_extraction)

    def _build_calibration_tab(self, parent):
        inputs = self._section(parent, "2theta Calibration Inputs", 0)
        source = self._radio_group(
            inputs,
            0,
            "Source type",
            "calibration.source_type",
            [
                ("txt", "TXT frames", "Use one or more exported I_vs_2th delta-frame TXT files."),
                ("csv", "All-frame CSV", "Use one combined single-scan CSV export."),
                ("nxs", "Raw NXS", "Export one SOLEIL XPAD delta-scan NXS file before calibration."),
            ],
            "txt",
        )
        self._entry(
            inputs,
            1,
            "Input path(s)",
            "calibration.input_paths",
            "For TXT frames, Browse can select multiple files. For CSV/NXS, select one file.",
            "calibration_input",
            placeholder=r"C:\path\I_vs_2th_100_delta_0.txt; C:\path\I_vs_2th_100_delta_1.txt",
        )
        self._entry(inputs, 2, "Output folder", "calibration.output_dir", "Folder for calibration JSON, plots, and combined profile. Blank creates a calibration folder beside the input.", "directory", optional=True)
        self._entry(inputs, 3, "Flat directory", "calibration.flat_dir", "Flat-field folder used only for raw NXS calibration input.", "directory", optional=True)
        self._entry(inputs, 4, "Flat scan numbers", "calibration.flat_scans", "Flat scan numbers used only for raw NXS calibration input.", optional=True)

        material = self._section(parent, "Material and Energy", 1)
        self._combo(material, 0, "Calibration material", "calibration.material", ["LaB6 (cubic, Pm-3m)", "custom"], "Standard material used to calculate expected peak positions.", "LaB6 (cubic, Pm-3m)")
        self._combo(material, 1, "Lattice type", "calibration.lattice_type", ["cubic", "fcc", "bcc", "hcp", "tetragonal", "orthorhombic"], "Lattice type for custom calibration material.", "cubic", optional=True)
        self._entry(material, 2, "a", "calibration.a", "Lattice parameter a in Angstrom.", default=str(tth_cal.LAB6_A))
        self._entry(material, 3, "b", "calibration.b", "Lattice parameter b in Angstrom where needed.", optional=True)
        self._entry(material, 4, "c", "calibration.c", "Lattice parameter c in Angstrom where needed.", optional=True)
        self._entry(material, 5, "Wavelength", "calibration.wavelength", "Wavelength in Angstrom. Leave blank to use energy metadata.", optional=True)
        self._entry(material, 6, "Energy", "calibration.energy", "Energy in keV or eV. Leave blank to use scan metadata.", optional=True)

        fitting = self._section(parent, "Fitting", 2)
        self._entry(fitting, 0, "Polynomial degree", "calibration.polynomial_degree", "Degree for offset-vs-2theta polynomial; 0 is allowed.", default="2")
        self._entry(fitting, 1, "Max hkl index", "calibration.max_index", "Maximum h/k/l index to enumerate.", default="8")
        self._entry(fitting, 2, "Fit window", "calibration.fit_window", "Half-width in 2theta degrees around each predicted peak.", default="1.00")
        self._combo(
            fitting,
            3,
            "Outlier mode",
            "calibration.outlier_mode",
            ["review", "automatic", "manual", "off"],
            "review: propose automatic exclusions and let you confirm/edit. automatic: apply thresholds directly. manual: use typed exclusions only. off: use all usable peaks.",
            "review",
        )
        self._combo(
            fitting,
            4,
            "Outlier sensitivity",
            "calibration.outlier_sensitivity",
            ["conservative", "normal", "aggressive"],
            "Controls automatic residual thresholds. Conservative excludes fewer points; aggressive excludes more.",
            "normal",
            optional=True,
        )
        normal_preset = tth_cal.OUTLIER_SENSITIVITY_PRESETS["normal"]
        self._entry(fitting, 5, "Offset floor", "calibration.offset_floor", "Advanced: minimum offset residual in 2theta degrees before automatic exclusion.", optional=True, default=f"{normal_preset['offset_floor']:g}")
        self._entry(fitting, 6, "FWHM floor", "calibration.caglioti_floor", "Advanced: minimum FWHM residual in degrees before automatic Caglioti exclusion.", optional=True, default=f"{normal_preset['caglioti_floor']:g}")
        self._entry(fitting, 7, "Max FWHM multiplier", "calibration.max_fwhm_multiplier", "Advanced: broad peaks above this multiple of median FWHM can be proposed for exclusion.", optional=True, default=f"{normal_preset['max_fwhm_multiplier']:g}")
        self._entry(fitting, 8, "Max excluded fraction", "calibration.max_outlier_fraction", "Advanced: maximum fraction of peaks automatic exclusion may remove.", optional=True, default=f"{normal_preset['max_outlier_fraction']:g}")
        self._entry(fitting, 9, "Manual exclusions", "calibration.manual_exclusions", "Manual mode only: HKL keys or peak indices separated by commas/spaces, e.g. 011, 222, 7.", optional=True, placeholder="e.g. 011, 222, 7")
        self._checkbox(fitting, 10, "Show plots", "calibration.show_plots", "Display generated calibration plots interactively after fitting.", optional=True, default=True)

        source.trace_add("write", lambda *_: self._update_calibration_mode())
        self.variables["calibration.material"].trace_add("write", lambda *_: self._update_calibration_mode())
        self.variables["calibration.lattice_type"].trace_add("write", lambda *_: self._update_calibration_mode())
        self.variables["calibration.outlier_mode"].trace_add("write", lambda *_: self._update_calibration_mode())
        self.variables["calibration.outlier_sensitivity"].trace_add("write", lambda *_: self._autofill_calibration_outlier_preset())
        self._link_energy_wavelength_fields("calibration.wavelength", "calibration.energy")
        self._update_calibration_mode()
        buttons = ttk.Frame(parent)
        buttons.grid(row=3, column=0, sticky="w", pady=(0, 10))
        ttk.Button(buttons, text="Generate Calibration", command=self.run_twotheta_calibration).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Extract with Correction", command=self.extract_and_apply_twotheta_correction).grid(row=0, column=1)

    def _build_plotting_tab(self, parent):
        options = self._section(parent, "Plot Inputs", 0)
        self._entry(options, 0, "Export/data directory", "plot.data_dir", "Folder containing exported data or sin2psi_export.", "directory")
        self._entry(
            options,
            1,
            "Scan range/list",
            "plot.scans",
            "Single scan, comma list, range, stepped range, or :n for every nth discovered scan.",
            optional=True,
            placeholder="blank for all, :5, or 440-500:5",
        )
        plot_type = self._radio_group(
            options,
            2,
            "Plot type",
            "plot.type",
            [
                ("spectra", "Spectra", "Plot exported intensity spectra from TXT/CSV data."),
                ("gradient", "Gradient", f"Plot {SIN2PSI_LABEL} gradient versus scan number or metadata."),
                ("stress", "Stress", f"Plot calculated {SIN2PSI_LABEL} stress versus scan number or metadata."),
                ("fwhm", "FWHM", "Plot fitted peak FWHM versus scan number or metadata."),
                ("peak_position", "Peak position", "Plot fitted peak position versus scan number or metadata."),
            ],
            "spectra",
        )
        self._scan_type_checkboxes(options, 3)
        self._spectra_label_checkboxes(options, 4)
        self._entry(options, 5, "Offset", "plot.offset", "Vertical offset multiplier for spectra traces.", default="0.0", optional=True)
        self._combo(
            options,
            6,
            "X/metadata field",
            "plot.x",
            X_METADATA_OPTIONS,
            "Metadata column for trend x-axis.",
            "scan_number",
        )
        self._entry(
            options,
            7,
            "Summary CSV",
            "plot.summary_csv",
            "Optional existing sin2psi scan summary CSV to plot for Gradient or Stress. Leave blank to collect/reuse the current latest summary.",
            "file",
            optional=True,
        )
        self._radio_group(
            options,
            8,
            "FWHM selector",
            "plot.fwhm_selector",
            [
                ("frame", "Frame number", "Select the FWHM row with this exact frame index in each scan."),
                ("chi", "Chi value", "Select rows whose chi value is within 0.1 degrees of the requested value."),
            ],
            "frame",
        )
        self._entry(options, 9, "Frame number(s)", "plot.fwhm_frame", "Frame index or comma/range list used for FWHM or peak-position trends.")
        self._entry(options, 10, "Chi value(s)", "plot.fwhm_chi", "Chi value or comma list used for FWHM or peak-position trends; matches within 0.1 degrees.")
        self._checkbox(options, 11, "Show final plot", "plot.show_final", "Display the final plot interactively.", optional=True, default=True)
        self._checkbox(options, 11, "Save final plot", "plot.save_final", "Automatically save the final plot when Run Selected Plot is used.", column=1, optional=True)
        self._checkbox(options, 12, "Show predicted peaks", "plot.show_predicted_peaks", "Overlay predicted peak positions on spectra.", optional=True)
        self._radio_group(
            options,
            13,
            "Peak source",
            "plot.predicted_source",
            [
                ("list", "2theta list", "Use manually typed 2theta positions."),
                ("lattice", "Lattice parameters", "Calculate peak positions from simple lattice parameters."),
            ],
            "list",
        )
        self._entry(options, 14, "2theta peaks", "plot.predicted_twotheta", "Comma-separated peak positions, optionally with labels.", optional=True)
        self._combo(options, 15, "Lattice type", "plot.predicted_lattice_type", ["cubic", "fcc", "bcc", "hcp", "tetragonal", "orthorhombic"], "Structure/lattice type for peak prediction.", "fcc", optional=True)
        self._entry(options, 16, "a", "plot.predicted_a", "Lattice parameter a in Angstrom.", optional=True)
        self._entry(options, 17, "b", "plot.predicted_b", "Lattice parameter b in Angstrom.", optional=True)
        self._entry(options, 18, "c", "plot.predicted_c", "Lattice parameter c in Angstrom.", optional=True)
        self._entry(options, 19, "Wavelength", "plot.predicted_wavelength", "Wavelength in Angstrom. Leave blank to use energy.", optional=True)
        self._entry(options, 20, "Energy", "plot.predicted_energy", "Energy in keV or eV. Leave blank to use scan metadata when available.", optional=True)
        self._entry(options, 21, "Max hkl index", "plot.predicted_max_index", "Maximum h/k/l index to enumerate.", default="8", optional=True)
        self._entry(options, 22, "Phase label", "plot.predicted_phase", "Optional phase name prepended to hkl labels.", optional=True)
        plot_type.trace_add("write", lambda *_: self._update_plot_mode())
        self.variables["plot.fwhm_selector"].trace_add("write", lambda *_: self._update_plot_mode())
        self.variables["plot.show_predicted_peaks"].trace_add("write", lambda *_: self._update_plot_mode())
        self.variables["plot.predicted_source"].trace_add("write", lambda *_: self._update_plot_mode())
        self.variables["plot.predicted_lattice_type"].trace_add("write", lambda *_: self._update_plot_mode())
        self._link_energy_wavelength_fields("plot.predicted_wavelength", "plot.predicted_energy")
        self._update_plot_mode()
        self._plot_buttons(parent, 1)

    def _build_sorting_tab(self, parent):
        paths = self._section(parent, "Sorting Inputs", 0)
        self._entry(paths, 0, "NXS directory", "sort.nxs_dir", "Unsorted NXS directory.", "directory")
        self._entry(paths, 1, "Sample spreadsheet", "sort.sample_file", "Spreadsheet used by sort_nxs_by_sample.", "file")
        self._entry(paths, 2, "Output directory", "sort.output_dir", "Destination for sorted NXS folders.", "directory")
        self._entry(paths, 3, "Existing export directory", "sort.export_dir", "Existing exported TXT/CSV/PNG files to sort by sample.", "directory")
        sort_mode = self._radio_group(
            paths,
            4,
            "Sort mode",
            "sort.mode",
            [
                ("nxs", "NXS by sample", "Sort raw NXS files into sample folders using the spreadsheet."),
                ("extracted", "Extracted data", "Copy or move existing exported TXT/CSV/PNG files into sample folders."),
            ],
            "nxs",
        )
        self._combo(paths, 5, "Calibration handling", "sort.calibrations", ["sub", "copy", "skip"], "How calibration scans should be handled.", "sub", optional=True)
        self._checkbox(paths, 6, "Move extracted files", "sort.move", "Move instead of copy when sorting already-extracted data.", optional=True)
        sort_mode.trace_add("write", lambda *_: self._update_sort_mode())
        self._update_sort_mode()
        self._action_button(parent, 1, "Run Selected Sorting", command=self.run_sorting)

    def _build_peak_analysis_tab(self, parent):
        controls = self._section(parent, "Peak Analysis Mode", 0)
        action = self._radio_group(
            controls,
            0,
            "Action",
            "peak.action",
            [
                ("run", "Run analysis", "Fit peak models from exported TXT files."),
                ("replot", "Replot existing results", "Read a saved peak_series CSV and redraw the trend plot without refitting."),
            ],
            default="run",
        )
        self._radio_group(
            controls,
            1,
            "Peak plot model",
            "peak.fit_mode",
            [
                ("compare", "1 vs 2", "Fit/show both one-peak and two-peak models, including comparison metrics when available."),
                ("single", "1 peak", "Fit/show only the one-peak model."),
                ("two", "2 peaks", "Fit/show only the two-peak model."),
            ],
            default="compare",
        )
        self._combo(
            controls,
            2,
            "X/metadata field",
            "peak.x",
            X_METADATA_OPTIONS,
            "Trend x-axis. scan_number is always available; metadata fields work when present in TXT headers or the results CSV.",
            "scan_number",
        )
        self._combo(
            controls,
            3,
            "Second y-axis",
            "peak.secondary_y",
            PEAK_SECONDARY_Y_OPTIONS,
            "Optional right-hand y-axis for Peak Analysis trend plots. You can type any numeric column in the peak_series CSV.",
            "none",
            optional=True,
            readonly=False,
        )
        self._checkbox(controls, 4, "Show final plot", "peak.show_final", "Display only the final trend plot interactively. Diagnostic plots are saved but not opened.", optional=True)

        inputs = self._named_section(parent, "peak.inputs", "Peak Analysis Inputs", 1)
        self._entry(inputs, 0, "Data directory", "peak.data_dir", "Folder containing exported I_vs_2th TXT files.", "directory")
        self._entry(
            inputs,
            1,
            "Scan range/list",
            "peak.scans",
            "Leave blank to use all matching scans, or enter a scan, comma list, range, stepped range, or :n for every nth discovered scan.",
            optional=True,
            placeholder=":5, 440-460:5, or 440,441,445",
        )
        self._combo(
            inputs,
            2,
            "Scan type",
            "peak.scan_type",
            ["chi", "delta", "z", "omega", ""],
            "Exported scan family to use. Leave blank if filenames do not include a scan type.",
            "chi",
            optional=True,
        )
        self._entry(inputs, 3, "Frame index", "peak.frame_index", "Frame file index to fit for every scan. Disabled for delta scans, which use all frames.", default="0")

        replot = self._named_section(parent, "peak.replot", "Existing Results", 2)
        self._entry(
            replot,
            0,
            "Existing results CSV",
            "peak.results_csv",
            "Saved peak_series CSV to replot on a different X/metadata field without refitting.",
            "file",
            optional=True,
        )

        fitting = self._named_section(parent, "peak.fit_options", "One/Two-Peak Fit", 3)
        self._entry(fitting, 0, "Peak center", "peak.center", "Approximate 2theta center of the peak or split-peak pair.")
        self._entry(fitting, 1, "Fit window", "peak.window", "Half-width in 2theta degrees around Peak center.", default="0.5")
        self._entry(
            fitting,
            2,
            "Initial split",
            "peak.split_guess",
            "Optional initial separation between the two peak centers in degrees.",
            optional=True,
        )
        self._checkbox(
            fitting,
            3,
            "Save every diagnostic plot",
            "peak.diagnostic_all_fits",
            "Save one diagnostic image for every successful scan. Each image shows only the final one-peak and/or two-peak fits.",
            optional=True,
            default=True,
        )
        buttons = ttk.Frame(parent)
        buttons.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.peak_action_button = ttk.Button(buttons, text="Run Peak Analysis", command=self.run_peak_action)
        self.peak_action_button.grid(row=0, column=0, padx=(0, 8))
        ToolTip(self.peak_action_button, "Run the selected Peak Analysis action.")
        action.trace_add("write", lambda *_: self._update_peak_mode())
        self.variables["peak.scan_type"].trace_add("write", lambda *_: self._update_peak_mode())
        self._update_peak_mode()

    def _build_williamson_hall_tab(self, parent):
        inputs = self._named_section(parent, "wh.inputs", "Williamson-Hall Inputs", 0)
        self._entry(inputs, 0, "Data directory", "wh.data_dir", "Folder containing exported I_vs_2th TXT files or scan CSV files.", "directory")
        self._entry(
            inputs,
            1,
            "Scan range/list",
            "wh.scans",
            "Leave blank to use all matching scans, or enter a scan, list, range, stepped range, or :n for every nth discovered scan.",
            optional=True,
            placeholder=":5, 440-460:5, or 440,441,445",
        )
        self._combo(
            inputs,
            2,
            "Profile source",
            "wh.profile_source",
            ["txt", "csv"],
            "TXT reads exported I_vs_2th frame files. CSV reads a suitable scan CSV with 2theta and intensity columns.",
            "txt",
        )
        self._combo(
            inputs,
            3,
            "Scan type",
            "wh.scan_type",
            ["delta", "chi", "z", "omega", ""],
            "TXT scan family to use. Delta scans are combined across all frames by overlap averaging.",
            "delta",
            optional=True,
        )
        self._entry(inputs, 4, "Frame index", "wh.frame_index", "Frame file index for non-delta TXT scans or grouped CSV profiles.", default="0", optional=True)
        self._combo(
            inputs,
            5,
            "X/metadata field",
            "wh.x",
            X_METADATA_OPTIONS,
            "Trend x-axis for the saved WH summary plot.",
            "scan_number",
        )
        self._combo(
            inputs,
            6,
            "Second y-axis",
            "wh.secondary_y",
            ["none", "temperature", "energy", "usable_peak_count", "target_count", "registration_initial_shift"],
            "Optional right-hand y-axis on the microstrain trend panel. You can type any numeric summary column.",
            "none",
            optional=True,
            readonly=False,
        )
        self._checkbox(inputs, 7, "Show final plot", "wh.show_final", "Display the final Williamson-Hall trend plot interactively.", optional=True)

        targets = self._named_section(parent, "wh.targets", "Peak Targets", 1)
        target_source = self._radio_group(
            targets,
            0,
            "Target source",
            "wh.target_source",
            [
                ("lattice", "Lattice", "Generate expected peak positions from lattice parameters and energy/wavelength."),
                ("manual", "2theta list", "Use manually supplied 2theta positions, optionally with labels."),
            ],
            default="lattice",
        )
        self._entry(
            targets,
            1,
            "2theta positions",
            "wh.manual_two_theta",
            "Comma-separated positions, optionally with labels, e.g. 31.8 (111), 38.5 shoulder.",
            optional=True,
        )
        self._combo(
            targets,
            2,
            "Lattice type",
            "wh.lattice_type",
            ["cubic", "fcc", "bcc", "tetragonal", "hcp", "orthorhombic"],
            "Crystal/lattice family for calculated target positions.",
            "cubic",
        )
        self._entry(targets, 3, "a (Angstrom)", "wh.a", "Lattice parameter a in Angstrom.")
        self._entry(targets, 4, "b (Angstrom)", "wh.b", "Lattice parameter b in Angstrom where required.", optional=True)
        self._entry(targets, 5, "c (Angstrom)", "wh.c", "Lattice parameter c in Angstrom where required.", optional=True)
        self._entry(targets, 6, "Phase label", "wh.phase_name", "Optional label added to hkl peak labels.", optional=True)
        self._entry(targets, 7, "Max hkl index", "wh.max_index", "Maximum h, k, l index used for lattice-generated targets.", default="8")
        self._entry(
            targets,
            8,
            "Thermal alpha (1/K)",
            "wh.thermal_alpha",
            "Optional isotropic linear thermal expansion coefficient for WH target positions only.",
            optional=True,
        )
        self._entry(
            targets,
            9,
            "Reference temperature",
            "wh.reference_temperature",
            "Temperature corresponding to the entered lattice parameters. Use the same scale as scan metadata; only T - T0 is used.",
            default="25",
            optional=True,
        )

        settings = self._named_section(parent, "wh.settings", "Broadening and Fit Settings", 2)
        self._entry(settings, 0, "Wavelength (Angstrom)", "wh.wavelength", "X-ray wavelength. If blank, quixrd tries energy metadata.", optional=True)
        self._entry(settings, 1, "Energy (keV)", "wh.energy", "Beam energy in keV or eV. Linked to wavelength while typing.", optional=True)
        self._entry(settings, 2, "Fit window", "wh.fit_window", "Half-width in 2theta degrees around each assigned peak.", default="0.35")
        self._entry(settings, 3, "Shape factor K", "wh.shape_factor", "Shape factor used for D = K lambda / intercept.", default="0.9")
        self._entry(
            settings,
            4,
            "2theta calibration JSON",
            "wh.twotheta_calibration_json",
            "Optional 2theta calibration JSON. If blank, the selected default calibration is used when enabled.",
            "json",
            optional=True,
        )
        self._entry(
            settings,
            5,
            "Uncalibrated registration window",
            "wh.registration_window",
            "Maximum broad fingerprint offset allowed for uncorrected profiles.",
            default="2.0",
        )
        self._entry(
            settings,
            6,
            "Calibrated residual shift",
            "wh.residual_shift_limit",
            "Maximum residual registration shift when the profile is already 2theta-corrected.",
            default="0.15",
        )

        buttons = ttk.Frame(parent)
        buttons.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        button = ttk.Button(buttons, text="Run Williamson-Hall Analysis", command=self.run_williamson_hall)
        button.grid(row=0, column=0, padx=(0, 8))
        ToolTip(button, "Fit multiple peaks and generate Williamson-Hall diagnostic outputs.")

        self._link_energy_wavelength_fields("wh.wavelength", "wh.energy")
        target_source.trace_add("write", lambda *_: self._update_wh_mode())
        self.variables["wh.profile_source"].trace_add("write", lambda *_: self._update_wh_mode())
        self.variables["wh.scan_type"].trace_add("write", lambda *_: self._update_wh_mode())
        self.variables["wh.lattice_type"].trace_add("write", lambda *_: self._update_wh_mode())
        self._update_wh_mode()

    def _build_sin2psi_tab(self, parent):
        inputs = self._named_section(parent, "sin2psi.inputs", "Inputs", 0)
        self._entry(inputs, 0, "Data directory", "sin2psi.data_dir", "Folder containing I_vs_2th files and/or sin2psi_export.", "directory")
        self._entry(
            inputs,
            1,
            "Scan range/list",
            "sin2psi.scans",
            "Single scan, comma list, range, stepped range, or :n for every nth discovered scan.",
            placeholder="e.g. :5, 440,441,443 or 440-500:5",
        )
        action = self._radio_group(
            inputs,
            2,
            "Action",
            "sin2psi.action",
            [
                ("process", "Process peaks", f"Fit peaks from exported frame files, then fit the {SIN2PSI_LABEL} trend."),
                ("refit", "Refit trends only", f"Reuse existing peak-fit CSV files and refit only the {SIN2PSI_LABEL} trend."),
                ("correction", "Generate correction", "Create a chi/psi correction curve from a stress-free reference scan."),
                ("summaries", "Plot summaries", "Plot existing gradient or FWHM summaries without refitting peaks."),
            ],
            "process",
        )
        self._checkbox(inputs, 3, "Show final plot", "sin2psi.show_final", "Display final summary/correction plots interactively when supported.", optional=True, default=True)

        peak_options = self._named_section(parent, "sin2psi.peak_options", "Peak Fitting Options", 1)
        self._entry(peak_options, 0, "Peak center", "sin2psi.peak_center", "Initial 2theta peak center; leave blank for auto.", optional=True)
        self._entry(peak_options, 1, "Track window", "sin2psi.track_window", "Half-width in 2theta degrees around a tracked peak.", default="1.0")
        self._checkbox(peak_options, 2, "Track peak", "sin2psi.track_peak", "Seed each frame from previous successful peak fit.", default=True)
        self._checkbox(peak_options, 3, "Plot frames", "sin2psi.plot_frames", "Save per-frame peak-fit plots.", default=True)
        self._checkbox(peak_options, 4, "Preview first scan before batch", "sin2psi.preview", "Fit or preview one scan first, then ask before continuing.", optional=True)

        exclusions = self._named_section(parent, "sin2psi.exclusions", "Exclusions and Correction", 2)
        self._entry(exclusions, 0, "Exclude frames", "sin2psi.exclude_frames", "Comma-separated frame indices to exclude from sin2psi fit.", optional=True)
        self._entry(exclusions, 1, "Exclude chi ranges", "sin2psi.exclude_chi", "Ranges such as 0-5,85-90.", optional=True)
        self._entry(exclusions, 2, f"Exclude {SIN2PSI_LABEL} ranges", "sin2psi.exclude_sin2psi", "Ranges such as 0.95-1.0.", optional=True)
        self._checkbox(exclusions, 3, "Auto exclude", "sin2psi.auto_exclude", "Trial residual-based outlier exclusion.", optional=True)
        self._entry(exclusions, 4, "Correction JSON(s)", "sin2psi.correction_json", f"{SIN2PSI_LABEL} correction JSON file, or multiple files separated by semicolons.", "json_multi", optional=True)

        stress = self._named_section(parent, "sin2psi.stress", "Stress Calculation", 3)
        self._entry(stress, 0, "E", "sin2psi.elastic_E", "Young's modulus. Stress is reported in the same units as E.", optional=True)
        self._entry(stress, 1, "E units", "sin2psi.elastic_E_units", "Optional label for E and calculated stress units, e.g. MPa or GPa.", default="GPa", optional=True)
        self._entry(stress, 2, "nu", "sin2psi.elastic_nu", "Poisson ratio.", optional=True)
        self._entry(stress, 3, "Stress-free 2theta0", "sin2psi.stress_reference_two_theta", "Optional stress-free 2theta reference.", optional=True)
        self._entry(stress, 4, "Stress-free d0", "sin2psi.stress_reference_d0", "Optional stress-free d-spacing reference.", optional=True)
        self._entry(stress, 5, "Stress wavelength", "sin2psi.stress_wavelength", "Optional wavelength in Angstrom.", optional=True)
        self._entry(stress, 6, "Stress energy", "sin2psi.stress_energy", "Optional energy in keV or eV; otherwise scan metadata is used.", optional=True)
        self._link_energy_wavelength_fields("sin2psi.stress_wavelength", "sin2psi.stress_energy")

        correction = self._named_section(parent, "sin2psi.calibration", "Calibration Curve", 4)
        self._entry(correction, 0, "Reference folder", "sin2psi.reference_folder", "Reference export folder containing sin2psi_export.", "directory")
        self._entry(correction, 1, "Reference scan", "sin2psi.reference_scan", "Stress-free reference scan number.")
        self._combo(correction, 2, "Method", "sin2psi.correction_method", ["polynomial", "gaussian_process"], "Correction fit method.", "polynomial")
        self._entry(correction, 3, "Polynomial degree", "sin2psi.correction_degree", "Degree used for polynomial correction fitting.", default="2")
        self._entry(correction, 4, "Reference 2theta", "sin2psi.reference_two_theta", "Optional reference angle; leave blank to fit true 2theta.", optional=True)

        summary = self._named_section(parent, "sin2psi.summary", "Summary Plotting", 5)
        self._combo(
            summary,
            0,
            "X/metadata field",
            "sin2psi.summary_x",
            X_METADATA_OPTIONS,
            "Trend x-axis for sin2psi summary plots.",
            "scan_number",
        )
        self._combo(
            summary,
            1,
            "Second y-axis",
            "sin2psi.summary_secondary_y",
            ["none", "slope", "stress", "temperature", "energy", "chi", "psi_deg", "sin2psi"],
            "Optional right-hand y-axis for sin2psi summary plots. You can type any numeric column in the summary CSV.",
            "none",
            optional=True,
            readonly=False,
        )
        action.trace_add("write", lambda *_: self._update_sin2psi_mode())
        self.variables["sin2psi.correction_method"].trace_add("write", lambda *_: self._update_sin2psi_mode())
        self._update_sin2psi_mode()
        self._sin2psi_buttons(parent, 6)

    def show_help(self):
        if self.help_window is not None and self.help_window.winfo_exists():
            self.help_window.lift()
            self.help_window.focus_set()
            self.set_status("Help is already open.")
            return

        window = tk.Toplevel(self.master)
        window.title("quixrd Workflow Help")
        self._apply_window_icon(window)
        window.geometry("900x720")
        window.minsize(720, 520)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        window.protocol("WM_DELETE_WINDOW", self._close_help)
        self.help_window = window

        frame = ttk.Frame(window, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        help_notebook = ttk.Notebook(frame)
        help_notebook.grid(row=0, column=0, sticky="nsew")
        self.help_notebook = help_notebook
        for title, text in self._help_sections():
            tab = ttk.Frame(help_notebook, padding=8)
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
            help_text = tk.Text(tab, wrap="word", height=28)
            help_text.grid(row=0, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(tab, orient="vertical", command=help_text.yview)
            scrollbar.grid(row=0, column=1, sticky="ns")
            help_text.configure(yscrollcommand=scrollbar.set)
            self._render_help_text(help_text, text)
            help_text.configure(state="disabled")
            help_notebook.add(tab, text=title)

        close_button = ttk.Button(frame, text="Close", command=self._close_help)
        close_button.grid(row=1, column=0, sticky="e", pady=(8, 0))
        self.log("Opened Help.")

    def _render_help_text(self, widget, text):
        default_font = tkfont.nametofont("TkDefaultFont")
        body_font = default_font.copy()
        body_font.configure(size=max(default_font.cget("size"), 10))
        title_font = body_font.copy()
        title_font.configure(size=body_font.cget("size") + 4, weight="bold")
        heading_font = body_font.copy()
        heading_font.configure(size=body_font.cget("size") + 1, weight="bold")
        bullet_font = body_font.copy()

        widget.configure(
            font=body_font,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            spacing1=2,
            spacing2=2,
            spacing3=8,
            background=self.master.cget("background"),
        )
        widget.tag_configure("title", font=title_font, spacing3=12)
        widget.tag_configure("heading", font=heading_font, spacing1=10, spacing3=6)
        widget.tag_configure("body", font=body_font, lmargin1=0, lmargin2=0)
        widget.tag_configure("bullet", font=bullet_font, lmargin1=16, lmargin2=34, spacing3=4)

        lines = text.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            if next_line and set(next_line) <= {"="}:
                widget.insert("end", line + "\n", "title")
                idx += 2
                continue
            if next_line and set(next_line) <= {"-"}:
                widget.insert("end", line + "\n", "heading")
                idx += 2
                continue
            if line.startswith("- "):
                widget.insert("end", line + "\n", "bullet")
            else:
                widget.insert("end", line + "\n", "body")
            idx += 1

    def _close_help(self):
        if self.help_window is not None and self.help_window.winfo_exists():
            self.help_window.destroy()
        self.help_window = None
        self.set_status("Help closed.")

    def _help_sections(self):
        return [
            (
                "Overview",
                "quixrd Workflow Help\n"
                "=====================\n\n"
                "This GUI runs the XRD workflow tools for extraction, plotting, sorting, and sin2psi analysis. It also saves and "
                "restores analysis setups through JSON parameter files.\n\n"
                "General conventions\n"
                "-------------------\n"
                "- Optional field labels are shown in italics. Leave them blank to use the workflow default.\n"
                "- Grey example text inside an empty box is only a hint; it is ignored until you type a real value.\n"
                "- Greyed-out controls are not used by the selected mode or action.\n"
                "- Scan range/list fields accept a single scan such as 440, a comma list such as 440,441,443, a range such as 440-450, "
                "a stepped range such as 440-500:5, or :5 for every fifth discovered scan where automatic scan discovery is available.\n"
                "- Browse buttons fill path fields with folders or files from the filesystem.\n"
                "- Show graph / Show final plot means display the plot interactively. Saved plots can still be produced without showing them.\n"
                "- The bottom command log records GUI actions. The status bar shows the most recent command or setting change.\n\n"
                "Status and troubleshooting\n"
                "--------------------------\n"
                "The command log is deliberately visible at the bottom of the window so long-running jobs can report progress. "
                "If a control is disabled, first check the radio-button mode on that tab. If an output is missing, "
                "check the selected data directory and scan range/list before changing analysis settings.\n",
            ),
            (
                "File",
                "File Menu and Parameter Sets\n"
                "============================\n\n"
                "Use the File menu to export current GUI parameters to JSON or import a previous JSON file to resume an analysis.\n\n"
                "Export Current Tab Parameters writes only the fields belonging to the tab currently selected in the main notebook.\n"
                "Export All Parameters writes every GUI field from every tab into one JSON file.\n"
                "Import Into Current Tab reads a JSON file and applies only values that belong to the currently selected tab. This is useful "
                "when you want to reuse, for example, an old sin2psi setup without disturbing extraction or sorting fields.\n"
                "Import Into All Tabs reads a JSON file and applies every recognised GUI value it contains. Unknown keys are ignored, so "
                "older or hand-edited parameter files should not break the GUI.\n"
                "After importing, radio-button dependent fields are refreshed so irrelevant entries are greyed out again.\n"
                "\n"
                "Reveal Selected Path in File Explorer opens the path field currently selected in the GUI. If focus is not in a path "
                "field, it opens the first filled path field on the active tab.\n"
                "Select Local Cache Folder chooses where cached input files are stored. The selected folder is saved between GUI "
                f"sessions. The default is {DEFAULT_CACHE_ROOT}.\n"
                "Use Local Cache in the File menu copies only needed exported I_vs_2th TXT scan files into the cache "
                "on demand. Existing cached files with the same filename are reused. Output folders are not changed.\n"
                "Clear Local Cache deletes the selected local cache folder after confirmation.\n",
            ),
            (
                "Calibration",
                "Calibration Menu\n"
                "================\n\n"
                "2theta Calibration opens a separate window for generating a 2theta-axis calibration from LaB6 or a custom lattice reference. "
                "This is separate from the sin2psi chi/psi correction: 2theta calibration is applied before peak fitting.\n\n"
                "Select 2theta Calibration File chooses a quixrd 2theta-axis calibration JSON. The GUI remembers the selected file "
                "between sessions and uses it when calibration application is enabled.\n"
                "Apply 2theta Calibration by Default applies the selected calibration during extraction post-processing, spectra plotting, "
                "and sin2psi peak processing. Exported TXT files are marked with a TwoTheta Correction metadata line containing the "
                "calibration JSON path. Files already carrying that metadata are treated as corrected and are not corrected again.\n\n"
                "Calibration window inputs\n"
                "-------------------------\n"
                "Source type accepts raw SOLEIL XPAD NXS delta scans, exported TXT delta-frame files, or a single all-frame CSV export. "
                "For TXT input, Browse opens a multi-file picker and stores paths separated by semicolons.\n"
                "Output folder is where the calibration JSON, combined profile, and plots are written. If blank, quixrd creates a "
                "calibration subfolder beside the selected input.\n"
                "LaB6 autofills a cubic lattice with a = 4.156826 Angstrom. Custom lets you choose lattice type and lattice parameters.\n"
                "Energy and wavelength are linked; if both are blank, quixrd tries to read energy from the input metadata.\n"
                "Polynomial degree controls the fitted offset-vs-2theta curve. Degree 0 is allowed for a constant offset.\n"
                "Outlier mode controls peak exclusions for the offset polynomial and Caglioti broadening fits. Review proposes "
                "automatic exclusions first, then opens a table for you to accept or edit them. Automatic applies the same "
                "thresholds without stopping. Manual uses only the HKL keys or peak indices typed in Manual exclusions. Off uses "
                "every usable fitted peak.\n"
                "Outlier sensitivity controls the automatic thresholds. Conservative excludes fewer peaks, normal is the default, "
                "and aggressive excludes more. Advanced fields can override the minimum offset residual, minimum FWHM residual, "
                "broad-peak FWHM multiplier, and maximum fraction of fitted peaks that automatic mode may remove.\n"
                "All exclusions are traceable in the calibration JSON, including mode, thresholds, excluded hkl/2theta/FWHM, and "
                "whether each exclusion came from automatic filtering, review, or manual selection. Excluded points are still shown "
                "on the final fit plot in cyan; blue points are used in the fit and orange lines are the fitted polynomial/Caglioti models.\n"
                "Generate Calibration writes the JSON, combined TXT/CSV profile, profile plot with predicted hkl/multiplicity labels, "
                "and fit plot with the offset polynomial and Caglioti FWHM curve.\n"
                "Extract and Apply Corrections enables default 2theta calibration application and switches to the Extraction tab.\n",
            ),
            (
                "Extraction",
                "Extraction\n"
                "==========\n\n"
                "Use this tab for exporting data from NXS files into TXT/CSV files that later plotting and sin2psi analysis can read.\n\n"
                "NXS input directory: Folder containing raw NXS files. This can be the normal date-folder layout or a sample-sorted tree.\n"
                "Export directory: Destination folder for extracted TXT/CSV/PNG output.\n"
                "Flat directory and flat scan numbers: Optional flat-field inputs used when the extraction workflow needs detector correction.\n"
                "Date range: Optional date-folder lookup range, for example 20260609-20260615.\n\n"
                "Extraction mode:\n"
                "- TXT frames exports one I_vs_2th text file per frame. This is the usual input for sin2psi peak fitting.\n"
                "- Combined CSV exports a scan-level CSV table instead of separate frame text files.\n"
                "- Batch runs extraction over multiple scans or a scan range and writes both TXT frame files and combined CSV files.\n\n"
                "Mirror sorted structure: Used for sample-sorted NXS trees. When enabled, exported files are written into matching sample "
                "subfolders under the export directory.\n"
                "Save graph: Saves extraction overview PNG files.\n"
                "Show graph: Displays extraction plots while processing.\n",
            ),
            (
                "Plotting",
                "Plotting\n"
                "========\n\n"
                "Use this tab for plotting already-extracted spectra and summary trends from existing analysis outputs.\n\n"
                "Export/data directory: For spectra this is normally the export folder containing TXT/CSV files. For gradient, FWHM, "
                "and peak-position trends this should be the folder containing sin2psi_export.\n"
                "Scan range/list: Leave blank to include all available scans, or enter a single scan, comma list, range, stepped range such as 440-500:5, or :5 for every fifth discovered scan.\n"
                "Scan types: Check the exported scan families to include in spectra plots: chi, delta, z, or omega.\n"
                "Legend labels: Choose which extra metadata are added to spectra legend entries. Scan number is always included.\n"
                "Offset: Vertical spacing multiplier for spectra traces.\n"
                "X/metadata field: Trend x-axis. Common values include scan_number, temperature, energy, start_time, and frame_time. "
                "Other metadata columns can work if they were collected into the summary files.\n"
                "Summary CSV: Optional exact sin2psi scan summary file for Gradient or Stress plots. Leave blank to collect/reuse the current summary.\n"
                "Show final plot displays the result interactively. Save final plot writes the result automatically when Run Selected Plot is used.\n"
                "Save Current Plot: Saves the currently open Matplotlib figure into saved_plots under the data directory, or under the "
                "selected Summary CSV folder if one is set.\n\n"
                "Plot type:\n"
                "- Spectra plots intensity versus 2theta from exported TXT/CSV data.\n"
                f"- Gradient plots the {SIN2PSI_LABEL} fitted slope versus scan number or metadata, with slope uncertainty as error bars.\n"
                f"- Stress plots calculated {SIN2PSI_LABEL} stress versus scan number or metadata, with stress uncertainty as error bars.\n"
                "- FWHM plots fitted peak width versus scan number or metadata, with FWHM uncertainty as error bars when available.\n"
                "- Peak position plots fitted peak center versus scan number or metadata, with peak-center uncertainty when available.\n\n"
                "Predicted peaks:\n"
                "- Show predicted peaks overlays narrow vertical lines on spectra.\n"
                "- 2theta list accepts comma-separated peak positions, optionally with short labels.\n"
                "- Lattice parameters calculate simple powder peak positions from a, b, c and wavelength or energy. If energy is blank, "
                "the spectra plotter tries the exported scan metadata.\n\n"
                "FWHM / peak-position selector:\n"
                "- Frame number accepts one frame, a comma list, or a range. Each selected frame is plotted as a separate series.\n"
                "- Chi value accepts one chi value or a comma list. Each chi value is plotted as a separate series.\n"
                "- Chi matching uses a 0.1 degree tolerance to handle small numeric metadata differences.\n",
            ),
            (
                "Sorting",
                "Sorting\n"
                "=======\n\n"
                "Use this tab for organising raw NXS files or already-extracted data into sample folders.\n\n"
                "NXS directory: Source folder containing unsorted or date-organised NXS files. For extracted-data sorting, this should be "
                "the sorted NXS directory used to infer sample folders.\n"
                "Sample spreadsheet: Spreadsheet used by sort_nxs_by_sample to map scans to samples.\n"
                "Output directory: Destination for sorted NXS folders or sorted exported files.\n"
                "Existing export directory: Folder containing already-extracted TXT/CSV/PNG files that should be copied or moved into "
                "matching sample folders.\n\n"
                "Sort mode:\n"
                "- NXS by sample sorts raw NXS files according to the sample spreadsheet.\n"
                "- Extracted data reorganises existing exported files based on scan numbers and a sorted NXS tree.\n\n"
                "Calibration handling: Controls how calibration scans are handled when sorting NXS files.\n"
                "Move extracted files: When off, sorting extracted data copies files. When on, it moves them, so use it only "
                "when you are ready to modify the source export folder.\n",
            ),
            (
                "Peak Analysis",
                "Peak Analysis\n"
                "=============\n\n"
                "Use this tab to test whether a selected peak is better described by one peak or by two overlapping peaks, "
                "for example during spinodal decomposition. It can also track peak center and FWHM across a series of scans.\n\n"
                "Data directory: Folder containing exported I_vs_2th TXT frame files.\n"
                "Scan range/list: Leave blank to fit all scans matching the selected scan type and frame index, or enter a scan, "
                "comma list, range, stepped range such as 440-500:5, or :5 for every fifth discovered scan.\n"
                "Scan type: Selects the filename family such as chi or delta. Delta scans are homogenised automatically: all "
                "I_vs_2th_<scan>_delta_<frame>.txt files are interpolated onto a common 2theta grid and overlapping regions are averaged.\n"
                "Frame index: The frame file to fit in each scan. This is disabled for delta scans because all delta frames are combined.\n"
                "X/metadata field: The trend x-axis. scan_number always works; temperature, energy, chi, and time fields work "
                "when they are present in the TXT metadata.\n"
                "Second y-axis: Optional right-hand y-axis on the peak trend plot. Choose one of the common peak-analysis metrics, "
                "or type any numeric column present in the peak_series CSV.\n"
                "Existing results CSV: A saved peak_series CSV. Select Replot existing results to change the x-axis, second y-axis, "
                "or displayed peak model without refitting.\n\n"
                "One/two-peak fit:\n"
                "- Peak center is the approximate 2theta location of the peak or split-peak pair.\n"
                "- Fit window is the half-width around that center used for fitting.\n"
                "- 1 vs 2 fits or shows both models. 1 peak and 2 peaks force fitting or plotting of only that model.\n"
                "- Initial split is an optional starting separation for the two-peak fit. If blank, quixrd estimates starting "
                "positions from local maxima or a small symmetric split.\n\n"
                "Validity of two peaks:\n"
                "The comparison uses delta BIC = BIC(single peak) - BIC(two peaks). Positive values favour two peaks; values "
                "should be read alongside the fitted separation, relative peak intensity, and diagnostic plots rather than as a hard "
                "threshold. BIC is the Bayesian Information Criterion: it compares fit quality while penalising extra fitted "
                "parameters, so the two-peak model must improve the residuals enough to justify its added complexity. "
                "The CSV stores selected_model as quixrd's assessment of which model is preferred. The trend plot shows the "
                "one-peak fit in orange and the two-peak fit in green where available; non-selected model points are faded. "
                "The delta BIC comparison panel also shows the minor/major peak-height ratio on a right-hand axis when two-peak results exist. The comparison panel "
                "appears when Peak plot model is 1 vs 2.\n\n"
                "Diagnostics:\n"
                "Each run saves one compact diagnostic plot with representative fits by default. It includes the first successful "
                "scan, a median-comparison scan when delta BIC is available, and the worst-RMSE scan. Enable Save every diagnostic plot "
                "to save one image for every successful scan so you can flick through them. These images show "
                "only the final one-peak and/or two-peak model fits, not intermediate optimiser attempts.\n",
            ),
            (
                "Williamson-Hall",
                "Williamson-Hall\n"
                "================\n\n"
                "Use this tab for quick line-broadening diagnostics across one scan or a scan series. It fits multiple expected "
                "peaks, corrects instrumental broadening when a 2theta calibration with Caglioti parameters is supplied, and "
                "plots microstrain and crystallite-size trends. This is intended as a screening tool, not a replacement for "
                "specialist whole-pattern or line-profile refinement software.\n\n"
                "Data directory: Folder containing exported TXT frame files or scan CSV files.\n"
                "Scan range/list: Leave blank to fit all matching scans, or use the normal scan syntax, including :5 for every fifth discovered scan.\n"
                "Profile source: TXT is the usual exported I_vs_2th frame format. CSV reads a suitable scan CSV containing 2theta and intensity columns.\n"
                "Scan type: For TXT input, delta scans are combined across all frames by interpolating to a common 2theta grid and averaging overlaps. "
                "Non-delta scans use the selected frame index.\n"
                "Target source: Lattice calculates expected hkl positions from lattice parameters and energy/wavelength. 2theta list uses "
                "manual peak positions, optionally with labels, for unusual structures or peaks that are not covered by the simple lattice calculator.\n"
                "Thermal expansion: Optional WH-only isotropic thermal expansion can shift lattice-generated target positions using "
                "a_T = a0 * (1 + alpha * (T - T0)). Enter alpha in 1/K and the reference temperature for the lattice parameters; "
                "use the same temperature scale as the scan metadata because only T - T0 is used. quixrd uses each scan's "
                "temperature metadata when available. This is not used for 2theta calibration, where the "
                "reference lattice value should remain fixed.\n"
                "Wavelength/Energy: Required for lattice target generation and for converting Williamson-Hall intercept into crystallite size. "
                "If blank, quixrd tries the profile metadata.\n"
                "2theta calibration JSON: Optional instrumental calibration. If the field is blank, the GUI uses the selected default calibration "
                "when Apply 2theta Calibration by Default is enabled. Profiles already marked with a TwoTheta Correction metadata line are treated "
                "as corrected.\n"
                "Caglioti broadening correction: If the selected 2theta calibration JSON contains Caglioti U/V/W parameters, Williamson-Hall "
                "subtracts instrumental FWHM broadening in quadrature before fitting: beta_sample = sqrt(beta_observed^2 - beta_instrument^2), "
                "with beta values in radians. If no calibration JSON is supplied, or the JSON has no Caglioti fit, no instrumental broadening "
                "subtraction is applied.\n"
                "Registration: Uncorrected data may use a broad fingerprint offset search to align expected peaks with observed peaks. Corrected "
                "data skip broad fingerprinting and allow only the small calibrated residual shift, so the calibration is not effectively undone.\n\n"
                "Williamson-Hall calculation:\n"
                "The first implementation uses total fitted FWHM. It plots beta cos(theta) against 4 sin(theta), with beta in radians. "
                "The fitted slope is reported as microstrain. A positive intercept gives crystallite size through D = K lambda / intercept, "
                "where K is the editable shape factor. If the intercept is not positive, the crystallite size is reported as nonphysical.\n",
            ),
            (
                f"{SIN2PSI_LABEL}",
                f"{SIN2PSI_LABEL} Analysis\n"
                "=================\n\n"
                "Use this tab for peak fitting, refitting trend lines from existing peak fits, generating correction curves, and plotting "
                "analysis summaries.\n\n"
                "Data directory: Folder containing I_vs_2th files and/or an existing sin2psi_export folder.\n"
                "The boxes below Inputs are action-specific. Hidden boxes are not used by the selected action, but their values are kept "
                "if you switch actions and come back.\n\n"
                "Peak fitting options:\n"
                "Peak center: Optional initial 2theta value for the first frame. Leave blank for automatic detection.\n"
                "Track window: Half-width in 2theta degrees around the previous successful fitted peak center.\n"
                "Track peak: Seeds each frame from the previous successful fitted center when processing peaks.\n"
                "Plot frames: Saves per-frame peak-fit plots.\n"
                "Preview first scan before batch: Runs one scan first, shows/logs the result, then asks whether to continue.\n"
                "Show final plot: Displays final trend or correction plots interactively when supported.\n\n"
                "Action:\n"
                "- Process peaks fits peak positions from exported frame TXT files and then fits the sin2psi trend.\n"
                "- Refit trends only reuses existing scan_N_fits.csv peak results, so it can apply new exclusions or corrections without "
                "refitting peak shapes.\n"
                "- Generate correction creates a correction curve from a stress-free reference scan. Polynomial and Gaussian-process "
                "methods are available.\n"
                "- Plot summaries creates gradient summary plots from existing output files. Use the Summary Plotting section to choose "
                "the x-axis and an optional second y-axis.\n\n"
                "Exclusions and correction:\n"
                "- Exclude frames removes specific frame indices from the trend fit.\n"
                "- Exclude chi ranges removes chi intervals such as 0-5 or 85-90 degrees.\n"
                f"- Exclude {SIN2PSI_LABEL} ranges removes intervals such as 0.95-1.0 from the fit.\n"
                "- Auto exclude is the residual-based outlier exclusion option.\n"
                "- Correction JSON applies a previously generated chi/psi correction curve to the trend fit.\n\n"
                "Stress calculation:\n"
                "- E and nu enable optional stress calculation during Process peaks or Refit trends only.\n"
                "- Enter stress-free 2theta0 or d0 to calculate strain relative to that reference.\n"
                "- If no reference is supplied, an equibiaxial stress state is assumed and d0 is inferred from the fitted d-spacing trend.\n"
                "- Wavelength or energy can be supplied explicitly; otherwise energy metadata from the scan is used when available.\n\n"
                "Calibration curve:\n"
                "- Reference folder points to the folder containing the reference scan's sin2psi_export output.\n"
                "- Reference scan is the stress-free reference scan number.\n"
                "- Method chooses polynomial or gaussian_process correction fitting.\n"
                "- Polynomial degree is used only for polynomial corrections and is hidden for Gaussian-process corrections.\n"
                "- Reference 2theta is optional. If blank, the correction is fitted on the true measured angle rather than as an offset "
                "from a supplied reference angle.\n",
            ),
        ]

    def _help_text(self):
        return "\n\n".join(text for _title, text in self._help_sections())

    def _build_log_panel(self):
        frame = ttk.LabelFrame(self, text="Command Log", padding=6)
        frame.grid(row=1, column=0, sticky="ew", pady=(10, 4))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(2, weight=0)
        self.log_text = tk.Text(frame, height=6, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.cancel_button = ttk.Button(frame, text="Cancel Task", command=self.cancel_current_task, state="disabled")
        self.cancel_button.grid(row=0, column=2, sticky="ne", padx=(8, 0))
        ToolTip(self.cancel_button, "Request cancellation of the current long-running task. Some operations stop between scans or files.")

    def _build_status_bar(self):
        self.status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken", padding=(6, 3))
        self.status_bar.grid(row=2, column=0, sticky="ew")

    def set_status(self, message):
        self.status_var.set(message)

    def cancel_current_task(self):
        if not self.task_running:
            self.log("No running task to cancel.")
            return
        self.cancel_event.set()
        self.log("Cancellation requested. Waiting for the current file/scan operation to finish...")

    def _set_task_running(self, running):
        self.task_running = bool(running)
        if hasattr(self, "cancel_button"):
            self.cancel_button.configure(state="normal" if running else "disabled")

    def log(self, message):
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.set_status(message)


def create_app():
    root = tk.Tk()
    app = XRDGuiApp(root)
    return root, app


def main():
    root, _app = create_app()
    root.mainloop()


if __name__ == "__main__":
    main()
