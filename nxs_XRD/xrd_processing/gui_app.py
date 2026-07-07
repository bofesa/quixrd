from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


TAB_NAMES = ["Extraction", "Plotting", "Sorting", "Sin2psi Analysis", "Help"]


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


class XRDGuiApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.master = master
        self.variables = {}
        self.status_var = tk.StringVar(value="Ready")
        self._configure_master()
        self._build()

    def _configure_master(self):
        self.master.title("nxs_XRD Workflow")
        self.master.geometry("1120x780")
        self.master.minsize(900, 620)

    def _build(self):
        self.pack(fill="both", expand=True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tabs = {}
        for name in TAB_NAMES:
            frame = ttk.Frame(self.notebook, padding=12)
            frame.columnconfigure(0, weight=1)
            self.notebook.add(frame, text=name)
            self.tabs[name] = frame

        self._build_extraction_tab(self.tabs["Extraction"])
        self._build_plotting_tab(self.tabs["Plotting"])
        self._build_sorting_tab(self.tabs["Sorting"])
        self._build_sin2psi_tab(self.tabs["Sin2psi Analysis"])
        self._build_help_tab(self.tabs["Help"])
        self._build_log_panel()
        self._build_status_bar()
        self.log("GUI layout loaded. Processing buttons are placeholders for now.")

    def _section(self, parent, title, row):
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        return frame

    def _entry(self, parent, row, label, key, tooltip="", browse=None, default=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        var = tk.StringVar(value=default)
        self.variables[key] = var
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if tooltip:
            ToolTip(entry, tooltip)
        if browse:
            button = ttk.Button(parent, text="Browse", command=lambda: self._browse(var, browse))
            button.grid(row=row, column=2, sticky="e", padx=(8, 0), pady=4)
            if tooltip:
                ToolTip(button, tooltip)
        return entry

    def _checkbox(self, parent, row, label, key, tooltip="", default=False, column=0):
        var = tk.BooleanVar(value=default)
        self.variables[key] = var
        check = ttk.Checkbutton(parent, text=label, variable=var)
        check.grid(row=row, column=column, sticky="w", pady=4)
        if tooltip:
            ToolTip(check, tooltip)
        return check

    def _combo(self, parent, row, label, key, values, tooltip="", default=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        var = tk.StringVar(value=default or values[0])
        self.variables[key] = var
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        if tooltip:
            ToolTip(combo, tooltip)
        return combo

    def _buttons(self, parent, row, buttons):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        for idx, label in enumerate(buttons):
            button = ttk.Button(frame, text=label, command=lambda text=label: self.placeholder(text))
            button.grid(row=0, column=idx, padx=(0, 8))
            ToolTip(button, "Layout-only placeholder. This button will be wired in the next pass.")

    def _browse(self, variable, browse):
        if browse == "directory":
            value = filedialog.askdirectory()
        else:
            value = filedialog.askopenfilename()
        if value:
            variable.set(value)
            self.set_status(f"Selected: {value}")

    def _build_extraction_tab(self, parent):
        paths = self._section(parent, "Paths", 0)
        self._entry(paths, 0, "NXS input directory", "extract.nxs_dir", "Folder containing .nxs files.", "directory")
        self._entry(paths, 1, "Export directory", "extract.export_dir", "Folder where TXT/CSV/PNG exports will be written.", "directory")
        self._entry(paths, 2, "Flat directory", "extract.flat_dir", "Folder containing flat-field NXS files.", "directory")
        self._entry(paths, 3, "Flat scan numbers", "extract.flat_scans", "Comma-separated flat scan numbers, e.g. 39,40.")

        scans = self._section(parent, "Scans and Options", 1)
        self._entry(scans, 0, "Scan range/list", "extract.scans", "Single scan, comma list, or range, e.g. 1515 or 1500-1520.")
        self._entry(scans, 1, "Date range", "extract.date_range", "Date range used for date-folder NXS lookup, e.g. 20260609-20260615.")
        self._checkbox(scans, 2, "Mirror sorted structure", "extract.mirror", "Mirror sample-sorted NXS folders in the export directory.")
        self._checkbox(scans, 3, "Save graph", "extract.save_graph", "Save extraction overview PNGs.")
        self._checkbox(scans, 4, "Show graph", "extract.show_graph", "Display extraction plots interactively.")
        self._buttons(parent, 2, ["Extract TXT", "Extract CSV", "Batch Extract"])

    def _build_plotting_tab(self, parent):
        options = self._section(parent, "Plot Inputs", 0)
        self._entry(options, 0, "Export/data directory", "plot.data_dir", "Folder containing exported data or sin2psi_export.", "directory")
        self._entry(options, 1, "Scan range/list", "plot.scans", "Single scan, comma list, or range.")
        self._entry(options, 2, "Scan types", "plot.scan_types", "Comma-separated types such as chi,delta,z,omega.", default="chi")
        self._entry(options, 3, "X/metadata field", "plot.x", "Metadata column for trend x-axis, e.g. scan_number, temperature, frame_time.", default="scan_number")
        self._entry(options, 4, "FWHM frame number", "plot.fwhm_frame", "Frame index used for FWHM trends.")
        self._entry(options, 5, "FWHM chi value", "plot.fwhm_chi", "Exact chi value used for FWHM trends.")
        self._buttons(parent, 1, ["Plot Spectra", "Plot Gradients", "Plot FWHM"])

    def _build_sorting_tab(self, parent):
        paths = self._section(parent, "Sorting Inputs", 0)
        self._entry(paths, 0, "NXS directory", "sort.nxs_dir", "Unsorted NXS directory.", "directory")
        self._entry(paths, 1, "Sample spreadsheet", "sort.sample_file", "Spreadsheet used by sort_nxs_by_sample.", "file")
        self._entry(paths, 2, "Output directory", "sort.output_dir", "Destination for sorted NXS folders.", "directory")
        self._entry(paths, 3, "Existing export directory", "sort.export_dir", "Existing exported TXT/CSV/PNG files to sort by sample.", "directory")
        self._combo(paths, 4, "Calibration handling", "sort.calibrations", ["sub", "copy", "skip"], "How calibration scans should be handled.", "sub")
        self._checkbox(paths, 5, "Move extracted files", "sort.move", "Move instead of copy when sorting already-extracted data.")
        self._buttons(parent, 1, ["Sort NXS", "Sort Extracted Data"])

    def _build_sin2psi_tab(self, parent):
        inputs = self._section(parent, "Processing", 0)
        self._entry(inputs, 0, "Data directory", "sin2psi.data_dir", "Folder containing I_vs_2th files and/or sin2psi_export.", "directory")
        self._entry(inputs, 1, "Scan range/list", "sin2psi.scans", "Single scan, comma list, or range.")
        self._entry(inputs, 2, "Peak center", "sin2psi.peak_center", "Initial 2theta peak center; leave blank for auto.")
        self._entry(inputs, 3, "Track window", "sin2psi.track_window", "Half-width in 2theta degrees around a tracked peak.", default="1.0")
        self._checkbox(inputs, 4, "Track peak", "sin2psi.track_peak", "Seed each frame from previous successful peak fit.", True)
        self._checkbox(inputs, 5, "Plot frames", "sin2psi.plot_frames", "Save per-frame peak-fit plots.", True)

        exclusions = self._section(parent, "Exclusions and Correction", 1)
        self._entry(exclusions, 0, "Exclude frames", "sin2psi.exclude_frames", "Comma-separated frame indices to exclude from sin2psi fit.")
        self._entry(exclusions, 1, "Exclude chi ranges", "sin2psi.exclude_chi", "Ranges such as 0-5,85-90.")
        self._entry(exclusions, 2, "Exclude sin2psi ranges", "sin2psi.exclude_sin2psi", "Ranges such as 0.95-1.0.")
        self._checkbox(exclusions, 3, "Auto exclude", "sin2psi.auto_exclude", "Trial residual-based outlier exclusion.")
        self._entry(exclusions, 4, "Correction JSON", "sin2psi.correction_json", "Sin2psi correction JSON file.", "file")

        correction = self._section(parent, "Calibration Curve", 2)
        self._entry(correction, 0, "Reference folder", "sin2psi.reference_folder", "Reference export folder containing sin2psi_export.", "directory")
        self._entry(correction, 1, "Reference scan", "sin2psi.reference_scan", "Stress-free reference scan number.")
        self._combo(correction, 2, "Method", "sin2psi.correction_method", ["polynomial", "gaussian_process"], "Correction fit method.", "polynomial")
        self._entry(correction, 3, "Reference 2theta", "sin2psi.reference_two_theta", "Optional reference angle; leave blank to fit true 2theta.")
        self._buttons(parent, 3, ["Process Scans", "Refit Trends Only", "Generate Correction", "Plot Summaries"])

    def _build_help_tab(self, parent):
        help_text = tk.Text(parent, wrap="word", height=20)
        help_text.grid(row=0, column=0, sticky="nsew")
        parent.rowconfigure(0, weight=1)
        text = (
            "Extraction\n"
            "Choose NXS, export, and flat-field folders. Scan range/list accepts forms like 1515, 1515-1520, or 1515,1517.\n\n"
            "Plotting\n"
            "Use exported data folders for spectra plots and sin2psi_export folders for gradient/FWHM summaries.\n\n"
            "Sorting\n"
            "Sort NXS files by sample spreadsheet, or copy/move already-extracted data into sample folders.\n\n"
            "Sin2psi Analysis\n"
            "Process scans from I_vs_2th TXT files, refit trends from existing CSVs, generate correction curves, and plot summaries.\n"
            "Correction JSON files are generated from stress-free reference scans.\n\n"
            "Status and Log\n"
            "The bottom log records command-style messages. The status bar shows the most recent command or activity."
        )
        help_text.insert("1.0", text)
        help_text.configure(state="disabled")

    def _build_log_panel(self):
        frame = ttk.LabelFrame(self, text="Command Log", padding=6)
        frame.grid(row=1, column=0, sticky="ew", pady=(10, 4))
        frame.columnconfigure(0, weight=1)
        self.log_text = tk.Text(frame, height=6, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _build_status_bar(self):
        self.status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken", padding=(6, 3))
        self.status_bar.grid(row=2, column=0, sticky="ew")

    def set_status(self, message):
        self.status_var.set(message)

    def log(self, message):
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.set_status(message)

    def placeholder(self, command_name):
        self.log(f"{command_name}: not wired yet.")


def create_app():
    root = tk.Tk()
    app = XRDGuiApp(root)
    return root, app


def main():
    root, _app = create_app()
    root.mainloop()


if __name__ == "__main__":
    main()
