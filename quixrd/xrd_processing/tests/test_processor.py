from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import numpy as np
import pandas as pd

from quixrd.xrd_processing import run_workflow
from quixrd.xrd_processing import quixrd_gui_app as gui_app
from quixrd.xrd_processing import peak_overlay
from quixrd.xrd_processing import sin2psi_processor as proc
from quixrd.xrd_processing import spinodal_peak_analysis as spinodal
from quixrd.xrd_processing import twotheta_calibration as tth_cal


class ProcessorSmokeTest(unittest.TestCase):
    @property
    def repo_root(self):
        return Path(__file__).resolve().parents[3]

    def _write_scan_txt(self, path, scan_type="ascan_chi", chi=0.0):
        path.write_text(
            "\n".join(
                [
                    f"# Scan Type: {scan_type}",
                    f"# Chi: {chi}",
                    "# Temperature: 300",
                    "# Energy: 12000",
                    "",
                    "40.0 1.0",
                    "40.1 2.0",
                    "40.2 4.0",
                    "40.3 2.0",
                    "40.4 1.0",
                ]
            ),
            encoding="utf-8",
        )

    def test_parse_and_fit_smoke(self):
        sample = self.repo_root / "export" / "I_vs_2th_440_chi_0.txt"
        self.assertTrue(sample.exists(), f"Missing sample file: {sample}")

        parsed = proc.parse_txt_scan(str(sample))
        self.assertEqual(parsed["filename"], "I_vs_2th_440_chi_0.txt")
        self.assertEqual(parsed["scan_type"].lower(), "ascan_chi")
        self.assertIsNotNone(parsed["chi"])
        self.assertGreater(len(parsed["tth"]), 10)
        self.assertEqual(len(parsed["intensity"]), len(parsed["tth"]))

        result = proc.fit_frame(parsed["tth"], parsed["intensity"], plot=False)
        for key in [
            "center",
            "center_err",
            "amplitude",
            "fwhm",
            "nu",
            "bg_coef_0",
            "bg_coef_1",
            "bg_coef_2",
            "x_fit",
            "y_combined_fit",
        ]:
            self.assertIn(key, result)

    def test_discover_accepts_ascan_and_dscan_two_part_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_scan_txt(tmp_path / "I_vs_2th_900_0.txt", scan_type="ascan_chi", chi=0.0)
            self._write_scan_txt(tmp_path / "I_vs_2th_900_1.txt", scan_type="dscan_chi", chi=5.0)
            self._write_scan_txt(tmp_path / "I_vs_2th_900_chi_2.txt", scan_type="other", chi=10.0)
            self._write_scan_txt(tmp_path / "I_vs_2th_900_delta_3.txt", scan_type="ascan_chi", chi=15.0)

            found = [Path(path).name for path in proc.discover_scan_files(tmp_path, 900)]

        self.assertEqual(
            found,
            [
                "I_vs_2th_900_0.txt",
                "I_vs_2th_900_1.txt",
                "I_vs_2th_900_chi_2.txt",
            ],
        )

    def test_seeded_fit_frame_reports_window_mode(self):
        sample = self.repo_root / "export" / "I_vs_2th_440_chi_0.txt"
        parsed = proc.parse_txt_scan(str(sample))
        initial = proc.fit_frame(parsed["tth"], parsed["intensity"], plot=False)

        seeded = proc.fit_frame(
            parsed["tth"],
            parsed["intensity"],
            plot=False,
            seed_center=initial["center"],
            track_window=0.2,
        )

        self.assertEqual(seeded["window_mode"], "seeded")
        self.assertAlmostEqual(seeded["seed_center"], initial["center"])
        self.assertIn("background_lower", seeded)
        self.assertIn("peak_upper", seeded)

    def test_predicted_peak_helpers_parse_lattice_and_thin_labels(self):
        manual = peak_overlay.parse_two_theta_peaks("31.8 TiO2, 38.5")
        self.assertEqual(len(manual), 2)
        self.assertAlmostEqual(manual[0].two_theta, 31.8)
        self.assertEqual(manual[0].label, "TiO2")

        peaks = peak_overlay.lattice_predicted_peaks(
            "fcc",
            a=4.05,
            wavelength=1.5406,
            max_index=3,
            min_two_theta=20,
            max_two_theta=90,
            phase_name="Al",
        )
        self.assertTrue(peaks)
        self.assertTrue(all(20 <= peak.two_theta <= 90 for peak in peaks))
        self.assertIn("Al", peaks[0].label)
        rounded_positions = [round(peak.two_theta, 5) for peak in peaks]
        self.assertEqual(len(rounded_positions), len(set(rounded_positions)))

        thinned = peak_overlay.thin_labelled_peaks(
            [
                peak_overlay.PredictedPeak(30.0, "a"),
                peak_overlay.PredictedPeak(30.2, "b"),
                peak_overlay.PredictedPeak(32.0, "c"),
            ],
            min_spacing=1.0,
        )
        self.assertEqual([peak.label for peak in thinned], ["a", "c"])
        unique = peak_overlay.unique_peak_positions(
            [
                peak_overlay.PredictedPeak(30.0, "a"),
                peak_overlay.PredictedPeak(30.000001, "b"),
                peak_overlay.PredictedPeak(31.0, "c"),
            ]
        )
        self.assertEqual([peak.label for peak in unique], ["a", "c"])

    def test_spectrum_plot_accepts_predicted_peak_overlay(self):
        from quixrd.nxs_export import XRD_spectra_anal

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "I_vs_2th_1_chi_0.txt"
            path.write_text(
                "\n".join(
                    [
                        "# Scan Type: ascan_chi",
                        "# Energy: 12",
                        "30.0 1",
                        "31.0 3",
                        "32.0 1",
                    ]
                ),
                encoding="utf-8",
            )
            spectrum = XRD_spectra_anal.Spectrum(tmp)
            with mock.patch.object(XRD_spectra_anal.plt, "show"):
                spectrum.plot_Ivs2theta(
                    [1],
                    plot_only=["chi"],
                    predicted_peaks={"source": "list", "two_theta_list": "31.0 test"},
                )
            XRD_spectra_anal.plt.close("all")

    def test_twotheta_calibration_from_txt_frames_and_double_correction_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wavelength = 1.0
            peaks = peak_overlay.lattice_predicted_peaks(
                "cubic",
                tth_cal.LAB6_A,
                wavelength=wavelength,
                max_index=5,
                min_two_theta=12,
                max_two_theta=70,
                phase_name="LaB6",
            )[:5]
            self.assertGreaterEqual(len(peaks), 3)

            def offset(x):
                return 0.02 + 0.0002 * x

            def intensity_at(x):
                y = np.full_like(x, 20.0, dtype=float)
                for peak in peaks:
                    observed = peak.two_theta + offset(peak.two_theta)
                    y += 800.0 * np.exp(-0.5 * ((x - observed) / 0.035) ** 2)
                return y

            frame_paths = []
            for idx, (low, high) in enumerate([(10.0, 43.0), (35.0, 75.0)]):
                x = np.arange(low, high, 0.01)
                y = intensity_at(x)
                path = tmp_path / f"I_vs_2th_50_delta_{idx}.txt"
                path.write_text(
                    "\n".join(
                        ["# Scan Type: ascan_delta", "# Energy: 12.3984193", "# 2theta intensity"]
                        + [f"{xx:.5f} {yy:.6f}" for xx, yy in zip(x, y)]
                    ),
                    encoding="utf-8",
                )
                frame_paths.append(path)

            result = tth_cal.build_twotheta_calibration(
                frame_paths,
                source_type="txt",
                output_dir=tmp_path / "calibration_out",
                wavelength=wavelength,
                polynomial_degree=1,
                show_plots=False,
            )
            calibration = tth_cal.load_calibration(result["path"])
            self.assertTrue(Path(result["combined_txt"]).exists())
            self.assertTrue(Path(result["combined_csv"]).exists())
            self.assertTrue(Path(result["profile_plot"]).exists())
            self.assertTrue(Path(result["profile_plot_svg"]).exists())
            self.assertTrue(Path(result["fit_plot"]).exists())
            self.assertTrue(Path(result["fit_plot_svg"]).exists())
            self.assertEqual(calibration["material"], "LaB6 (cubic, Pm-3m)")
            self.assertEqual(calibration["overlap_policy"], "blend")
            self.assertGreaterEqual(sum(1 for peak in calibration["peaks"] if peak["usable"]), 3)
            self.assertIn("caglioti", calibration)
            estimated = np.polyval(calibration["offset_polynomial_coefficients"], [30.0])[0]
            self.assertAlmostEqual(estimated, offset(30.0), delta=0.025)

            corrected = tth_cal.apply_calibration_to_txt_file(frame_paths[0], result["path"])
            self.assertTrue(corrected["applied"])
            corrected_again = tth_cal.apply_calibration_to_txt_file(frame_paths[0], result["path"])
            self.assertFalse(corrected_again["applied"])

    def test_twotheta_calibration_shows_generated_figures_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wavelength = 1.0
            peaks = peak_overlay.lattice_predicted_peaks(
                "cubic",
                tth_cal.LAB6_A,
                wavelength=wavelength,
                max_index=5,
                min_two_theta=12,
                max_two_theta=70,
                phase_name="LaB6",
            )[:5]
            x = np.arange(10.0, 75.0, 0.01)
            y = np.full_like(x, 20.0, dtype=float)
            for peak in peaks:
                y += 800.0 * np.exp(-0.5 * ((x - peak.two_theta) / 0.035) ** 2)
            path = tmp_path / "I_vs_2th_70_delta_0.txt"
            path.write_text(
                "\n".join(
                    ["# Scan Type: ascan_delta", "# Energy: 12.3984193", "# 2theta intensity"]
                    + [f"{xx:.5f} {yy:.6f}" for xx, yy in zip(x, y)]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(tth_cal.plt, "show") as shown:
                result = tth_cal.build_twotheta_calibration(
                    [path],
                    source_type="txt",
                    output_dir=tmp_path / "shown",
                    wavelength=wavelength,
                    polynomial_degree=1,
                    show_plots=True,
                )

            self.assertEqual(shown.call_count, 1)
            self.assertTrue(Path(result["profile_plot"]).exists())
            self.assertTrue(Path(result["fit_plot"]).exists())
            tth_cal.plt.close("all")

    def test_twotheta_calibration_from_single_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            x = np.arange(20.0, 45.0, 0.02)
            y = 10 + 500 * np.exp(-0.5 * ((x - 30.02) / 0.04) ** 2)
            csv_path = tmp_path / "scan_60_delta.csv"
            pd.DataFrame(
                {
                    "frame_index": [0] * len(x),
                    "2theta": x,
                    "intensity": y,
                    "energy": [12.3984193] * len(x),
                }
            ).to_csv(csv_path, index=False)
            profiles = tth_cal.read_csv_profiles(csv_path)
            self.assertEqual(len(profiles), 1)
            combined = tth_cal.combine_profiles(profiles)
            self.assertGreater(len(combined["two_theta"]), 100)

    def test_calibration_peak_fit_seeds_large_offset_from_observed_peak(self):
        x = np.arange(20.0, 30.0, 0.01)
        expected = 24.0
        observed = 25.05
        y = 15.0 + 2000.0 * np.exp(-0.5 * ((x - observed) / 0.05) ** 2)

        fit = tth_cal.fit_expected_peak(x, y, expected, window=2.0)

        self.assertAlmostEqual(fit["observed_seed"], observed, delta=0.03)
        self.assertAlmostEqual(fit["center"], observed, delta=0.03)

        fitted, shift_summary = tth_cal.fit_calibration_peaks(
            x,
            y,
            [peak_overlay.PredictedPeak(expected, "(100)", hkl=(1, 0, 0))],
            fit_window=2.0,
        )
        self.assertAlmostEqual(shift_summary["initial_shift"], observed - expected, delta=0.03)
        self.assertTrue(fitted[0]["usable"])
        self.assertAlmostEqual(fitted[0]["offset"], observed - expected, delta=0.03)

    def test_initial_shift_prefers_prominent_peak_when_expected_sits_between_peaks(self):
        x = np.arange(20.0, 40.0, 0.01)
        expected = 30.0
        lower_peak = 29.55
        higher_peak = 30.45
        y = 10.0
        y = y + 700.0 * np.exp(-0.5 * ((x - lower_peak) / 0.04) ** 2)
        y = y + 1800.0 * np.exp(-0.5 * ((x - higher_peak) / 0.04) ** 2)
        peaks = [peak_overlay.PredictedPeak(expected, "(111)", hkl=(1, 1, 1))]

        shift, matches = tth_cal.estimate_initial_twotheta_shift(x, y, peaks, search_window=1.0)
        fitted, _summary = tth_cal.fit_calibration_peaks(x, y, peaks, fit_window=0.5, initial_shift=shift)

        self.assertAlmostEqual(matches[0]["observed_two_theta"], higher_peak, delta=0.03)
        self.assertAlmostEqual(shift, higher_peak - expected, delta=0.03)
        self.assertTrue(fitted[0]["usable"])
        self.assertAlmostEqual(fitted[0]["center"], higher_peak, delta=0.03)

    def test_calibration_assignment_uses_smooth_global_offset_and_skips_unmatched_lines(self):
        x = np.arange(20.0, 70.0, 0.01)
        expected = [25.0, 35.0, 45.0, 55.0, 65.0]
        peaks = [peak_overlay.PredictedPeak(value, f"({idx})", hkl=(idx, 0, 0)) for idx, value in enumerate(expected)]
        y = np.full_like(x, 10.0, dtype=float)
        for value in expected[:-1]:
            observed = value + 0.4 + 0.015 * value
            y += 1500.0 * np.exp(-0.5 * ((x - observed) / 0.04) ** 2)
        y += 1800.0 * np.exp(-0.5 * ((x - 64.4) / 0.04) ** 2)

        fitted, summary = tth_cal.fit_calibration_peaks(x, y, peaks, fit_window=2.0)

        usable = [peak for peak in fitted if peak.get("usable")]
        self.assertEqual(len(usable), 4)
        self.assertEqual(summary["observed_peak_count"], 5)
        self.assertTrue(fitted[-1].get("error"))
        for row in usable:
            expected_offset = 0.4 + 0.015 * row["expected_two_theta"]
            self.assertAlmostEqual(row["offset"], expected_offset, delta=0.03)

    def test_calibration_outlier_exclusion_marks_points_without_hiding_them(self):
        fitted = []
        for idx, expected in enumerate(np.linspace(20.0, 90.0, 10)):
            offset = 0.05 + 0.001 * expected
            fwhm = 0.08 + 0.0004 * expected
            if idx == 4:
                offset += 0.45
            if idx == 7:
                fwhm = 0.65
            fitted.append(
                {
                    "usable": True,
                    "expected_two_theta": float(expected),
                    "offset": float(offset),
                    "center_err": 0.002,
                    "fwhm": float(fwhm),
                    "fwhm_err": 0.002,
                }
            )

        summary = tth_cal.annotate_calibration_fit_outliers(fitted, polynomial_degree=1, discard_outliers=True)

        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["offset_outliers"], 1)
        self.assertEqual(summary["caglioti_outliers"], 1)
        self.assertTrue(fitted[4]["offset_fit_outlier"])
        self.assertTrue(fitted[7]["caglioti_fit_outlier"])

    def test_spinodal_two_peak_fit_and_series_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.linspace(39.0, 41.0, 401)
            for scan, shift in [(10, 0.0), (11, 0.04)]:
                y = (
                    20.0
                    + 900.0 * np.exp(-np.log(2.0) * ((x - (39.82 + shift)) / 0.08) ** 2)
                    + 760.0 * np.exp(-np.log(2.0) * ((x - (40.18 + shift)) / 0.10) ** 2)
                )
                path = root / f"I_vs_2th_{scan}_chi_0.txt"
                path.write_text(
                    "\n".join(
                        ["# Scan Type: ascan_chi", f"# Temperature: {300 + scan}", "# 2theta intensity"]
                        + [f"{xx:.5f} {yy:.8f}" for xx, yy in zip(x, y)]
                    ),
                    encoding="utf-8",
                )

            profile = spinodal.read_txt_profile(root / "I_vs_2th_10_chi_0.txt")
            fit = spinodal.fit_peak_models(
                profile["two_theta"],
                profile["intensity"],
                peak_center=40.0,
                fit_window=0.7,
                fit_mode="compare",
            )

            self.assertTrue(fit["two_peak_preferred"])
            self.assertGreater(fit["delta_bic"], 10.0)
            self.assertAlmostEqual(fit["two"]["center_1"], 39.82, delta=0.03)
            self.assertAlmostEqual(fit["two"]["center_2"], 40.18, delta=0.03)

            messages = []
            result = spinodal.run_peak_series(
                root,
                scans=[10, 11],
                scan_type="chi",
                frame_index=0,
                peak_center=40.0,
                fit_window=0.7,
                fit_mode="compare",
                diagnostic_all_fits=True,
                show=False,
                progress_callback=messages.append,
            )
            df = result["data"]
            self.assertEqual(len(df), 2)
            self.assertTrue(Path(result["csv_path"]).exists())
            self.assertTrue(Path(result["plot_path"]).exists())
            self.assertTrue(Path(result["diagnostic_plot_path"]).exists())
            self.assertTrue(Path(result["diagnostic_plot_path"]).parent.name.startswith("diagnostics_"))
            self.assertEqual(len(result["diagnostic_plot_paths"]), 2)
            for diagnostic_path in result["diagnostic_plot_paths"]:
                path = Path(diagnostic_path)
                self.assertTrue(path.parent.name.startswith("diagnostics_"))
                self.assertIn("_scan_", path.name)
                self.assertTrue(path.exists())
            saved_messages = [message for message in messages if "saved diagnostic" in message]
            self.assertEqual(len(saved_messages), 2)
            self.assertTrue((df["selected_model"] == "two").all())
            self.assertNotIn("two_peak_preferred", df.columns)
            self.assertIn("center_2", df.columns)
            self.assertIn("minor_major_height_ratio", df.columns)

            replot = spinodal.plot_peak_series_from_csv(result["csv_path"], x="temperature", save=True, show=False)
            self.assertTrue(Path(replot["plot_path"]).exists())
            self.assertEqual(replot["x"], "temperature")

    def test_spinodal_two_peak_fit_finds_clear_shoulder(self):
        x = np.linspace(28.5, 30.1, 450)
        y = (
            205.0
            - 5.0 * (x - 29.3)
            + 410.0 * np.exp(-np.log(2.0) * ((x - 29.06) / 0.20) ** 2)
            + 170.0 * np.exp(-np.log(2.0) * ((x - 29.27) / 0.11) ** 2)
        )

        fit = spinodal.fit_peak_models(
            x,
            y,
            peak_center=29.3,
            fit_window=0.8,
            fit_mode="compare",
        )

        self.assertTrue(fit["two_peak_preferred"])
        self.assertGreater(fit["delta_bic"], 10.0)
        self.assertGreaterEqual(fit["two"]["initial_guess_count"], 2)
        self.assertGreater(fit["two"]["minor_major_height_ratio"], 0.1)
        self.assertLessEqual(fit["two"]["minor_major_height_ratio"], 1.0)
        self.assertAlmostEqual(fit["two"]["center_1"], 29.06, delta=0.04)
        self.assertAlmostEqual(fit["two"]["center_2"], 29.27, delta=0.04)

    def test_spinodal_delta_scans_are_homogenised_and_show_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x1 = np.linspace(39.0, 40.2, 241)
            x2 = np.linspace(39.8, 41.0, 241)
            for idx, x in enumerate([x1, x2]):
                y = 10.0 + 1000.0 * np.exp(-np.log(2.0) * ((x - 40.05) / 0.08) ** 2)
                path = root / f"I_vs_2th_20_delta_{idx}.txt"
                path.write_text(
                    "\n".join(
                        ["# Scan Type: ascan_delta", "# Temperature: 330", "# 2theta intensity"]
                        + [f"{xx:.5f} {yy:.8f}" for xx, yy in zip(x, y)]
                    ),
                    encoding="utf-8",
                )

            combined = spinodal.load_scan_profile(root, 20, scan_type="delta", frame_index=None)
            self.assertTrue(combined["metadata"]["homogenised_profile"])
            self.assertEqual(combined["metadata"]["source_frame_count"], 2)

            with mock.patch.object(spinodal.plt, "show") as shown:
                messages = []
                result = spinodal.run_peak_series(
                    root,
                    scans=[20],
                    scan_type="delta",
                    frame_index=None,
                    peak_center=40.05,
                    fit_window=0.4,
                    fit_mode="single",
                    show=True,
                    progress_callback=messages.append,
                )

            shown.assert_called_once()
            self.assertIn("Peak Analysis: fitting 1 scan(s)", messages)
            self.assertTrue(any("scan 20 fitted" in message for message in messages))
            self.assertTrue(any("finished (1 succeeded, 0 failed)" in message for message in messages))
            df = result["data"]
            self.assertTrue(pd.isna(df.loc[0, "frame_index"]))
            self.assertTrue(df.loc[0, "homogenised_profile"])
            self.assertNotIn("delta_bic", df.columns)
            self.assertAlmostEqual(df.loc[0, "center_1"], 40.05, delta=0.03)
            self.assertTrue(Path(result["diagnostic_plot_path"]).exists())
            self.assertTrue(Path(result["diagnostic_plot_path"]).parent.name.startswith("diagnostics_"))

    def test_spinodal_trend_panel_count_depends_on_comparison(self):
        compare_df = pd.DataFrame(
            {
                "scan_number": [1, 2],
                "start_time": ["2026-07-24T10:00:00", "2026-07-24T10:05:00"],
                "center_1": [40.0, 40.1],
                "fwhm_1": [0.1, 0.1],
                "delta_bic": [12.0, 15.0],
                "minor_major_height_ratio": [0.25, 0.4],
                "selected_model": ["single", "two"],
                "single_center_1": [40.0, 40.1],
                "single_fwhm_1": [0.1, 0.1],
                "two_center_1": [39.95, 40.05],
                "two_center_2": [40.08, 40.18],
                "two_fwhm_1": [0.08, 0.08],
                "two_fwhm_2": [0.09, 0.09],
            }
        )
        single_df = compare_df.drop(
            columns=[
                "delta_bic",
                "minor_major_height_ratio",
                "selected_model",
                "single_center_1",
                "single_fwhm_1",
                "two_center_1",
                "two_center_2",
                "two_fwhm_1",
                "two_fwhm_2",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            compare_fig = spinodal._plot_trends(Path(tmp) / "compare.png", compare_df, show=True)
            time_fig = spinodal._plot_trends(Path(tmp) / "time.png", compare_df, x="start_time", show=True)
            single_fig = spinodal._plot_trends(Path(tmp) / "single.png", single_df, show=True)
            try:
                self.assertEqual(len(compare_fig.axes), 4)
                self.assertEqual(len(time_fig.axes), 4)
                self.assertNotEqual(type(time_fig.axes[2].xaxis.get_major_locator()).__name__, "StrCategoryLocator")
                self.assertEqual(len(single_fig.axes), 2)
            finally:
                spinodal.plt.close(compare_fig)
                spinodal.plt.close(time_fig)
                spinodal.plt.close(single_fig)

    def test_spinodal_fit_downsamples_dense_windows_and_caps_optimizer(self):
        x = np.linspace(39.0, 41.0, 5000)
        y = 10.0 + 1000.0 * np.exp(-np.log(2.0) * ((x - 40.0) / 0.08) ** 2)

        fit = spinodal.fit_peak_models(x, y, peak_center=40.0, fit_window=0.8, fit_mode="single")

        self.assertEqual(fit["data_point_count"], 4000)
        self.assertLessEqual(fit["fit_point_count"], spinodal.MAX_FIT_POINTS)
        self.assertAlmostEqual(fit["single"]["center_1"], 40.0, delta=0.02)

    def test_stress_with_reference_d0_and_equibiaxial_fallback(self):
        sin2psi = np.array([0.0, 0.25, 0.5, 0.75])
        wavelength = 1.0
        d0 = 2.0
        strain_slope = 0.001
        d_values = d0 * (1.0 + strain_slope * sin2psi)
        two_theta = np.degrees(2.0 * np.arcsin(wavelength / (2.0 * d_values)))
        df = pd.DataFrame(
            {
                "sin2psi": sin2psi,
                "peak_center": two_theta,
                "peak_center_err": [0.001] * len(sin2psi),
                "excluded_from_sin2psi": [False] * len(sin2psi),
            }
        )

        reference = proc.calculate_sin2psi_stress(
            df,
            elastic_E=200000,
            elastic_nu=0.3,
            elastic_E_units="MPa",
            reference_d0=d0,
            wavelength=wavelength,
        )
        self.assertAlmostEqual(reference["stress"], 200000 / 1.3 * strain_slope, delta=1.0)
        self.assertEqual(reference["stress_method"], "reference_d0")
        self.assertEqual(reference["stress_units"], "MPa")

        equibiaxial = proc.calculate_sin2psi_stress(
            df,
            elastic_E=200000,
            elastic_nu=0.3,
            wavelength=wavelength,
        )
        ratio = (d0 * strain_slope) / d0
        expected = 200000 * ratio / (1.3 + 0.6 * ratio)
        self.assertAlmostEqual(equibiaxial["stress"], expected, delta=1.0)
        self.assertEqual(equibiaxial["stress_method"], "equibiaxial_inferred_d0")

    def test_perform_sin2psi_fit_writes_optional_stress_fields(self):
        sin2psi = np.array([0.0, 0.25, 0.5, 0.75])
        wavelength = 1.0
        d0 = 2.0
        d_values = d0 * (1.0 + 0.001 * sin2psi)
        two_theta = np.degrees(2.0 * np.arcsin(wavelength / (2.0 * d_values)))
        psi = np.degrees(np.arcsin(np.sqrt(sin2psi)))
        chi = 90.0 - psi
        df = pd.DataFrame(
            {
                "frame_index": range(len(sin2psi)),
                "chi": chi,
                "peak_center": two_theta,
                "peak_center_err": [0.001] * len(sin2psi),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            scan_dir = Path(tmp) / "scan_9"
            scan_dir.mkdir()
            summary = proc.perform_sin2psi_fit(
                df,
                scan_dir,
                elastic_E=200000,
                elastic_nu=0.3,
                elastic_E_units="MPa",
                stress_reference_d0=d0,
                stress_wavelength=wavelength,
            )
            saved = json.loads((scan_dir / "sin2psi_fit_params.json").read_text(encoding="utf-8"))
        self.assertIn("stress", summary)
        self.assertEqual(saved["stress_method"], "reference_d0")
        self.assertEqual(saved["stress_units"], "MPa")
        self.assertIn("stress_err", saved)

    def test_process_scan_writes_tracking_metadata_to_csv(self):
        source_files = sorted((self.repo_root / "export").glob("I_vs_2th_440_chi_*.txt"))
        self.assertGreaterEqual(len(source_files), 2, "Need at least two scan 440 chi frames")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            copied = []
            for source in source_files:
                destination = tmp_path / source.name
                shutil.copy2(source, destination)
                copied.append(str(destination))

            result = proc.process_scan(
                data_dir=tmp_path,
                scan_number=440,
                files=copied,
                plot_frames=False,
                peak_center=None,
                track_peak=True,
                track_window=0.4,
            )
            df = pd.read_csv(result["csv_path"])
            with open(Path(result["scan_dir"]) / "sin2psi_fit_params.json", "r", encoding="utf-8") as fh:
                summary_json = json.load(fh)

        for column in [
            "window_mode",
            "seed_center",
            "background_lower",
            "peak_lower",
            "peak_upper",
            "background_upper",
            "start_time",
            "frame_time",
            "metadata_json",
        ]:
            self.assertIn(column, df.columns)
        self.assertEqual(df.loc[0, "window_mode"], "auto")
        self.assertTrue((df.loc[1:, "window_mode"] == "seeded").any())
        self.assertIn("metadata", summary_json)
        self.assertEqual(summary_json["metadata"]["temperature"], 639.0)
        self.assertEqual(summary_json["metadata"]["scan_type"], "ascan_chi")

    def test_collect_sin2psi_summaries_reads_json_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "sin2psi_export"
            scan_dir = export_root / "scan_101"
            scan_dir.mkdir(parents=True)
            (scan_dir / "sin2psi_fit_params.json").write_text(
                json.dumps(
                    {
                        "slope": 1.2,
                        "slope_err": 0.3,
                        "intercept": 4.5,
                        "intercept_err": 0.6,
                        "n_points": 5,
                        "metadata": {"temperature": 450.0, "start_time": "2026-01-01T00:00:00"},
                    }
                ),
                encoding="utf-8",
            )

            df = proc.collect_sin2psi_summaries(tmp)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "scan_number"], 101)
        self.assertEqual(df.loc[0, "slope"], 1.2)
        self.assertEqual(df.loc[0, "temperature"], 450.0)

    def test_collect_sin2psi_summaries_falls_back_to_csv_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "sin2psi_export"
            scan_dir = export_root / "scan_102"
            scan_dir.mkdir(parents=True)
            (scan_dir / "sin2psi_fit_params.json").write_text(
                json.dumps({"slope": 2.0, "slope_err": 0.2, "intercept": 5.0, "intercept_err": 0.5}),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "frame_index": 0,
                        "filename": "I_vs_2th_102_chi_0.txt",
                        "scan_type": "ascan_chi",
                        "chi": 90.0,
                        "psi_deg": 0.0,
                        "sin2psi": 0.0,
                        "temperature": 500.0,
                        "energy": 9.5,
                        "start_time": "2026-01-01T01:00:00",
                        "frame_time": "2026-01-01T01:00:01",
                        "metadata_json": json.dumps({"operator": "test"}),
                    }
                ]
            ).to_csv(scan_dir / "scan_102_fits.csv", index=False)

            df = proc.collect_sin2psi_summaries(tmp)

        self.assertEqual(df.loc[0, "temperature"], 500.0)
        self.assertEqual(df.loc[0, "operator"], "test")

    def test_plot_sin2psi_gradients_writes_errorbar_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "sin2psi_export"
            for scan, slope, err, temp in [(101, 1.0, 0.1, 300.0), (102, 1.5, 0.2, 350.0)]:
                scan_dir = export_root / f"scan_{scan}"
                scan_dir.mkdir(parents=True)
                (scan_dir / "sin2psi_fit_params.json").write_text(
                    json.dumps(
                        {
                            "slope": slope,
                            "slope_err": err,
                            "intercept": 4.0,
                            "intercept_err": 0.4,
                            "metadata": {"temperature": temp},
                        }
                    ),
                    encoding="utf-8",
                )

            result = proc.plot_sin2psi_gradients(tmp, x="temperature", show=False)

            self.assertTrue(Path(result["plot_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            self.assertIn("temperature", Path(result["plot_path"]).name)
            self.assertRegex(Path(result["summary_path"]).name, r"^sin2psi_scan_summary_\d{8}_\d{6}\.csv$")

    def test_plot_sin2psi_gradients_reuses_unchanged_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "sin2psi_export"
            scan_dir = export_root / "scan_101"
            scan_dir.mkdir(parents=True)
            json_path = scan_dir / "sin2psi_fit_params.json"
            json_path.write_text(
                json.dumps(
                    {
                        "slope": 1.0,
                        "slope_err": 0.1,
                        "intercept": 4.0,
                        "intercept_err": 0.4,
                        "metadata": {"temperature": 300.0},
                    }
                ),
                encoding="utf-8",
            )

            first = proc.plot_sin2psi_gradients(tmp, x="temperature", show=False)
            second = proc.plot_sin2psi_gradients(tmp, x="temperature", show=False)
            self.assertEqual(first["summary_path"], second["summary_path"])

            json_path.write_text(
                json.dumps(
                    {
                        "slope": 1.5,
                        "slope_err": 0.1,
                        "intercept": 4.0,
                        "intercept_err": 0.4,
                        "metadata": {"temperature": 300.0},
                    }
                ),
                encoding="utf-8",
            )
            third = proc.plot_sin2psi_gradients(tmp, x="temperature", show=False)
            self.assertNotEqual(first["summary_path"], third["summary_path"])

    def test_plot_sin2psi_gradients_reuses_matching_older_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "sin2psi_export"
            for scan, slope, temp in [(101, 1.0, 300.0), (102, 2.0, 310.0)]:
                scan_dir = export_root / f"scan_{scan}"
                scan_dir.mkdir(parents=True)
                (scan_dir / "sin2psi_fit_params.json").write_text(
                    json.dumps(
                        {
                            "slope": slope,
                            "slope_err": 0.1,
                            "intercept": 4.0,
                            "intercept_err": 0.4,
                            "metadata": {"temperature": temp},
                        }
                    ),
                    encoding="utf-8",
                )

            all_scans = proc.plot_sin2psi_gradients(tmp, x="temperature", show=False)
            subset = proc.plot_sin2psi_gradients(tmp, x="temperature", scans=[101], show=False)
            repeated_all_scans = proc.plot_sin2psi_gradients(tmp, x="temperature", show=False)

            self.assertNotEqual(all_scans["summary_path"], subset["summary_path"])
            self.assertEqual(all_scans["summary_path"], repeated_all_scans["summary_path"])

    def test_plot_sin2psi_gradients_can_use_selected_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp) / "chosen_summary.csv"
            pd.DataFrame(
                [
                    {"scan_number": 101, "slope": 1.0, "slope_err": 0.1, "temperature": 300.0},
                    {"scan_number": 102, "slope": 1.5, "slope_err": 0.2, "temperature": 350.0},
                ]
            ).to_csv(selected, index=False)

            result = proc.plot_sin2psi_gradients(tmp, x="temperature", scans=[102], show=False, summary_csv=selected)

            self.assertEqual(Path(result["summary_path"]), selected)
            self.assertEqual(result["summary"].loc[0, "scan_number"], 102)
            self.assertTrue(Path(result["plot_path"]).exists())

    def test_scan_title_uses_plotted_scans_when_selection_is_empty(self):
        self.assertEqual(proc._scan_title(None, [101, 102, 103]), "scans 101-103")
        self.assertEqual(proc._scan_title(None, [101, 103]), "scans 101, 103")

    def test_selector_title_omits_chi_tolerance(self):
        self.assertEqual(proc._selector_title("chi_5_0"), "chi=5.0\u00b0")
        self.assertEqual(proc._selector_title("frame_2"), "frame 2")

    def test_point_limited_y_axis_ignores_large_errorbars(self):
        fig, ax = proc.plt.subplots()
        try:
            ax.errorbar([1, 2], [10, 11], yerr=[1000, 1000], fmt=".-")
            proc._set_ylim_from_points(ax, [10, 11])
            ymin, ymax = ax.get_ylim()
            self.assertGreater(ymin, 0)
            self.assertLess(ymax, 20)
        finally:
            proc.plt.close(fig)

    def test_processing_params_round_trip_and_workflow_override_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            params_path = Path(tmp) / "params.json"
            params = proc.build_processing_params(
                data_dir="old",
                processing_options={"peak_center": 99.0, "track_window": 0.2},
            )
            params_path.write_text(json.dumps(params), encoding="utf-8")

            loaded = proc.load_processing_params(params_path)
            self.assertEqual(loaded["processing_options"]["peak_center"], 99.0)

            with mock.patch.object(run_workflow, "DATA_DIR", tmp), mock.patch.object(
                run_workflow, "EXCLUDE_FRAMES", [2]
            ), mock.patch.object(run_workflow.proc, "discover_scan_files", return_value=[]), mock.patch(
                "logging.error"
            ), mock.patch("builtins.print"):
                log_path = run_workflow.sin2psi_scans_fit(
                    [123],
                    peak_center=12.3,
                    track_window=0.7,
                    params_json=params_path,
                )

            saved = proc.load_processing_params(log_path)
            self.assertEqual(saved["processing_options"]["peak_center"], 12.3)
            self.assertEqual(saved["processing_options"]["track_window"], 0.7)
            self.assertEqual(saved["exclude_frames"], [2])
            self.assertEqual(saved["scan_results"][0]["status"], "error")

    def test_collect_and_plot_fwhm_by_frame_and_exact_chi(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "sin2psi_export"
            for scan, temp in [(101, 300.0), (102, 350.0)]:
                scan_dir = export_root / f"scan_{scan}"
                scan_dir.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {
                            "frame_index": 0,
                            "filename": f"I_vs_2th_{scan}_chi_0.txt",
                            "scan_type": "ascan_chi",
                            "chi": 0.0,
                            "psi_deg": 90.0,
                            "sin2psi": 1.0,
                            "temperature": temp,
                            "energy": 12.0,
                            "start_time": "2026-01-01T00:00:00",
                            "frame_time": "2026-01-01T00:00:01",
                            "metadata_json": json.dumps({"operator": "test"}),
                            "fwhm": 0.2 + scan / 1000.0,
                            "fwhm_err": 0.01,
                            "peak_center": 20.0 + scan / 1000.0,
                            "peak_center_err": 0.001,
                            "fit_success": True,
                        },
                        {
                            "frame_index": 1,
                            "filename": f"I_vs_2th_{scan}_chi_1.txt",
                            "scan_type": "ascan_chi",
                            "chi": 5.05 if scan == 102 else 5.0,
                            "psi_deg": 85.0,
                            "sin2psi": 0.99,
                            "temperature": temp,
                            "energy": 12.0,
                            "fwhm": 0.3 + scan / 1000.0,
                            "fwhm_err": 0.02,
                            "peak_center": 20.2 + scan / 1000.0,
                            "peak_center_err": 0.002,
                            "fit_success": True,
                        },
                    ]
                ).to_csv(scan_dir / f"scan_{scan}_fits.csv", index=False)

            by_frame = proc.collect_fwhm_summaries(tmp, frame_index=1)
            by_chi = proc.collect_fwhm_summaries(tmp, chi=5.0)
            self.assertEqual(len(by_frame), 2)
            self.assertEqual(len(by_chi), 2)
            self.assertTrue(((by_chi["chi"] - 5.0).abs() <= 0.1).all())

            by_chi_multi = proc.collect_fwhm_summaries(tmp, chi=[0.0, 5.0])
            self.assertEqual(len(by_chi_multi), 4)
            self.assertEqual(set(by_chi_multi["selector"]), {"chi_0_0", "chi_5_0"})

            by_frame_multi = proc.collect_fwhm_summaries(tmp, frame_index=[0, 1])
            self.assertEqual(len(by_frame_multi), 4)
            self.assertEqual(set(by_frame_multi["selector"]), {"frame_0", "frame_1"})

            result = proc.plot_fwhm_trends(tmp, x="temperature", chi=5.0, show=False)
            self.assertTrue(Path(result["plot_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            self.assertIn("temperature", Path(result["plot_path"]).name)

            multi_result = proc.plot_fwhm_trends(tmp, x="scan_number", chi=[0.0, 5.0], show=False)
            self.assertTrue(Path(multi_result["plot_path"]).exists())
            self.assertIn("multi", Path(multi_result["plot_path"]).name)

            multi_frame_result = proc.plot_fwhm_trends(tmp, x="scan_number", frame_index=[0, 1], show=False)
            self.assertTrue(Path(multi_frame_result["plot_path"]).exists())
            self.assertIn("multi", Path(multi_frame_result["plot_path"]).name)

            peak_summary = proc.collect_peak_position_summaries(tmp, chi=[0.0, 5.0])
            self.assertEqual(len(peak_summary), 4)
            peak_result = proc.plot_peak_position_trends(tmp, x="temperature", chi=[0.0, 5.0], show=False)
            self.assertTrue(Path(peak_result["plot_path"]).exists())
            self.assertIn("peak_position", Path(peak_result["plot_path"]).name)

            peak_frame_result = proc.plot_peak_position_trends(tmp, x="temperature", frame_index=[0, 1], show=False)
            self.assertTrue(Path(peak_frame_result["plot_path"]).exists())
            self.assertIn("peak_position", Path(peak_frame_result["plot_path"]).name)

    def test_refit_sin2psi_from_csv_applies_range_exclusions_without_peak_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            scan_dir = Path(tmp) / "sin2psi_export" / "scan_10"
            scan_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"frame_index": 0, "chi": 90.0, "peak_center": 10.0, "peak_center_err": 0.1, "temperature": 300},
                    {"frame_index": 1, "chi": 60.0, "peak_center": 11.0, "peak_center_err": 0.1, "temperature": 300},
                    {"frame_index": 2, "chi": 30.0, "peak_center": 12.0, "peak_center_err": 0.1, "temperature": 300},
                    {"frame_index": 3, "chi": 0.0, "peak_center": 20.0, "peak_center_err": 0.1, "temperature": 300},
                ]
            ).to_csv(scan_dir / "scan_10_fits.csv", index=False)

            with mock.patch.object(proc, "fit_frame", side_effect=AssertionError("peak refit called")):
                result = proc.refit_sin2psi_from_csv(
                    tmp,
                    10,
                    excluded_frames=[0],
                    exclude_chi_ranges=[(0.0, 0.0)],
                    exclude_sin2psi_ranges=[(0.25, 0.25)],
                )

            df = pd.read_csv(result["csv_path"])
            summary = proc.load_processing_params(scan_dir / "sin2psi_fit_params.json")
            self.assertTrue(df.loc[df["frame_index"] == 0, "excluded_from_sin2psi"].iloc[0])
            self.assertTrue(df.loc[df["frame_index"] == 3, "excluded_from_sin2psi"].iloc[0])
            self.assertEqual(summary["excluded_frames"], [0])
            self.assertEqual(summary["exclude_chi_ranges"], [[0.0, 0.0]])

    def test_refit_sin2psi_from_csv_auto_excludes_outlier(self):
        with tempfile.TemporaryDirectory() as tmp:
            scan_dir = Path(tmp) / "sin2psi_export" / "scan_11"
            scan_dir.mkdir(parents=True)
            rows = []
            for frame, chi, center in [
                (0, 90.0, 10.0),
                (1, 60.0, 10.25),
                (2, 45.0, 10.5),
                (3, 30.0, 10.75),
                (4, 0.0, 20.0),
            ]:
                rows.append({"frame_index": frame, "chi": chi, "peak_center": center, "peak_center_err": 0.1})
            pd.DataFrame(rows).to_csv(scan_dir / "scan_11_fits.csv", index=False)

            result = proc.refit_sin2psi_from_csv(
                tmp,
                11,
                auto_exclude=True,
                auto_exclude_sigma=1.0,
                auto_exclude_max_iter=1,
            )

            summary = proc.load_processing_params(scan_dir / "sin2psi_fit_params.json")
            df = pd.read_csv(result["csv_path"])
            self.assertTrue(summary["auto_exclude"])
            self.assertGreaterEqual(len(summary["auto_excluded_frames"]), 1)
            self.assertTrue(df["excluded_from_sin2psi"].any())

    def test_generate_and_apply_sin2psi_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "sin2psi_export"
            ref_dir = export_root / "scan_20"
            ref_dir.mkdir(parents=True)
            chi_values = [90.0, 60.0, 45.0, 30.0, 0.0]
            ref_rows = []
            for idx, chi in enumerate(chi_values):
                psi = 90.0 - chi
                sin2psi = math.sin(math.radians(psi)) ** 2
                ref_rows.append(
                    {
                        "frame_index": idx,
                        "chi": chi,
                        "psi_deg": psi,
                        "sin2psi": sin2psi,
                        "peak_center": 20.0 + 0.2 * sin2psi,
                        "peak_center_err": 0.01,
                    }
                )
            pd.DataFrame(ref_rows).to_csv(ref_dir / "scan_20_fits.csv", index=False)

            correction_result = proc.generate_sin2psi_correction(
                tmp,
                20,
                degree=1,
                reference_two_theta=20.0,
            )
            correction = proc.load_sin2psi_correction(correction_result["path"])
            self.assertTrue(Path(correction_result["plot_path"]).exists())

            sample_dir = export_root / "scan_21"
            sample_dir.mkdir(parents=True)
            sample_rows = []
            for row in ref_rows:
                correction_at_ref = float(np.polyval(correction["coefficients"], row["sin2psi"]))
                sample_rows.append(
                    {
                        "frame_index": row["frame_index"],
                        "chi": row["chi"],
                        "psi_deg": row["psi_deg"],
                        "sin2psi": row["sin2psi"],
                        "peak_center": 40.0 + correction_at_ref,
                        "peak_center_err": 0.01,
                    }
                )
            pd.DataFrame(sample_rows).to_csv(sample_dir / "scan_21_fits.csv", index=False)

            result = proc.refit_sin2psi_from_csv(tmp, 21, correction_json=correction_result["path"])
            summary = proc.load_processing_params(sample_dir / "sin2psi_fit_params.json")
            df = pd.read_csv(result["csv_path"])

            self.assertTrue(summary["correction_applied"])
            self.assertEqual(summary["fit_y_column"], "peak_center_corrected")
            self.assertEqual(summary["correction_selection"]["selected_correction_file"], correction_result["path"])
            self.assertIn("peak_center_corrected", df.columns)
            self.assertTrue((sample_dir / "scan_21_sin2psi_plot.png").exists())
            self.assertAlmostEqual(summary["slope"], 0.0, delta=2e-6)

    def test_multiple_sin2psi_corrections_select_closest_reference_angle(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "sin2psi_export"
            sample_dir = export_root / "scan_22"
            sample_dir.mkdir(parents=True)
            low_path = Path(tmp) / "correction_low.json"
            high_path = Path(tmp) / "correction_high.json"
            low_path.write_text(
                json.dumps(
                    {
                        "method": "polynomial",
                        "coefficients": [9.0, 0.0],
                        "reference_two_theta": 20.0,
                        "reference_two_theta_provided": True,
                    }
                ),
                encoding="utf-8",
            )
            high_path.write_text(
                json.dumps(
                    {
                        "method": "polynomial",
                        "coefficients": [0.3, 0.0],
                        "reference_two_theta": 49.0,
                        "reference_two_theta_provided": True,
                    }
                ),
                encoding="utf-8",
            )

            rows = []
            for idx, chi in enumerate([90.0, 60.0, 45.0, 30.0, 0.0]):
                psi = 90.0 - chi
                sin2psi = math.sin(math.radians(psi)) ** 2
                rows.append(
                    {
                        "frame_index": idx,
                        "chi": chi,
                        "psi_deg": psi,
                        "sin2psi": sin2psi,
                        "peak_center": 50.0 + 0.3 * sin2psi,
                        "peak_center_err": 0.01,
                        "fit_success": True,
                    }
                )
            pd.DataFrame(rows).to_csv(sample_dir / "scan_22_fits.csv", index=False)

            result = proc.refit_sin2psi_from_csv(tmp, 22, correction_json=[low_path, high_path])
            summary = proc.load_processing_params(sample_dir / "sin2psi_fit_params.json")
            df = pd.read_csv(result["csv_path"])

            self.assertTrue(summary["correction_applied"])
            self.assertEqual(summary["correction_file"], str(high_path))
            self.assertEqual(summary["correction_selection"]["selected_correction_reference_two_theta"], 49.0)
            self.assertEqual(summary["correction_selection"]["selection_rule"], "first_frame_peak_center")
            self.assertAlmostEqual(summary["slope"], 0.0, delta=2e-6)
            self.assertIn("peak_center_corrected", df.columns)
            self.assertNotIn("correction_selection", df.columns)
            self.assertNotIn("scan_representative_two_theta", df.columns)

    def test_generate_gaussian_process_correction_defaults_to_true_angle(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp) / "sin2psi_export" / "scan_30"
            ref_dir.mkdir(parents=True)
            rows = []
            for idx, chi in enumerate([90.0, 60.0, 45.0, 30.0, 0.0]):
                psi = 90.0 - chi
                sin2psi = math.sin(math.radians(psi)) ** 2
                rows.append(
                    {
                        "frame_index": idx,
                        "chi": chi,
                        "psi_deg": psi,
                        "sin2psi": sin2psi,
                        "peak_center": 30.0 + 0.1 * sin2psi,
                        "peak_center_err": 0.01,
                    }
                )
            pd.DataFrame(rows).to_csv(ref_dir / "scan_30_fits.csv", index=False)

            result = proc.generate_sin2psi_correction(
                tmp,
                30,
                method="gaussian_process",
                gp_length_scale=0.5,
            )
            correction = proc.load_sin2psi_correction(result["path"])

            self.assertEqual(correction["method"], "gaussian_process")
            self.assertEqual(correction["y"], "peak_center")
            self.assertFalse(correction["reference_two_theta_provided"])
            self.assertIn("training_x", correction)
            self.assertTrue(Path(result["plot_path"]).exists())
            _, std = proc._gp_predict(
                [0.0, 0.5, 1.0],
                correction["training_x"],
                correction["training_y"],
                correction["training_noise"],
                correction["gp_length_scale"],
                correction["gp_signal_variance"],
                return_std=True,
            )
            self.assertTrue((std >= 0).all())

    def test_cli_module_import_path(self):
        completed = subprocess.run(
            [sys.executable, "-m", "quixrd.xrd_processing.cli", "--help"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--peak-center", completed.stdout)

    def test_gui_log_stream_emits_complete_lines(self):
        emitted = []
        stream = gui_app.GuiLogStream(emitted.append)
        stream.write("Local cache: started")
        self.assertEqual(emitted, [])
        stream.write(" now\nLocal cache: completed")
        self.assertEqual(emitted, ["Local cache: started now"])
        stream.flush()
        self.assertEqual(emitted[-1], "Local cache: completed")

    def test_gui_layout_import_and_tabs(self):
        try:
            root, app = gui_app.create_app()
        except Exception as exc:
            self.skipTest(f"Tkinter display unavailable: {exc}")
        try:
            root.withdraw()
            tabs = [app.notebook.tab(tab_id, "text") for tab_id in app.notebook.tabs()]
            self.assertEqual(tabs, gui_app.TAB_NAMES)
            self.assertNotIn("Help", tabs)
            self.assertTrue(hasattr(app, "menu_bar"))
            self.assertTrue(hasattr(app, "file_menu"))
            self.assertTrue(hasattr(app, "calibration_menu"))
            self.assertTrue(hasattr(app, "help_menu"))
            self.assertTrue(hasattr(app, "log_text"))
            self.assertTrue(hasattr(app, "status_bar"))
            self.assertTrue(gui_app.APP_ICON_PATH.exists())
            self.assertTrue(app._apply_window_icon(root))
            file_labels = [
                app.file_menu.entrycget(idx, "label")
                for idx in range(app.file_menu.index("end") + 1)
                if app.file_menu.type(idx) in {"command", "checkbutton"}
            ]
            self.assertIn("Select Local Cache Folder...", file_labels)
            self.assertIn("Use Local Cache", file_labels)
            self.assertNotIn("Create Local Cache", file_labels)
            calibration_labels = [
                app.calibration_menu.entrycget(idx, "label")
                for idx in range(app.calibration_menu.index("end") + 1)
                if app.calibration_menu.type(idx) in {"command", "checkbutton"}
            ]
            self.assertEqual(calibration_labels[0], "2theta Calibration...")
            self.assertIn("Select 2theta Calibration File...", calibration_labels)
            self.assertIn("Apply 2theta Calibration by Default", calibration_labels)
            for section_name in [
                "sin2psi.inputs",
                "sin2psi.peak_options",
                "sin2psi.exclusions",
                "sin2psi.stress",
                "sin2psi.calibration",
            ]:
                self.assertIn(section_name, app.sections)
            help_titles = [title for title, _text in app._help_sections()]
            self.assertEqual(
                help_titles,
                [
                    "Overview",
                    "File",
                    "Calibration",
                    "Extraction",
                    "Plotting",
                    "Sorting",
                    "Peak Analysis",
                    gui_app.SIN2PSI_LABEL,
                ],
            )

            with tempfile.TemporaryDirectory() as tmp:
                params_path = Path(tmp) / "gui_params.json"
                self.assertEqual(app._parse_int_list("440-450:5"), [440, 445, 450])
                self.assertEqual(app._parse_int_list("450-440:5"), [450, 445, 440])
                self.assertEqual(app._parse_int_list("440,450-460:5,470"), [440, 450, 455, 460, 470])
                self.assertEqual(app._parse_every_nth(":5"), 5)
                self.assertEqual(app._parse_int_list(":5"), [])
                app.variables["plot.scans"].set(":2")
                with mock.patch.object(gui_app.proc, "discover_scan_numbers", return_value=[100, 101, 102, 103, 104]):
                    self.assertEqual(app._selected_plot_scans(str(Path(tmp)), "spectra"), [100, 102, 104])
                with self.assertRaises(ValueError):
                    app._parse_int_list("440-450:0")
                with self.assertRaises(ValueError):
                    app._parse_int_list("440:5")
                app.variables["plot.x"].set("temperature")
                app.variables["extract.scans"].set("440-450")
                app.export_parameters(params_path, scope="all")
                app.variables["plot.x"].set("scan_number")
                app.import_parameters(params_path, scope="all")
                self.assertEqual(app.variables["plot.x"].get(), "temperature")

                app.notebook.select(app.tab_frames["Plotting"])
                current_path = Path(tmp) / "plot_params.json"
                app.export_parameters(current_path, scope="current")
                payload = json.loads(current_path.read_text(encoding="utf-8"))
                self.assertIn("plot.x", payload["parameters"])
                self.assertNotIn("extract.scans", payload["parameters"])

                settings_path = Path(tmp) / "gui_settings.json"
                cache_dir = Path(tmp) / "fast_cache"
                app.settings_path = settings_path
                app.settings = {}
                app.set_cache_root(cache_dir)
                self.assertEqual(app._cache_root(), cache_dir)
                saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))
                self.assertEqual(Path(saved_settings["cache_root"]), cache_dir)
                app.use_local_cache_var.set(True)
                app._sync_use_local_cache_setting()
                saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))
                self.assertTrue(saved_settings["use_local_cache"])
                app.twotheta_calibration_file.set(str(Path(tmp) / "calibration.json"))
                app.apply_twotheta_calibration_var.set(True)
                app._sync_twotheta_calibration_setting()
                saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))
                self.assertTrue(saved_settings["apply_twotheta_calibration"])
                self.assertTrue(saved_settings["twotheta_calibration_file"].endswith("calibration.json"))

            calibration_window = app.open_calibration_window()
            self.assertTrue(calibration_window.winfo_exists())
            self.assertIn("calibration.input_paths", app.variables)
            self.assertEqual(app.open_calibration_window(), calibration_window)

            app.variables["calibration.material"].set("LaB6 (cubic, Pm-3m)")
            app._update_calibration_mode()
            self.assertEqual(app.variables["calibration.a"].get(), str(tth_cal.LAB6_A))
            self.assertEqual(app.widgets["calibration.b"][0].cget("state"), "disabled")
            app.variables["calibration.material"].set("custom")
            app.variables["calibration.lattice_type"].set("orthorhombic")
            app._update_calibration_mode()
            self.assertNotEqual(app.widgets["calibration.b"][0].cget("state"), "disabled")
            self.assertNotEqual(app.widgets["calibration.c"][0].cget("state"), "disabled")

            app.variables["calibration.source_type"].set("txt")
            with mock.patch.object(gui_app.filedialog, "askopenfilenames", return_value=("a.txt", "b.txt")):
                app._browse(app.variables["calibration.input_paths"], "calibration_input", "calibration.input_paths")
            self.assertEqual(app.variables["calibration.input_paths"].get(), "a.txt; b.txt")

            app.variables["calibration.source_type"].set("csv")
            with mock.patch.object(gui_app.filedialog, "askopenfilename", return_value="single.csv"):
                app._browse(app.variables["calibration.input_paths"], "calibration_input", "calibration.input_paths")
            self.assertEqual(app.variables["calibration.input_paths"].get(), "single.csv")

            with mock.patch.object(gui_app.tth_cal, "build_twotheta_calibration", return_value={"path": "calibration.json", "combined_txt": "combined.txt", "combined_csv": "combined.csv", "profile_plot": "profile.png", "fit_plot": "fit.png"}) as generated_calibration:
                captured = {}

                def immediate_run(title, func, on_success=None, run_on_main=False):
                    captured["result"] = func()
                    if on_success:
                        on_success(captured["result"])

                app._run_task = immediate_run
                app.variables["calibration.source_type"].set("txt")
                app.variables["calibration.input_paths"].set(str(Path(tmp) / "a.txt") + ";" + str(Path(tmp) / "b.txt"))
                app.variables["calibration.output_dir"].set(str(Path(tmp) / "calibration"))
                app.variables["calibration.wavelength"].set("1.0")
                app.variables["calibration.discard_outliers"].set(True)
                app.variables["calibration.show_plots"].set(False)
                app.run_twotheta_calibration()
                self.assertEqual(len(generated_calibration.call_args.args[0]), 2)
                self.assertTrue(generated_calibration.call_args.kwargs["discard_outliers"])
                self.assertEqual(app.twotheta_calibration_file.get(), "calibration.json")
                self.assertTrue(app.apply_twotheta_calibration_var.get())

            with mock.patch.object(gui_app.peak_analysis, "run_peak_series", return_value={"csv_path": "peaks.csv", "plot_path": "peaks.png"}) as peak_run:
                app.variables["peak.data_dir"].set(str(Path(tmp)))
                app.variables["peak.scans"].set("10-11")
                app.variables["peak.scan_type"].set("delta")
                app.variables["peak.frame_index"].set("0")
                app.variables["peak.center"].set("40.0")
                app.variables["peak.window"].set("0.7")
                app.variables["peak.fit_mode"].set("compare")
                app.variables["peak.show_final"].set(True)
                app.variables["peak.diagnostic_all_fits"].set(True)
                app._update_peak_mode()
                self.assertFalse(app.widgets["peak.frame_index"][1].grid_info())
                app.run_peak_analysis()
                self.assertEqual(peak_run.call_args.kwargs["scans"], [10, 11])
                self.assertIsNone(peak_run.call_args.kwargs["frame_index"])
                self.assertEqual(peak_run.call_args.kwargs["peak_center"], 40.0)
                self.assertEqual(peak_run.call_args.kwargs["fit_mode"], "compare")
                self.assertTrue(peak_run.call_args.kwargs["show"])
                self.assertTrue(peak_run.call_args.kwargs["diagnostic_all_fits"])
                self.assertTrue(callable(peak_run.call_args.kwargs["progress_callback"]))

                app.variables["peak.scan_type"].set("chi")
                app._update_peak_mode()
                self.assertTrue(app.widgets["peak.frame_index"][1].grid_info())
                self.assertNotEqual(app.widgets["peak.frame_index"][1].cget("state"), "disabled")
                app.run_peak_analysis()
                self.assertEqual(peak_run.call_args.kwargs["frame_index"], 0)

                app.variables["peak.scan_type"].set("omega")
                app.variables["peak.scans"].set(":2")
                app._update_peak_mode()
                self.assertFalse(app.widgets["peak.frame_index"][1].grid_info())
                with mock.patch.object(gui_app.peak_analysis, "discover_scan_numbers", return_value=[20, 21, 22, 23, 24]):
                    app.run_peak_analysis()
                self.assertIsNone(peak_run.call_args.kwargs["frame_index"])
                self.assertEqual(peak_run.call_args.kwargs["scans"], [20, 22, 24])

            with mock.patch.object(gui_app.peak_analysis, "plot_peak_series_from_csv", return_value={"csv_path": "peaks.csv", "plot_path": "replot.png"}) as peak_replot:
                app.variables["peak.results_csv"].set(str(Path(tmp) / "peak_series.csv"))
                app.variables["peak.x"].set("temperature")
                app.variables["peak.show_final"].set(False)
                app.replot_peak_analysis()
                self.assertEqual(peak_replot.call_args.args[0], str(Path(tmp) / "peak_series.csv"))
                self.assertEqual(peak_replot.call_args.kwargs["x"], "temperature")

            help_text = app._help_text()
            self.assertIn("Delta scans are homogenised automatically", help_text)
            self.assertIn("comparison panel appears only", help_text)

            app.variables["sin2psi.action"].set("process")
            app._update_sin2psi_mode()
            self.assertTrue(app.sections["sin2psi.peak_options"].grid_info())
            self.assertTrue(app.sections["sin2psi.exclusions"].grid_info())
            self.assertTrue(app.sections["sin2psi.stress"].grid_info())
            self.assertFalse(app.sections["sin2psi.calibration"].grid_info())

            app.variables["sin2psi.action"].set("refit")
            app._update_sin2psi_mode()
            self.assertFalse(app.sections["sin2psi.peak_options"].grid_info())
            self.assertTrue(app.sections["sin2psi.exclusions"].grid_info())
            self.assertTrue(app.sections["sin2psi.stress"].grid_info())
            self.assertFalse(app.sections["sin2psi.calibration"].grid_info())

            app.variables["sin2psi.action"].set("correction")
            app.variables["sin2psi.correction_method"].set("polynomial")
            app._update_sin2psi_mode()
            self.assertFalse(app.sections["sin2psi.peak_options"].grid_info())
            self.assertTrue(app.sections["sin2psi.exclusions"].grid_info())
            self.assertFalse(app.sections["sin2psi.stress"].grid_info())
            self.assertTrue(app.sections["sin2psi.calibration"].grid_info())
            self.assertTrue(app.widgets["sin2psi.correction_degree"][0].grid_info())

            app.variables["sin2psi.correction_method"].set("gaussian_process")
            app._update_sin2psi_mode()
            self.assertFalse(app.widgets["sin2psi.correction_degree"][0].grid_info())

            with mock.patch.object(gui_app.proc, "generate_sin2psi_correction", return_value={"path": "correction.json", "plot_path": "correction.png"}) as generated:
                captured = {}

                def immediate_run(title, func, on_success=None, run_on_main=False):
                    captured["result"] = func()
                    if on_success:
                        on_success(captured["result"])

                app._run_task = immediate_run
                app.variables["sin2psi.action"].set("correction")
                app.variables["sin2psi.correction_method"].set("polynomial")
                app.variables["sin2psi.correction_degree"].set("3")
                app.variables["sin2psi.data_dir"].set(str(self.repo_root))
                app.variables["sin2psi.scans"].set("101")
                app.variables["sin2psi.reference_folder"].set(str(self.repo_root))
                app.variables["sin2psi.reference_scan"].set("101")
                app.run_sin2psi_action()
                self.assertEqual(generated.call_args.kwargs["degree"], 3)

            app.variables["plot.type"].set("stress")
            app._update_plot_mode()
            self.assertEqual(app.variables["plot.type"].get(), "stress")
            self.assertNotEqual(app.widgets["plot.summary_csv"][1].cget("state"), "disabled")
            app.variables["plot.type"].set("spectra")
            app._update_plot_mode()
            self.assertEqual(app.widgets["plot.summary_csv"][1].cget("state"), "disabled")
            self.assertFalse(app.variables["plot.save_final"].get())
            self.assertTrue(app.widgets["plot.offset"][0].grid_info())
            self.assertTrue(app.widgets["plot.labels"][0].grid_info())
            self.assertFalse(app.widgets["plot.x"][0].grid_info())
            self.assertFalse(app.widgets["plot.predicted_source"][0].grid_info())
            self.assertEqual(app._selected_plot_scan_types(), ["chi"])
            app.variables["plot.scan_type.delta"].set(True)
            self.assertEqual(app._selected_plot_scan_types(), ["chi", "delta"])
            self.assertEqual(app._selected_spectra_labels(), [])
            app.variables["plot.label.type"].set(True)
            app.variables["plot.label.temp"].set(True)
            self.assertEqual(app._selected_spectra_labels(), ["type", "temp"])
            app.variables["plot.show_predicted_peaks"].set(True)
            app.variables["plot.predicted_source"].set("list")
            app._update_plot_mode()
            self.assertTrue(app.widgets["plot.predicted_source"][0].grid_info())
            self.assertTrue(app.widgets["plot.predicted_twotheta"][0].grid_info())
            self.assertFalse(app.widgets["plot.predicted_lattice_type"][0].grid_info())
            app.variables["plot.predicted_source"].set("lattice")
            app.variables["plot.predicted_lattice_type"].set("fcc")
            app._update_plot_mode()
            self.assertFalse(app.widgets["plot.predicted_twotheta"][0].grid_info())
            self.assertTrue(app.widgets["plot.predicted_lattice_type"][0].grid_info())
            self.assertNotEqual(app.widgets["plot.predicted_a"][0].cget("state"), "disabled")
            self.assertEqual(app.widgets["plot.predicted_b"][0].cget("state"), "disabled")
            self.assertEqual(app.widgets["plot.predicted_c"][0].cget("state"), "disabled")

            app.variables["plot.predicted_lattice_type"].set("tetragonal")
            app._update_plot_mode()
            self.assertEqual(app.widgets["plot.predicted_b"][0].cget("state"), "disabled")
            self.assertNotEqual(app.widgets["plot.predicted_c"][0].cget("state"), "disabled")

            app.variables["plot.predicted_lattice_type"].set("orthorhombic")
            app._update_plot_mode()
            self.assertNotEqual(app.widgets["plot.predicted_b"][0].cget("state"), "disabled")
            self.assertNotEqual(app.widgets["plot.predicted_c"][0].cget("state"), "disabled")

            with tempfile.TemporaryDirectory() as tmp:
                app.variables["plot.data_dir"].set(tmp)
                fig = proc.plt.figure()
                try:
                    proc.plt.plot([0, 1], [0, 1])
                    saved_plot = app.save_current_plot()
                    self.assertTrue(Path(saved_plot).exists())
                    self.assertEqual(Path(saved_plot).parent.name, "saved_plots")
                    self.assertEqual(Path(saved_plot).parent.parent, Path(tmp))
                finally:
                    proc.plt.close(fig)

            with tempfile.TemporaryDirectory() as tmp:
                cache_root = Path(tmp) / "cache"
                app.set_cache_root(cache_root)
                for scan in (101, 102):
                    Path(tmp, f"I_vs_2th_{scan}_chi_0.txt").write_text(
                        "\n".join(["# Scan Type: ascan_chi", "# Chi: 0", "30 1", "31 2"]),
                        encoding="utf-8",
                    )
                Path(tmp, "I_vs_2th_999_chi_0.txt").write_text("30 1\n31 2", encoding="utf-8")
                cache_output = StringIO()
                with redirect_stdout(cache_output):
                    cache_info = app._ensure_scan_txt_cache(tmp, [101])
                self.assertEqual(cache_info["copied"], 1)
                self.assertEqual(cache_info["reused"], 0)
                self.assertIn("Local cache: checking source TXT files for 1 scan(s)", cache_output.getvalue())
                self.assertIn("Local cache: started - copying 1 file(s)", cache_output.getvalue())
                self.assertIn("Local cache: completed - 1 copied, 0 reused", cache_output.getvalue())
                self.assertTrue((Path(cache_info["cache_dir"]) / "I_vs_2th_101_chi_0.txt").exists())
                self.assertFalse((Path(cache_info["cache_dir"]) / "I_vs_2th_999_chi_0.txt").exists())
                repeated_output = StringIO()
                with redirect_stdout(repeated_output):
                    repeated_cache_info = app._ensure_scan_txt_cache(tmp, [101])
                self.assertEqual(repeated_cache_info["copied"], 0)
                self.assertEqual(repeated_cache_info["reused"], 1)
                self.assertIn("Local cache: started - copying 0 file(s), reusing 1 existing file(s)", repeated_output.getvalue())
                self.assertIn("Local cache: completed - 0 copied, 1 reused", repeated_output.getvalue())
                captured = {}

                class DummySpectrum:
                    def __init__(self, directory):
                        captured["directory"] = directory

                    def plot_Ivs2theta(self, scanNos, **kwargs):
                        captured["scanNos"] = scanNos
                        captured["kwargs"] = kwargs
                        return {"ok": True}

                def immediate_run(title, func, on_success=None, run_on_main=False):
                    captured["result"] = func()
                    if on_success:
                        on_success(captured["result"])

                app._run_task = immediate_run
                app.variables["plot.type"].set("spectra")
                app.variables["plot.data_dir"].set(tmp)
                app.variables["plot.scans"].set("")
                app.use_local_cache_var.set(True)
                app.variables["plot.offset"].set("0.25")
                app.variables["plot.show_final"].set(False)
                app.variables["plot.save_final"].set(True)
                app.variables["plot.label.type"].set(True)
                app.variables["plot.label.time"].set(True)
                with mock.patch.dict("sys.modules", {"XRD_spectra_anal": mock.Mock(Spectrum=DummySpectrum)}):
                    app.run_plotting()
                self.assertEqual(captured["scanNos"], [101, 102, 999])
                self.assertNotEqual(Path(captured["directory"]), Path(tmp))
                self.assertTrue(str(captured["directory"]).startswith(str(cache_root)))
                self.assertEqual(captured["kwargs"]["offset"], 0.25)
                self.assertEqual(captured["kwargs"]["label"], ["type", "temp", "time"])
                self.assertFalse(captured["kwargs"]["show_plot"])
                self.assertTrue(captured["kwargs"]["save_plot"])
                self.assertEqual(Path(captured["kwargs"]["save_directory"]).name, "saved_plots")

                process_captured = {}
                with mock.patch.object(gui_app.proc, "process_scan", return_value={"csv_path": "fits.csv", "scan_dir": "scan_101"}) as processed:
                    common = {
                        "data_dir": tmp,
                        "use_cache": True,
                        "exclude_frames": [],
                        "exclude_chi_ranges": [],
                        "exclude_sin2psi_ranges": [],
                        "auto_exclude": False,
                        "correction_json": None,
                    }
                    result = app._process_sin2psi_scans(common, [101])
                    process_captured["files"] = processed.call_args.kwargs["files"]
                    process_captured["data_dir"] = processed.call_args.kwargs["data_dir"]
                self.assertEqual(result[0]["csv_path"], "fits.csv")
                self.assertEqual(process_captured["data_dir"], tmp)
                self.assertTrue(process_captured["files"])
                self.assertTrue(str(process_captured["files"][0]).startswith(str(cache_root)))

            app.log("Smoke Test")
            self.assertIn("Smoke Test", app.status_var.get())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
