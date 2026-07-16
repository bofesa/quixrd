from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from nxs_XRD.xrd_processing import run_workflow
from nxs_XRD.xrd_processing import gui_app
from nxs_XRD.xrd_processing import peak_overlay
from nxs_XRD.xrd_processing import sin2psi_processor as proc


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
        from nxs_XRD.nxs_export import XRD_spectra_anal

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
        self.assertEqual(proc._selector_title("chi_5_0"), "chi=5.0°")
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
                peak_center = 40.0
                for _ in range(8):
                    scale = math.tan(math.radians(peak_center / 2.0)) / math.tan(
                        math.radians(correction["reference_two_theta"] / 2.0)
                    )
                    peak_center = 40.0 + correction_at_ref * scale
                sample_rows.append(
                    {
                        "frame_index": row["frame_index"],
                        "chi": row["chi"],
                        "psi_deg": row["psi_deg"],
                        "sin2psi": row["sin2psi"],
                        "peak_center": peak_center,
                        "peak_center_err": 0.01,
                    }
                )
            pd.DataFrame(sample_rows).to_csv(sample_dir / "scan_21_fits.csv", index=False)

            result = proc.refit_sin2psi_from_csv(tmp, 21, correction_json=correction_result["path"])
            summary = proc.load_processing_params(sample_dir / "sin2psi_fit_params.json")
            df = pd.read_csv(result["csv_path"])

            self.assertTrue(summary["correction_applied"])
            self.assertEqual(summary["fit_y_column"], "peak_center_corrected")
            self.assertIn("peak_center_corrected", df.columns)
            self.assertTrue((sample_dir / "scan_21_sin2psi_plot.png").exists())
            self.assertAlmostEqual(summary["slope"], 0.0, delta=2e-6)

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
            [sys.executable, "-m", "nxs_XRD.xrd_processing.cli", "--help"],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--peak-center", completed.stdout)

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
            self.assertTrue(hasattr(app, "help_menu"))
            self.assertTrue(hasattr(app, "log_text"))
            self.assertTrue(hasattr(app, "status_bar"))
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
                ["Overview", "File", "Extraction", "Plotting", "Sorting", gui_app.SIN2PSI_LABEL],
            )

            with tempfile.TemporaryDirectory() as tmp:
                params_path = Path(tmp) / "gui_params.json"
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
                for scan in (101, 102):
                    Path(tmp, f"I_vs_2th_{scan}_chi_0.txt").write_text(
                        "\n".join(["# Scan Type: ascan_chi", "# Chi: 0", "30 1", "31 2"]),
                        encoding="utf-8",
                    )
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
                app.variables["plot.offset"].set("0.25")
                app.variables["plot.show_final"].set(False)
                app.variables["plot.save_final"].set(True)
                app.variables["plot.label.type"].set(True)
                app.variables["plot.label.time"].set(True)
                with mock.patch.dict("sys.modules", {"XRD_spectra_anal": mock.Mock(Spectrum=DummySpectrum)}):
                    app.run_plotting()
                self.assertEqual(captured["scanNos"], [101, 102])
                self.assertEqual(captured["kwargs"]["offset"], 0.25)
                self.assertEqual(captured["kwargs"]["label"], ["type", "temp", "time"])
                self.assertFalse(captured["kwargs"]["show_plot"])
                self.assertTrue(captured["kwargs"]["save_plot"])
                self.assertEqual(Path(captured["kwargs"]["save_directory"]).name, "saved_plots")

            app.log("Smoke Test")
            self.assertIn("Smoke Test", app.status_var.get())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
