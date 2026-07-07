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
                            "fit_success": True,
                        },
                        {
                            "frame_index": 1,
                            "filename": f"I_vs_2th_{scan}_chi_1.txt",
                            "scan_type": "ascan_chi",
                            "chi": 5.0,
                            "psi_deg": 85.0,
                            "sin2psi": 0.99,
                            "temperature": temp,
                            "energy": 12.0,
                            "fwhm": 0.3 + scan / 1000.0,
                            "fwhm_err": 0.02,
                            "fit_success": True,
                        },
                    ]
                ).to_csv(scan_dir / f"scan_{scan}_fits.csv", index=False)

            by_frame = proc.collect_fwhm_summaries(tmp, frame_index=1)
            by_chi = proc.collect_fwhm_summaries(tmp, chi=5.0)
            self.assertEqual(len(by_frame), 2)
            self.assertEqual(len(by_chi), 2)
            self.assertTrue((by_chi["chi"] == 5.0).all())

            result = proc.plot_fwhm_trends(tmp, x="temperature", chi=5.0, show=False)
            self.assertTrue(Path(result["plot_path"]).exists())
            self.assertTrue(Path(result["summary_path"]).exists())
            self.assertIn("temperature", Path(result["plot_path"]).name)

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
            self.assertTrue(hasattr(app, "log_text"))
            self.assertTrue(hasattr(app, "status_bar"))
            app.placeholder("Smoke Test")
            self.assertIn("Smoke Test", app.status_var.get())
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
