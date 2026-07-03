from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
