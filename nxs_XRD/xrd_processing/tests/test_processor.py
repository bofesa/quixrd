from pathlib import Path
import unittest

from nxs_XRD.nxs_processing import sin2psi_processor as proc


class ProcessorSmokeTest(unittest.TestCase):
    def test_parse_and_fit_smoke(self):
        sample = Path(__file__).resolve().parents[3] / "export" / "I_vs_2th_440_chi_0.txt"
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


if __name__ == "__main__":
    unittest.main()
