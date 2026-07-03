from pathlib import Path
import tempfile
import unittest

from nxs_XRD.nxs_export.XPAD_XRD_nxs_export import S140XRD, sort_extracted_by_sample


class SortedExportSupportTest(unittest.TestCase):
    def test_resolves_scan_in_sorted_tree_and_mirrors_export_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sorted_root = root / "sorted"
            export_root = root / "export"
            sample_folder = sorted_root / "23_Cu sample"
            sample_folder.mkdir(parents=True)
            scan_file = sample_folder / "scan_0440_0001.nxs"
            scan_file.write_text("fake", encoding="utf-8")

            xrd = S140XRD(
                nxs_file_directory=str(sorted_root),
                export_directory=str(export_root),
                flat_file_numbers=[39],
                daterange=None,
            )

            resolved_path, rel_folder = xrd._resolve_scan_path(440)
            export_folder = Path(xrd._export_path_for_scan(440, mirror_sorted_structure=True))

            self.assertEqual(Path(resolved_path), scan_file)
            self.assertEqual(rel_folder, "23_Cu sample")
            self.assertEqual(export_folder, export_root / "23_Cu sample")
            self.assertTrue(export_folder.exists())

    def test_flat_export_folder_remains_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nxs_root = root / "nxs"
            export_root = root / "export"
            nxs_root.mkdir()
            (nxs_root / "scan_0440_0001.nxs").write_text("fake", encoding="utf-8")

            xrd = S140XRD(
                nxs_file_directory=str(nxs_root),
                export_directory=str(export_root),
                flat_file_numbers=[39],
                daterange=None,
            )

            export_folder = Path(xrd._export_path_for_scan(440, mirror_sorted_structure=False))

            self.assertEqual(export_folder, export_root)
            self.assertTrue(export_folder.exists())

    def test_sort_extracted_by_sample_copies_matching_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sorted_root = root / "sorted"
            source_export = root / "flat_export"
            output = root / "sorted_export"
            sample_folder = sorted_root / "23_Cu sample"
            sample_folder.mkdir(parents=True)
            source_export.mkdir()

            (sample_folder / "scan_0440_0001.nxs").write_text("fake", encoding="utf-8")
            txt = source_export / "I_vs_2th_440_chi_0.txt"
            csv = source_export / "scan_440_2026-01-01T000000_ascan-chi.csv"
            png = source_export / "scan_440_I_vs_2th.png"
            unmatched = source_export / "I_vs_2th_999_chi_0.txt"
            for path in [txt, csv, png, unmatched]:
                path.write_text(path.name, encoding="utf-8")

            summary = sort_extracted_by_sample(source_export, sorted_root, output_directory=output)

            self.assertEqual(len(summary["transferred"]), 3)
            self.assertEqual(len(summary["unmatched"]), 1)
            self.assertTrue((output / "23_Cu sample" / txt.name).exists())
            self.assertTrue((output / "23_Cu sample" / csv.name).exists())
            self.assertTrue((output / "23_Cu sample" / png.name).exists())
            self.assertTrue(unmatched.exists())


if __name__ == "__main__":
    unittest.main()
