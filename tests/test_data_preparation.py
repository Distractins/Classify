"""验证整患者复制、硬限额、临床关联和跨机器路径配置。"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data_explore as explore
import patient_stratified_sampling as sampling
from utils import path_config as config
import pandas as pd
from PIL import Image


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="prepare_test_", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.p = config.paths(self.root)
        for cls in config.CLASSES:
            (self.p["raw"] / cls).mkdir(parents=True)
        self.ids = ["00101", "00056", "00239", "00004", "00005", "00006"]
        for pid in self.ids:
            for cls in config.CLASSES:
                Image.new("RGB", (8, 8), "red").save(self.p["raw"] / cls / f"patient_{pid}_001.png")
        pd.DataFrame({"PatientID": self.ids + ["99999"],
                      "MSI": ["MSI-H", "MSS", "MSI-H", "MSS", "MSI-H", "MSS", "MSI-L"],
                      "Age": [50] * 7, "Sex": ["Male"] * 7}).to_excel(self.p["clinical"], index=False)

    def explore(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            explore.main(["--root", str(self.root), *args])

    def sample(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            sampling.main(["--root", str(self.root), *args])

    def test_explore_and_whole_patient_copy(self):
        before = {p: p.read_bytes() for p in self.p["raw"].rglob("*") if p.is_file()}
        self.explore("--check-images", "pil")
        stat = pd.read_csv(self.p["stat"] / "patient_patch_stats.csv", dtype={"patient_id": str})
        self.assertEqual(set(stat.patient_id), set(self.ids + ["99999"]))
        self.assertEqual(stat.loc[stat.patient_id == "99999", "total_patches"].iloc[0], 0)
        for name in ("class_patch_counts.png", "patient_patch_histogram.png", "patient_class_heatmap.png"):
            self.assertGreater((self.p["stat"] / name).stat().st_size, 1000)
        self.sample("--max-total-patches", "24", "--max-per-class", "3", "--dry-run")
        self.assertFalse(self.p["subset"].exists())
        planned = pd.read_csv(self.p["sampling"] / "planned_patch_patient_map.csv", dtype=str)
        self.sample("--max-total-patches", "24", "--max-per-class", "3")
        actual = explore.scan_patch_map(self.p["subset"])
        original = explore.scan_patch_map(self.p["raw"])
        selected = set(actual.patient_id)
        expected = original[original.patient_id.isin(selected)]
        key = lambda frame: set(frame.itertuples(index=False, name=None))
        self.assertEqual(key(actual), key(expected))
        self.assertEqual(key(actual), key(planned))
        self.assertEqual(len(actual), 24)
        self.assertTrue(actual.cls.value_counts().le(3).all())
        report = json.loads((self.p["subset"] / "sampling_report.json").read_text(encoding="utf-8"))
        self.assertEqual(abs(report["msi_patient_counts"]["MSI-H"] - report["msi_patient_counts"]["MSS"]), 1)
        subset_clinical = pd.read_excel(self.p["subset"] / config.CLINICAL_NAME, dtype=str)
        self.assertEqual(set(subset_clinical.PatientID), selected)
        self.assertEqual(before, {p: p.read_bytes() for p in self.p["raw"].rglob("*") if p.is_file()})
        with self.assertRaises(FileExistsError):
            self.sample("--max-total-patches", "24", "--max-per-class", "3")

    def test_partial_or_stale_mapping_is_rejected(self):
        self.explore()
        path = self.p["stat"] / "patch_patient_map.csv"
        table = pd.read_csv(path, dtype=str)
        table.iloc[1:].to_csv(path, index=False)
        with self.assertRaises(ValueError):
            self.sample("--dry-run")
        self.assertFalse(self.p["subset"].exists())

    def test_patient_over_cap_is_never_truncated(self):
        self.explore()
        with self.assertRaises(ValueError):
            self.sample("--max-total-patches", "7", "--dry-run")
        report = json.loads((self.p["sampling"] / "sampling_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["selected_patients"], 0)
        self.assertEqual(set(report["individually_over_cap"]), set(self.ids))

    def test_asymmetric_class_cap_and_determinism(self):
        self.explore()
        self.sample("--max-total-patches", "24", "--max-per-class", "3", "--class-cap", "LYM=2", "--dry-run")
        path = self.p["sampling"] / "planned_patch_patient_map.csv"
        first = pd.read_csv(path, dtype=str)
        self.assertEqual(len(first), 16)
        self.assertEqual(first.cls.eq("LYM").sum(), 2)
        mapping = self.p["stat"] / "patch_patient_map.csv"
        pd.read_csv(mapping, dtype=str).iloc[::-1].to_csv(mapping, index=False)
        self.sample("--max-total-patches", "24", "--max-per-class", "3", "--class-cap", "LYM=2", "--dry-run")
        pd.testing.assert_frame_equal(pd.read_csv(path, dtype=str), first)

    def test_missing_clinical_id_not_silently_dropped(self):
        clinical = pd.read_excel(self.p["clinical"], dtype=str)
        clinical.loc[clinical.PatientID == "00101", "PatientID"] = "101"
        clinical.to_excel(self.p["clinical"], index=False)
        self.explore()
        with self.assertRaises(ValueError):
            self.sample("--dry-run")

    def test_opencv_and_invalid_images(self):
        self.explore("--check-images", "opencv")
        (self.p["raw"] / "ADI" / "patient_00101_002.png").write_bytes(b"broken-image")
        with self.assertRaises(ValueError):
            self.explore("--check-images", "pil")
        bad = pd.read_csv(self.p["stat"] / "invalid_images.csv", dtype=str)
        self.assertEqual(bad.patient_id.tolist(), ["00101"])

    def test_patient_column_without_explicit_msi(self):
        import warnings
        clinical = pd.read_excel(self.p["clinical"], dtype=str).rename(columns={"PatientID": "Patient"})
        clinical = clinical.drop(columns="MSI")
        clinical["immunohistochemistry"] = "uninterpreted"
        clinical.to_excel(self.p["clinical"], index=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.explore()
            self.sample("--max-total-patches", "16", "--max-per-class", "2")
        self.assertTrue(caught)
        stats = pd.read_csv(self.p["stat"] / "patient_patch_stats.csv", dtype=str)
        self.assertEqual(set(stats.msi_status), {"UNKNOWN"})
        subset_clinical = pd.read_excel(self.p["subset"] / config.CLINICAL_NAME, dtype=str)
        self.assertEqual(len(subset_clinical), 2)
        self.assertTrue(set(subset_clinical.Patient).issubset(self.ids))


if __name__ == "__main__":
    unittest.main()
