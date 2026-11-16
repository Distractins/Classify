"""验证 WSI 完整复制、互斥划分、可复现性与过期映射拦截。"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import dataset_stat as stat
import dataset_sampling as sampling
from utils import path_config as config


class SlidePreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT, prefix="slide_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.p = config.paths(self.root)
        for cls in config.CLASSES:
            folder = self.p["raw"] / cls
            folder.mkdir(parents=True)
            for slide in range(20):
                for index in range(2):
                    (folder / f"{slide:05d}_{index}_100_{cls}.png").write_bytes(b"source-patch")
            (folder / "ignored.jpg").write_bytes(b"not-png")
        # 探查与复制都不解码 PNG，也不需要有效 Excel。
        (self.p["raw"] / "HMU-CRC-Clinical.xlsx").write_bytes(b"must-not-read")

    def call(self, function, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            function(["--root", str(self.root), *args])

    def mapping(self):
        mapping = stat.scan_dataset(self.p["raw"])
        stat.save_csv(mapping, self.p["stat"] / "mapping.csv")
        return stat.read_mapping(self.p["stat"] / "mapping.csv")

    def test_whole_slide_copy_and_disjoint_splits(self):
        source = self.mapping()
        self.call(sampling.main, "--mode", "paper_mode", "--select-slide-num", "10")
        plan = pd.read_csv(self.p["paper"] / "split_mapping.csv", dtype=str)
        self.assertEqual(plan["slideID"].nunique(), 10)
        self.assertEqual(plan.groupby("slideID")["split"].nunique().max(), 1)
        self.assertEqual(plan.groupby("split")["slideID"].nunique().to_dict(),
                         {"train": 7, "val": 2, "test": 1})
        selected = source[source.slideID.isin(plan.slideID)]
        self.assertEqual(len(selected), len(plan))
        self.assertEqual(len(plan), 10 * 8 * 2)
        for row in plan.itertuples(index=False):
            src = self.p["raw"] / row.cls / row.img_name
            dst = self.p["paper"] / row.split / row.cls / row.img_name
            self.assertEqual(src.read_bytes(), dst.read_bytes())
        self.assertFalse((self.p["paper"] / "HMU-CRC-Clinical.xlsx").exists())
        self.assertEqual(len(list(self.p["raw"].rglob("*.png"))), 320)
        with self.assertRaises(FileExistsError):
            self.call(sampling.main, "--mode", "paper_mode", "--select-slide-num", "10")

    def test_deterministic_plan_and_bounds(self):
        source = self.mapping()
        first = sampling.build_plan(source, "paper_mode", 1, 10)
        second = sampling.build_plan(source.iloc[::-1], "paper_mode", 1, 10)
        pd.testing.assert_frame_equal(first.sort_values(["cls", "img_name"]).reset_index(drop=True),
                                      second.sort_values(["cls", "img_name"]).reset_index(drop=True))
        for count in (1, 6, 21):
            with self.assertRaises(ValueError):
                sampling.build_plan(source, "paper_mode", 1, count)

    def test_fast_mode_exact_counts_and_dry_run(self):
        self.mapping()
        self.call(sampling.main, "--mode", "fast_mode", "--sample-per-class", "3", "--dry-run")
        self.assertFalse(self.p["fast"].exists())
        self.call(sampling.main, "--mode", "fast_mode", "--sample-per-class", "3")
        data = stat.read_mapping(self.p["fast"] / "mapping.csv")
        self.assertTrue(data.groupby("cls").size().eq(3).all())
        self.assertFalse((self.p["fast"] / "train").exists())

    def test_stale_map_rejected_before_copy(self):
        self.mapping()
        (self.p["raw"] / "ADI" / "00000_99_100_ADI.png").write_bytes(b"new")
        with self.assertRaisesRegex(ValueError, "不一致"):
            self.call(sampling.main, "--mode", "paper_mode", "--select-slide-num", "10")
        self.assertFalse(self.p["paper"].exists())

    def test_per_class_cache_and_summary(self):
        self.call(stat.main, "--scan-class", "ADI")
        with self.assertRaisesRegex(ValueError, "尚未扫描"):
            self.call(stat.main, "--summarize")
        for cls in config.CLASSES[1:]:
            self.call(stat.main, "--scan-class", cls)
        with patch.object(stat, "scan_class", side_effect=AssertionError("no rescan")):
            self.call(stat.main, "--summarize")
        self.assertEqual(len(stat.read_mapping(self.p["stat"] / "mapping.csv")), 320)
        self.assertTrue((self.p["stat"] / "class_patch_counts.png").is_file())

    def test_import_legacy_without_rescanning(self):
        source = self.mapping()
        legacy = self.root / "output_legacy" / "stat" / "patch_patient_map.csv"
        stat.save_csv(source.rename(columns={"slideID": "patient_id"}), legacy)
        with patch.object(stat, "scan_class", side_effect=AssertionError("no rescan")):
            self.call(stat.main, "--import-legacy-map", str(legacy))
        pd.testing.assert_frame_equal(source, stat.read_mapping(self.p["stat"] / "mapping.csv"))
        self.assertTrue(legacy.is_file())

    def test_import_legacy_rejects_wrong_ids_before_cache_write(self):
        source = self.mapping().rename(columns={"slideID": "patient_id"})
        source.loc[0, "patient_id"] = "wrong"
        legacy = self.root / "legacy.csv"
        stat.save_csv(source, legacy)
        with self.assertRaises(ValueError):
            self.call(stat.main, "--import-legacy-map", str(legacy))
        self.assertFalse((self.p["stat"] / "slide_scan_cache" / "ADI.csv").exists())

    def test_filename_validation(self):
        self.assertEqual(config.parse_slide_id("00123_1_2_ADI_ADI.png", "ADI"), "00123")
        for name in ("patient_001_002.png", "1_2_3_ADI_LYM.png", "1_2_3_TUM.png"):
            with self.assertRaises(ValueError):
                config.parse_slide_id(name, "ADI")


if __name__ == "__main__":
    unittest.main()
