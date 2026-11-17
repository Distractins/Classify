"""以 -S 禁用 site-packages，验证备用入口不需要第三方依赖。"""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import path_config as config
import wsi_subset_stdlib as subset


class StdlibSubsetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT, prefix="stdlib_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.p = config.paths(self.root)
        self.source = self.root / "output/stat/legacy_patch_map.csv"
        self.source.parent.mkdir(parents=True)
        self.rows = []
        for cls in config.CLASSES:
            folder = self.p["raw"] / cls
            folder.mkdir(parents=True)
            for slide in range(20):
                for patch in range(2):
                    name = f"{slide:05d}_{patch}_100_{cls}.png"
                    (folder / name).write_bytes(b"unchanged-original")
                    self.rows.append((name, f"{slide:05d}", cls))
        self.write_source()

    def write_source(self):
        with self.source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["img_name", "patient_id", "cls"])
            writer.writerows(self.rows)

    def run_script(self, *args):
        return subprocess.run([sys.executable, "-S", str(ROOT / "wsi_subset_stdlib.py"),
                               "--root", str(self.root), "--select-slide-num", "10", *args],
                              capture_output=True)

    def test_no_dependencies_complete_copy_and_disjoint_split(self):
        preview = self.run_script("--dry-run")
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertFalse(self.p["paper"].exists())
        self.assertFalse((self.p["stat"] / "mapping.csv").exists())
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.p["paper"] / "split_mapping.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        splits = {}
        for row in rows:
            splits.setdefault(row["slideID"], set()).add(row["split"])
            original = self.p["raw"] / row["cls"] / row["img_name"]
            copied = self.p["paper"] / row["split"] / row["cls"] / row["img_name"]
            self.assertEqual(original.read_bytes(), copied.read_bytes())
        self.assertEqual(len(splits), 10)
        self.assertTrue(all(len(values) == 1 for values in splits.values()))
        self.assertEqual(len(rows), 160)
        self.assertEqual(len(list(self.p["raw"].rglob("*.png"))), 320)
        self.assertNotEqual(self.run_script().returncode, 0)

    def test_wrong_prefix_rejected(self):
        name, slide, cls = self.rows[0]
        self.rows[0] = (name, "99999", cls)
        self.write_source()
        self.assertNotEqual(self.run_script("--dry-run").returncode, 0)
        self.assertFalse(self.p["paper"].exists())

    def test_resume_skips_csv_and_completed_classes(self):
        self.assertEqual(self.run_script("--dry-run").returncode, 0)
        cache = self.root / "output/wsi_subset_checkpoint"
        with patch.object(subset, "read_legacy", side_effect=AssertionError("CSV must be cached")):
            rows, expected, state, path = subset.load_checkpoint(self.source, self.p["raw"], cache)
        with patch.object(subset.os, "walk", side_effect=AssertionError("classes must be cached")):
            subset.verify_files(self.p["raw"], expected, state, path)
        (self.p["raw"] / "ADI/00000_99_100_ADI.png").write_bytes(b"new")
        self.assertNotEqual(self.run_script("--dry-run", "--recheck-files").returncode, 0)

    def test_copy_interruption_resumes_and_repairs_bad_copy(self):
        args = ["--root", str(self.root), "--select-slide-num", "10"]
        original_copy = subset.shutil.copy2
        calls = []
        def interrupt(source, target):
            calls.append(target)
            if len(calls) == 2:
                raise RuntimeError("simulated interruption")
            return original_copy(source, target)
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(subset.shutil, "copy2", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    subset.main(args)
        self.assertFalse(self.p["paper"].exists())
        staging = self.p["paper"].with_name(self.p["paper"].name + ".building")
        copied = next(staging.rglob("*.png"))
        copied.write_bytes(b"bad")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        for image in self.p["paper"].rglob("*.png"):
            self.assertEqual(image.read_bytes(), b"unchanged-original")

    def test_new_png_rejected(self):
        (self.p["raw"] / "ADI/00000_99_100_ADI.png").write_bytes(b"new")
        self.assertNotEqual(self.run_script("--dry-run").returncode, 0)
        self.assertFalse(self.p["paper"].exists())


if __name__ == "__main__":
    unittest.main()
