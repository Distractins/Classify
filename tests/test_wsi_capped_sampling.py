import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import wsi_capped_sampling as sampling


class CappedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.source, self.target = self.base / "FULL", self.base / "CAP"
        for slide, split in (("001", "train"), ("002", "val"), ("003", "test")):
            for cls in sampling.CLASSES:
                folder = self.source / split / cls
                folder.mkdir(parents=True)
                for i in range(4):
                    (folder / f"{slide}_{i}_100_{cls}.png").write_bytes(b"original")

    def plan(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return sampling.make_plan(self.source, 2)

    def test_cap_determinism_disjointness_and_resume(self):
        plan, assignments = self.plan()
        self.assertEqual(self.plan()[0], plan)
        self.assertEqual(len(plan), 48)
        self.assertEqual(assignments, {"001": "train", "002": "val", "003": "test"})
        for split in sampling.SPLITS:
            for cls in sampling.CLASSES:
                self.assertEqual(sum(row[3] == split and row[2] == cls for row in plan), 2)
        original_copy = sampling.shutil.copy2
        calls = []
        def interrupt(src, dst):
            calls.append(dst)
            if len(calls) == 3:
                raise RuntimeError("interrupt")
            return original_copy(src, dst)
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(sampling.shutil, "copy2", side_effect=interrupt):
                with self.assertRaises(RuntimeError):
                    sampling.copy_plan(self.source, self.target, plan, assignments, 2, [])
            sampling.copy_plan(self.source, self.target, plan, assignments, 2, [])
        self.assertEqual(len(list(self.target.rglob("*.png"))), 48)
        self.assertEqual(len(list(self.source.rglob("*.png"))), 96)
        self.assertTrue((self.target / "slide_splits.csv").is_file())

    def test_cross_split_wsi_rejected_before_sampling(self):
        (self.source / "test/TUM/001_99_100_TUM.png").write_bytes(b"leak")
        with self.assertRaisesRegex(ValueError, "跨集合"), contextlib.redirect_stdout(io.StringIO()):
            sampling.make_plan(self.source, 2)

    def test_exclusion_logged_in_plan(self):
        bad = "val/TUM/002_0_100_TUM.png"
        with contextlib.redirect_stdout(io.StringIO()):
            plan, _ = sampling.make_plan(self.source, 4, [bad])
        self.assertEqual(len(plan), 95)
        self.assertNotIn(bad, [f"{r[3]}/{r[2]}/{r[0]}" for r in plan])

    def test_plan_cache_saved_before_summary_and_reused(self):
        with patch.object(sampling, "print_summary", side_effect=RuntimeError("interrupted summary")):
            with self.assertRaisesRegex(RuntimeError, "interrupted summary"):
                with contextlib.redirect_stdout(io.StringIO()):
                    sampling.cached_plan(self.base, self.source, 2, [])
        with patch.object(sampling, "make_plan", side_effect=AssertionError("must not rescan")):
            with contextlib.redirect_stdout(io.StringIO()):
                plan, assignments = sampling.cached_plan(self.base, self.source, 2, [])
        self.assertEqual(len(plan), 48)
        self.assertEqual(len(assignments), 3)


if __name__ == "__main__":
    unittest.main()
