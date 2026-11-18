from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repair_patch_image import repair, decode
from utils.path_config import paths


class RepairTests(unittest.TestCase):
    def test_same_size_corrupt_copy_repaired_and_backed_up(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            source = paths(root)["raw"] / "TUM/001_1_2_TUM.png"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "red").save(source)
            dataset = root / "dataset"
            target = dataset / "val/TUM" / source.name
            target.parent.mkdir(parents=True)
            target.write_bytes(b"x" * source.stat().st_size)
            self.assertFalse(repair(root, dataset, "val/TUM/" + source.name))
            self.assertTrue(repair(root, dataset, "val/TUM/" + source.name, True))
            decode(target)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertEqual(len(list((root / "outputs/image_repairs").rglob("*.bad"))), 1)

    def test_bad_source_is_not_replaced(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            target = root / "dataset/val/TUM/001_1_2_TUM.png"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError, "原始图片"):
                repair(root, root / "dataset", "val/TUM/" + target.name, True)
            self.assertEqual(target.read_bytes(), b"bad")


if __name__ == "__main__":
    unittest.main()
