"""验证新基线的 WSI 隔离与真实训练/验证步骤。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/stage1_train_8cls"))
import train_wsi_baseline as baseline


class BaselineTests(unittest.TestCase):
    def test_explicit_bad_validation_image_exclusion(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            samples = []
            for label, cls in enumerate(baseline.CLASSES):
                path = root / "val" / cls / f"001_1_2_{cls}.png"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (8, 8)).save(path)
                samples.append((str(path), label))
            bad = root / "val/TUM/001_3_4_TUM.png"
            bad.write_bytes(b"not an image")
            samples.append((str(bad), baseline.CLASSES.index("TUM")))
            data = {"val": baseline.PatchDataset(samples, None)}
            names = baseline.exclude_validation_images(data, root, ["val/TUM/001_3_4_TUM.png"])
            self.assertEqual(len(data["val"]), 8)
            self.assertTrue(bad.exists())
            self.assertEqual(names, ["val/TUM/001_3_4_TUM.png"])
            with self.assertRaisesRegex(ValueError, "decodable"):
                baseline.exclude_validation_images(data, root, ["val/ADI/001_1_2_ADI.png"])
            with self.assertRaises(ValueError):
                baseline.exclude_validation_images(data, root, ["train/TUM/001_3_4_TUM.png"])

    def test_string_parser_preserves_id_and_checks_class(self):
        self.assertEqual(baseline.config.parse_slide_id(r"ADI\00123_1_2_ADI_ADI.png", "ADI"), "00123")
        with self.assertRaises(ValueError):
            baseline.config.parse_slide_id("00123_1_2_TUM.png", "ADI")

    def test_loader_checks_disjoint_wsi(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            for index, split in enumerate(("train", "val", "test")):
                for cls in baseline.CLASSES:
                    folder = root / split / cls
                    folder.mkdir(parents=True)
                    Image.new("RGB", (16, 16)).save(folder / f"{index}_1_2_{cls}.png")
            data, manifest = baseline.load_data(root)
            self.assertEqual(len(manifest), 24)
            self.assertEqual(tuple(data["train"][0][0].shape), (3, 224, 224))
            Image.new("RGB", (16, 16)).save(root / "val/ADI/0_3_4_ADI.png")
            with self.assertRaisesRegex(ValueError, "overlap"):
                baseline.load_data(root)

    def test_train_updates_weights_and_eval_does_not(self):
        model = torch.nn.Linear(4, 8)
        loader = DataLoader(TensorDataset(torch.ones(8, 4), torch.arange(8)), batch_size=4)
        optimizer = torch.optim.Adam(model.parameters())
        before = model.weight.detach().clone()
        baseline.run_epoch(model, loader, torch.device("cpu"), optimizer)
        self.assertFalse(torch.equal(before, model.weight))
        before = model.weight.detach().clone()
        scores, matrix = baseline.run_epoch(model, loader, torch.device("cpu"))
        self.assertTrue(torch.equal(before, model.weight))
        self.assertEqual(matrix.sum().item(), 8)
        self.assertTrue(0 <= scores["macro_f1"] <= 1)

    def test_checkpoint_resume_and_report_generation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            project = Path(temporary)
            root = project / "dataset"
            for index, split in enumerate(("train", "val", "test")):
                for cls in baseline.CLASSES:
                    folder = root / split / cls
                    folder.mkdir(parents=True)
                    Image.new("RGB", (8, 8)).save(folder / f"{index}_1_2_{cls}.png")
            def tiny(*args):
                return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 8))
            args = ["--root", str(root), "--project-root", str(project), "--batch-size", "8", "--epochs", "1"]
            with patch.object(baseline, "build_model", side_effect=tiny), contextlib.redirect_stdout(io.StringIO()):
                baseline.main(args)
                output = project / "outputs/experiments/efficientnet_b0/baseline"
                first = baseline.load_checkpoint(output / "weights/last.pth")
                self.assertEqual(first["epoch"], 1)
                self.assertEqual(first["best"], first["history"][0]["val_macro_f1"])
                with self.assertRaises(FileExistsError):
                    baseline.main(args)
                resumed = args[:-1] + ["2", "--resume"]
                baseline.main(resumed)
                second = baseline.load_checkpoint(output / "weights/last.pth")
                self.assertEqual(second["epoch"], 2)
                self.assertEqual(len(second["history"]), 2)
                self.assertTrue((output / "reports/loss_curve.png").is_file())
                baseline.main(args + ["--evaluate-test"])
                self.assertTrue((output / "reports/test_confusion_matrix.png").is_file())

    def test_mid_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            project = Path(temporary)
            root = project / "dataset"
            for index, split in enumerate(("train", "val", "test")):
                for cls in baseline.CLASSES:
                    folder = root / split / cls
                    folder.mkdir(parents=True)
                    Image.new("RGB", (8, 8), color=(index * 30, 50, 100)).save(folder / f"{index}_1_2_{cls}.png")
            def tiny(*args):
                return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                           torch.nn.Dropout(.3), torch.nn.Linear(3, 8))
            args = ["--root", str(root), "--project-root", str(project), "--batch-size", "2",
                    "--epochs", "1", "--save-every-batches", "1"]
            save = baseline.atomic_save
            def interrupted(value, path):
                save(value, path)
                if value.get("train_progress") and value["train_progress"]["batch_cursor"] == 2:
                    raise RuntimeError("simulated batch interruption")
            with patch.object(baseline, "build_model", side_effect=tiny), contextlib.redirect_stdout(io.StringIO()):
                baseline.main(args + ["--run-name", "continuous"])
                with patch.object(baseline, "atomic_save", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "batch interruption"):
                        baseline.main(args + ["--run-name", "resumed"])
                baseline.main(args + ["--run-name", "resumed", "--resume"])
            base = project / "outputs/experiments/efficientnet_b0"
            first = baseline.load_checkpoint(base / "continuous/weights/last.pth")
            second = baseline.load_checkpoint(base / "resumed/weights/last.pth")
            for key in first["model"]:
                torch.testing.assert_close(first["model"][key], second["model"][key], rtol=0, atol=0)
            self.assertEqual(first["history"], second["history"])


if __name__ == "__main__":
    unittest.main()
