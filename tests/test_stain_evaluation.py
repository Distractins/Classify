import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from src.stage1_train_8cls import evaluate_stain_robustness as evaluation


class EvaluationTests(unittest.TestCase):
    def test_three_conditions_use_same_samples_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            rng = np.random.default_rng(3)
            od = rng.uniform(0.2, 2, (24, 24, 2)) @ np.array([[.65, .70, .29], [.07, .99, .11]])
            image = Image.fromarray(np.uint8(255 * np.exp(-od)))
            for index, split in enumerate(("train", "val", "test")):
                for cls in evaluation.baseline.CLASSES:
                    folder = root / split / cls
                    folder.mkdir(parents=True)
                    image.save(folder / f"{index + 1}_10_20_{cls}.png")
            _, manifest = evaluation.baseline.load_data(root)
            model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
            checkpoint = Path(directory) / "best.pth"
            torch.save({"classes": list(evaluation.baseline.CLASSES), "model_name": "efficientnet_b0",
                        "fingerprint": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
                        "model": model.state_dict(), "epoch": 7}, checkpoint)
            output = Path(directory) / "reports"
            args = ["evaluate", "--root", str(root), "--checkpoint", str(checkpoint),
                    "--output", str(output), "--batch-size", "8"]
            with patch.object(sys, "argv", args), patch.object(evaluation.baseline, "build_model", return_value=model):
                evaluation.main()
            with (output / "comparison.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["condition"] for row in rows], ["original", "macenko", "hed"])
            self.assertTrue(all(int(row["sample_count"]) == 8 for row in rows))
            self.assertEqual(float(rows[0]["macro_f1_drop_pp"]), 0)
            names = []
            for method in ("original", "macenko", "hed"):
                with (output / method / "test_predictions.csv").open(encoding="utf-8-sig") as handle:
                    names.append([row["img_name"] for row in csv.DictReader(handle)])
            self.assertEqual(names[0], names[1])
            self.assertEqual(names[1], names[2])
            self.assertTrue((output / "comparison.png").is_file())
            # Independent runs must not evaluate other conditions or invent a
            # zero-drop reference for a perturbed-only run.
            for method in ("original", "macenko", "hed"):
                single = Path(directory) / method
                single_args = args.copy()
                single_args[single_args.index("--output") + 1] = str(single)
                single_args += ["--method", method]
                with patch.object(sys, "argv", single_args), patch.object(evaluation.baseline, "build_model", return_value=model):
                    evaluation.main()
                self.assertTrue((single / method / "test_metrics.json").is_file())
                self.assertFalse((single / "comparison.csv").exists())
                self.assertEqual([p.name for p in single.iterdir() if p.is_dir()], [method])


if __name__ == "__main__":
    unittest.main()
