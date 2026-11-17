"""验证新基线的 WSI 隔离与真实训练/验证步骤。"""
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/stage1_train_8cls"))
import train_wsi_baseline as baseline


class BaselineTests(unittest.TestCase):
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
            with self.assertRaisesRegex(ValueError, "跨集合"):
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


if __name__ == "__main__":
    unittest.main()
