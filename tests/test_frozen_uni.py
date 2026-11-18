import sys
from pathlib import Path
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from network.frozen_uni import FrozenUNIClassifier


class TinyBackbone(nn.Module):
    num_features = 16
    num_prefix_tokens = 1

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 16)
        self.dropout = nn.Dropout(.8)

    def forward_features(self, images):
        return self.dropout(self.proj(images))


class FrozenTests(unittest.TestCase):
    def test_backbone_frozen_and_heads_update(self):
        for mode in ("linear", "mlp", "aether"):
            with self.subTest(mode=mode):
                model = FrozenUNIClassifier(TinyBackbone(), mode, hidden_dim=8)
                model.train()
                self.assertFalse(model.backbone.training)
                original = {k: v.clone() for k, v in model.backbone.state_dict().items()}
                head_before = model.head[-1].weight.clone() if mode == "mlp" else model.head.weight.clone()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
                output = model(torch.randn(4, 5, 3))
                logits, reg = output if isinstance(output, tuple) else (output, 0)
                self.assertEqual(tuple(logits.shape), (4, 8))
                loss = nn.functional.cross_entropy(logits, torch.arange(4)) + reg
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
                if mode == "aether":
                    self.assertIsNotNone(model.enhancer.signal_sep.proj_texture[1].weight.grad)
                optimizer.step()
                for key, value in model.backbone.state_dict().items():
                    self.assertTrue(torch.equal(value, original[key]))
                head_after = model.head[-1].weight if mode == "mlp" else model.head.weight
                self.assertFalse(torch.equal(head_before, head_after))
                model.eval()
                model.train()
                self.assertFalse(model.backbone.training)

    def test_linear_uses_spatial_mean_without_cls(self):
        model = FrozenUNIClassifier(TinyBackbone(), "linear").eval()
        x = torch.randn(2, 5, 3)
        with torch.no_grad():
            expected = model.head(model.backbone.forward_features(x)[:, 1:].mean(1))
        torch.testing.assert_close(model(x), expected)


if __name__ == "__main__":
    unittest.main()
