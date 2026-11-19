import unittest
import torch
from src.stage1_train_8cls.train_wsi_baseline import build_model, PROJECT


class EfficientNetAETHERTests(unittest.TestCase):
    def test_spatial_features_backprop_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        model = build_model("efficientnet_b0", PROJECT, pretrained=False, enhancer="aether")
        model.train()
        logits, regularization = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(logits.shape, (2, 8))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([2, 6])) + regularization
        loss.backward()
        for component in (model.features, model.enhancer, model.head):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters()))
        clone = build_model("efficientnet_b0", PROJECT, pretrained=False, enhancer="aether")
        clone.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        clone.eval()
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(model(images)[0], clone(images)[0])

    def test_default_baseline_unchanged(self):
        model = build_model("efficientnet_b0", PROJECT, pretrained=False)
        self.assertEqual(model.classifier[1].out_features, 8)
        self.assertFalse(hasattr(model, "enhancer"))
