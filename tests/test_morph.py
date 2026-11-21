import unittest
import torch
from network.morph import MORPH
from src.stage1_train_8cls.train_wsi_baseline import build_model, PROJECT


class MORPHTests(unittest.TestCase):
    def test_zero_residual_is_identity_and_variants_receive_gradients(self):
        torch.set_num_threads(1)
        for variant in ("full", "single_scale", "no_gate"):
            module = MORPH(32, width=16, variant=variant)
            x = torch.randn(2, 32, 7, 7)
            with torch.no_grad():
                module.scale.zero_()
            torch.testing.assert_close(module(x), x, rtol=0, atol=0)
            with torch.no_grad():
                module.scale.fill_(1e-3)
            module(x).square().mean().backward()
            self.assertGreater(module.reduce.weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(module.scale.grad).all())

    def test_real_backbone_and_strict_restore(self):
        torch.set_num_threads(1)
        for variant in ("morph", "morph_single_scale", "morph_no_gate"):
            model = build_model("efficientnet_b0", PROJECT, pretrained=False, enhancer=variant)
            model.train()
            logits = model(torch.randn(2, 3, 64, 64))
            self.assertEqual(logits.shape, (2, 8))
            torch.nn.functional.cross_entropy(logits, torch.tensor([4, 6])).backward()
            self.assertGreater(model.features[0][0].weight.grad.abs().sum().item(), 0)
            copy = build_model("efficientnet_b0", PROJECT, pretrained=False, enhancer=variant)
            copy.load_state_dict(model.state_dict(), strict=True)
