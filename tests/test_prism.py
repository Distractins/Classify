import unittest
import torch
from network.prism import StatisticalResponseCalibrator, TissueEvidencePool
from src.stage1_train_8cls.train_wsi_baseline import build_model, PROJECT


class PRISMTests(unittest.TestCase):
    def test_zero_scales_preserve_features_and_average_pool(self):
        x = torch.randn(2, 32, 5, 5)
        calibrator, pool = StatisticalResponseCalibrator(32), TissueEvidencePool(32, width=8)
        with torch.no_grad():
            calibrator.scale.zero_()
            pool.scale.zero_()
        torch.testing.assert_close(calibrator(x), x, rtol=0, atol=0)
        torch.testing.assert_close(pool(x), x.mean((2, 3)))
        self.assertTrue(torch.isfinite(calibrator(torch.ones(2, 32, 1, 1))).all())

    def test_all_components_train_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        for variant in ("prism", "prism_no_local", "prism_no_stats", "prism_no_pool"):
            model = build_model("efficientnet_b0", PROJECT, pretrained=False, enhancer=variant)
            images = torch.randn(2, 3, 64, 64)
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (2, 8))
            torch.nn.functional.cross_entropy(logits, torch.tensor([4, 6])).backward()
            for component in (model.features, model.mcr, model.src, model.pool, model.classifier):
                parameters = list(component.parameters())
                if parameters:
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
                    self.assertTrue(any(p.grad.abs().sum() > 0 for p in parameters))
            restored = build_model("efficientnet_b0", PROJECT, pretrained=False, enhancer=variant)
            restored.load_state_dict(model.state_dict(), strict=True)
            model.eval()
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(images), restored(images))
