"""PRISM: Pathology Residual Integration of Statistics and Morphology.

Experimental integration of known ideas, not three established novel methods:
multi-scale depthwise convolutions, moment-based channel gating, and spatial
attention pooling. See PRISM.md for provenance, hypotheses and ablations.
"""
import torch
from torch import nn
from torch.nn import functional as F
from network.morph import MORPH


class MorphologyContextRefiner(nn.Module):
    """MCR: reuse MORPH's multi-scale residual branch without its channel gate."""
    def __init__(self, dim):
        super().__init__()
        self.refiner = MORPH(dim, variant="no_gate")

    def forward(self, features):
        return self.refiner(features)


class StatisticalResponseCalibrator(nn.Module):
    """SRC: signed small residual channel calibration from spatial mean/std.

    These are latent-feature moments, not estimated H&E stain concentrations.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Conv1d(2, 1, kernel_size=5, padding=2)
        self.scale = nn.Parameter(torch.full((1, dim, 1, 1), 1e-3))

    def forward(self, features):
        mean = features.mean((2, 3))
        std = (features.var((2, 3), unbiased=False) + 1e-6).sqrt()
        moments = torch.stack((mean, std), dim=1)
        moments = F.layer_norm(moments, (features.shape[1],))
        correction = self.gate(moments).tanh().squeeze(1)[:, :, None, None]
        return features * (1 + self.scale * correction)


class TissueEvidencePool(nn.Module):
    """TEP: retain GAP and add a small learned spatial attention correction.

    No WSI aggregation: all positions belong to one input patch.
    """
    def __init__(self, dim, width=64):
        super().__init__()
        self.scorer = nn.Sequential(nn.GroupNorm(1, dim), nn.Conv2d(dim, width, 1),
                                    nn.GELU(), nn.Conv2d(width, 1, 1))
        self.scale = nn.Parameter(torch.full((1, dim), 1e-3))

    def forward(self, features):
        weights = self.scorer(features).flatten(2).softmax(-1)
        spatial = features.flatten(2)
        mean = spatial.mean(-1)
        attended = (spatial * weights).sum(-1)
        return mean + self.scale * (attended - mean)


class EfficientNetPRISM(nn.Module):
    def __init__(self, backbone, variant="full"):
        super().__init__()
        if variant not in ("full", "no_local", "no_stats", "no_pool"):
            raise ValueError(f"Unknown PRISM variant: {variant}")
        dim = backbone.classifier[1].in_features
        self.features = backbone.features
        self.mcr = MorphologyContextRefiner(dim) if variant != "no_local" else nn.Identity()
        self.src = StatisticalResponseCalibrator(dim) if variant != "no_stats" else nn.Identity()
        self.pool = TissueEvidencePool(dim) if variant != "no_pool" else nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.classifier = nn.Sequential(backbone.classifier[0], nn.Linear(dim, 8))

    def forward(self, images):
        return self.classifier(self.pool(self.src(self.mcr(self.features(images)))))
