"""MORPH: Multi-scale mORPHology residual enhancement (experimental).

Uses established depthwise-separable convolution, ECA-style pooled channel
gating, and LayerScale-style small residual scaling. These ingredients are
not claimed as original; the pathology-specific combination needs validation.
"""
import torch
from torch import nn


class MORPH(nn.Module):
    def __init__(self, dim, width=160, variant="full"):
        super().__init__()
        self.variant = variant
        self.norm = nn.GroupNorm(1, dim)
        self.reduce = nn.Conv2d(dim, width, 1)
        self.local3 = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.local5 = nn.Conv2d(width, width, 5, padding=2, groups=width) if variant != "single_scale" else None
        self.mix = nn.Conv2d(width * 2 if variant != "single_scale" else width, width, 1)
        self.act = nn.GELU()
        self.channel_gate = nn.Conv1d(1, 1, 5, padding=2, bias=False) if variant != "no_gate" else None
        self.expand = nn.Conv2d(width, dim, 1)
        self.scale = nn.Parameter(torch.full((1, dim, 1, 1), 1e-3))

    def forward(self, features):
        reduced = self.act(self.reduce(self.norm(features)))
        local = self.local3(reduced)
        if self.variant != "single_scale":
            local = torch.cat((local, self.local5(reduced)), dim=1)
        local = self.act(self.mix(local))
        if self.channel_gate is not None:
            pooled = local.mean((2, 3)).unsqueeze(1)
            gate = self.channel_gate(pooled).sigmoid().squeeze(1)[:, :, None, None]
            local = local * gate
        return features + self.scale * self.expand(local)


class EfficientNetMORPH(nn.Module):
    def __init__(self, backbone, variant="full"):
        super().__init__()
        dim = backbone.classifier[1].in_features
        self.features = backbone.features
        self.enhancer = MORPH(dim, variant=variant)
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(backbone.classifier[0], nn.Linear(dim, 8))

    def forward(self, images):
        enhanced = self.enhancer(self.features(images))
        return self.classifier(self.avgpool(enhanced).flatten(1))
