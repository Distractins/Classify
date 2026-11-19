"""Jointly train EfficientNet spatial features, AETHER and a linear head."""
from torch import nn
from network.AETHER import AETHER


class EfficientNetAETHER(nn.Module):
    def __init__(self, backbone, ortho_weight=0.01):
        super().__init__()
        dim = backbone.classifier[1].in_features
        self.features = backbone.features
        self.enhancer = AETHER(dim, num_classes=8, hard_cls_idx=(2, 6, 4))
        self.dropout = backbone.classifier[0]
        self.head = nn.Linear(dim, 8)
        self.ortho_weight = ortho_weight

    def forward(self, images):
        tokens, _, ortho = self.enhancer(self.features(images))
        logits = self.head(self.dropout(tokens.mean(1)))
        return logits, self.ortho_weight * ortho
