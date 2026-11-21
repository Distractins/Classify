"""Frozen UNI tokens with trainable patch-classification heads."""
import torch
from torch import nn


class FrozenUNIClassifier(nn.Module):
    def __init__(self, backbone, mode="linear", hidden_dim=256, dropout=0.1,
                 ortho_weight=0.01, hard_cls_idx=(2, 6, 4)):
        super().__init__()
        if mode not in ("linear", "mlp"):
            raise ValueError("Frozen UNI supports only linear and mlp heads")
        self.backbone = backbone.requires_grad_(False)
        self.backbone.eval()
        self.mode = mode
        self.ortho_weight = ortho_weight
        dim = backbone.num_features
        self.head = (nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(hidden_dim, 8)) if mode == "mlp" else nn.Linear(dim, 8))

    def train(self, mode=True):
        super().train(mode)
        # Outer model.train() must never activate backbone dropout or update its buffers.
        self.backbone.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            tokens = self.backbone.forward_features(images)
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise ValueError("UNI forward_features must return [B,N,C] tokens")
            prefix = getattr(self.backbone, "num_prefix_tokens", 1)
            tokens = tokens[:, prefix:, :]
            if tokens.shape[1] == 0:
                raise ValueError("UNI returned no spatial tokens")
        # Both heads use identical spatial-token mean pooling (no CLS token).
        return self.head(tokens.mean(1))
