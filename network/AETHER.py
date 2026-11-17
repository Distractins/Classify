import torch
import torch.nn as nn
import torch.nn.functional as F

# 全称：**Adaptive Embedding for Tissue Heterogeneity & Equivariant Representation**
# 中文：**组织异质性自适应等变表征模块**（删掉 “面向”，干净利落）

class SignalSeparateBlock(nn.Module):
    """信号分离块：将特征投影至色度子空间、结构子空间，引入正交正则约束"""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.proj_color = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim)
        )
        self.proj_texture = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim)
        )

    def forward(self, x):
        feat_c = self.proj_color(x)
        feat_t = self.proj_texture(x)
        ortho_loss = torch.mean(torch.abs(torch.sum(F.normalize(feat_c, dim=-1) * F.normalize(feat_t, dim=-1), dim=-1)))
        return feat_c, feat_t, ortho_loss


class TintSuppressGate(nn.Module):
    """色调抑制门：自适应估计染色偏移，动态压制色度干扰分量"""
    def __init__(self, dim):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_color):
        gate = self.estimator(feat_color)
        feat_c_suppressed = feat_color * gate
        return feat_c_suppressed, gate


class StructureFocusMHA(nn.Module):
    """结构聚焦多头注意力：在结构分支内强化细粒度判别纹理"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_t):
        B, N, C = feat_t.shape
        qkv = self.qkv(self.norm(feat_t)).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_out(x)
        return feat_t + x


class DiscrimAnchorSet(nn.Module):
    """判别锚集合：维护全局类别锚点 + 难样本专属锚点"""
    def __init__(self, dim, num_classes=8, hard_cls_idx=(2, 6, 4), temp=0.1):
        super().__init__()
        self.temp = temp
        self.num_classes = num_classes
        self.hard_cls_idx = hard_cls_idx
        self.global_anchor = nn.Parameter(torch.randn(num_classes, dim))
        self.hard_anchor = nn.Parameter(torch.randn(len(hard_cls_idx), dim))
        self.proj_q = nn.Linear(dim, dim)

    def forward(self, tex_feat):
        B, N, D = tex_feat.shape
        q = self.proj_q(tex_feat)
        q = F.normalize(q, dim=-1)
        g_anchor = F.normalize(self.global_anchor, dim=-1)
        h_anchor = F.normalize(self.hard_anchor, dim=-1)

        sim_global = torch.matmul(q, g_anchor.T) / self.temp
        sim_hard = torch.matmul(q, h_anchor.T) / self.temp

        attn_g = F.softmax(sim_global, dim=-1)
        attn_h = F.softmax(sim_hard, dim=-1)

        enhance_g = torch.matmul(attn_g, g_anchor)
        enhance_h = torch.matmul(attn_h, h_anchor)
        enhanced = tex_feat + 0.25 * enhance_g + 0.15 * enhance_h
        sim_out = sim_global
        return enhanced, sim_out


class AETHER(nn.Module):
    """
    AETHER: Adaptive Embedding for Tissue Heterogeneity & Equivariant Representation
    组织异质性自适应等变表征模块
    即插即用，兼容 CNN([B,C,H,W]) / Transformer([B,N,C]) 主干特征
    输入：feature
    输出：enhanced_feature, sim_matrix, ortho_reg
    """
    def __init__(self, dim, num_classes=8, reduction=4, num_heads=8, temp=0.1, hard_cls_idx=(2, 6, 4)):
        super().__init__()
        self.dim = dim
        self.signal_sep = SignalSeparateBlock(dim, reduction)
        self.tint_gate = TintSuppressGate(dim)
        self.struct_attn = StructureFocusMHA(dim, num_heads)
        self.anchor_set = DiscrimAnchorSet(dim, num_classes, hard_cls_idx, temp)

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        feat_c, feat_t, ortho_reg = self.signal_sep(x)
        feat_c_suppressed, _ = self.tint_gate(feat_c)
        feat_t_refined = self.struct_attn(feat_t)
        feat_enhanced, sim_matrix = self.anchor_set(feat_t_refined)

        out_feat = feat_t_refined + 0.1 * feat_c_suppressed + 0.3 * feat_enhanced
        return out_feat, sim_matrix, ortho_reg
