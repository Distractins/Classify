# MORPH：多尺度组织形态残差增强模块

续跑加 `--resume`；原始测试加 `--evaluate-test`，不加 resume。输出为 `outputs/experiments/efficientnet_b0_morph/<run-name>/`，染色测试自动从 checkpoint 恢复结构。不能跨结构续跑。

消融使用 `--enhancer morph_single_scale`（去5×5分支）、`--enhancer morph_no_gate`（去通道门控），与 `none` 和 `morph` 对比，保持相同数据、训练预算和验证集选择规则。性能需实验验证，不应根据测试集反复选型；模拟染色偏移也不能代替外部中心验证。

实验性方法名：Multi-scale mORPHology Residual Enhancement。此名称指整体组合，不代表组成算子的原创性或已验证的染色不变性。


新结构：EfficientNet 最终特征 F → GroupNorm → 1×1降维到160通道 → 并行3×3/5×5深度卷积 → 拼接与1×1混合 → 通道门控 → 1×1升维 → 小幅逐通道残差融合 F + γΔF → 原有全局平均池化/Dropout → 8类Linear。

γ初始化1e-3，保留预训练特征通路，避免一开始就完全替换原特征。不使用色度解耦正交损失、类别锚点或困难类别特定损失。主干/模块/分类头全量训练，使用原始CE目标。局部卷积作用在末级低分辨率特征图，不应宣称直接恢复细胞尺度细节。

组成思想与来源须如实描述：深度可分离卷积、多分支多尺度融合为已有设计；池化后1D卷积的通道门控借鉴 ECA-Net（CVPR 2020）；小初始化逐通道残差系数借鉴 LayerScale（CaiT, ICCV 2021）。本实现不是这些论文的完整复现，不得以重命名主张这些算子的原创性。可研究的贡献是组合位置、轻量设计与组织分类/染色偏移实验的有效性，需要验证与文献检索支持。

训练示例：

```bash
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL-CAP50" --model efficientnet_b0 --enhancer morph --run-name cap50_10ep --epochs 10 --batch-size 8 --lr 0.0001 --weight-decay 0.00001 --save-every-batches 100
```


