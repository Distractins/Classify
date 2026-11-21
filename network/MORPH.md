# MORPH：多尺度组织形态残差增强模块

实验性方法名：Multi-scale mORPHology Residual Enhancement。此名称指整体组合，不代表组成算子的原创性或已验证的染色不变性。

旧 AETHER 存在的设计风险：用未经监督的正交投影声称色度/结构解耦；没有原始主干特征的直接恒等通路；hard_cls_idx 只确定锚点数量，没有将锚点与指定类别标签显式绑定。以上是代码层面的风险，不能代替实际失败原因诊断。本地目前没有 AETHER 实验结果，因此未对其训练收敛进行归因。

新结构：EfficientNet 最终特征 F → GroupNorm → 1×1降维到160通道 → 并行3×3/5×5深度卷积 → 拼接与1×1混合 → 通道门控 → 1×1升维 → 小幅逐通道残差融合 F + γΔF → 原有全局平均池化/Dropout → 8类Linear。

γ初始化1e-3，保留预训练特征通路，避免一开始就完全替换原特征。不使用色度解耦正交损失、类别锚点或困难类别特定损失。主干/模块/分类头全量训练，使用原始CE目标。局部卷积作用在末级低分辨率特征图，不应宣称直接恢复细胞尺度细节。

组成思想与来源须如实描述：深度可分离卷积、多分支多尺度融合为已有设计；池化后1D卷积的通道门控借鉴 ECA-Net（CVPR 2020）；小初始化逐通道残差系数借鉴 LayerScale（CaiT, ICCV 2021）。本实现不是这些论文的完整复现，不得以重命名主张这些算子的原创性。可研究的贡献是组合位置、轻量设计与组织分类/染色偏移实验的有效性，需要验证与文献检索支持。

训练示例：

```bash
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL-CAP50" --model efficientnet_b0 --enhancer morph --run-name cap50_10ep --epochs 10 --batch-size 8 --lr 0.0001 --weight-decay 0.00001 --save-every-batches 100
```

续跑加 --resume；原始测试加 --evaluate-test，不加 resume。独立输出 outputs/experiments/efficientnet_b0_morph/cap50_10ep/。染色测试从 best checkpoint 自动识别 morph。新结构不能接旧基线或 AETHER 的断点，需从 ImageNet 预训练开始公平对比。

消融：同一指令分别将 enhancer 换为 morph_single_scale（去5×5分支）、morph_no_gate（去通道门控）；输出自动分目录。比较 none / AETHER / MORPH / 两种消融，保持数据划分、增强、学习率、batch及验证集权重选择规则相同。建议报告多个随机种子均值/方差（当前训练器固定42，尚无CLI切换种子），参数量、耗时以及各类别指标。架构选型以验证集为准，已查看的测试集不能反复用于选型并当作独立确认；有条件时增加独立外部中心测试。性能未实测，不能保证超过基线；模拟染色扰动测试不等同于跨中心验证。
