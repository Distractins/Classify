# PRISM：病理形态与统计响应残差协同网络

Pathology Residual Integration of Statistics and Morphology。实验性整体名称，不代表已经验证的三项原创贡献，也不表示具有染色不变性。新增结构保留 MORPH 和原始 EfficientNet 作为对照，不改变已训练权重的解释。

## 这次结果支持什么

MORPH 常规测试 Macro-F1 0.8329，原基线0.8310，差约0.19个百分点；验证集最佳值却从0.8427变为0.8399。小幅测试收益不足以证明稳定优势。STR/MUS 的错分存在方向转移，不能仅凭单方向误差变小声称解决类别混淆。Macenko 下 MORPH F1 从0.8329降到0.7960；缺少同协议的基线染色测试，不能判断相对染色鲁棒性提升。HED 回传报告不完整。

新设计依据验证集中的 STR/MUS、NORM/LYM、DEB/STR 错分提出假设，而非根据测试集分数反复筛选。局部形态、通道响应、空间汇聚是三个待验证维度；仅增加模块数量不构成三项创新。

## 三个候选组件

1. MCR — Morphology Context Refiner，形态上下文细化器：重用 MORPH 的降维、并行3×3/5×5深度卷积、升维与小残差，去掉其通道门控，避免重复门控。假设：补充末级组织形态上下文。借鉴已有多尺度卷积及深度可分离卷积。
2. SRC — Statistical Response Calibrator，统计响应校准器：使用每个通道空间均值和标准差作为两个描述子，通道轴 LayerNorm 后经1D卷积产生校准量；采用小幅有符号乘性残差，不直接替换原特征。假设：根据响应强度及离散程度调节通道。特征统计不等于H/E浓度，更不证明染色解耦。借鉴 SRM（Style-based Recalibration Module, ICCV 2019）的统计描述思想与 ECA-Net（CVPR 2020）的局部通道交互；本实现不是任一论文的完整复现。
3. TEP — Tissue Evidence Pool，组织证据汇聚器：学习单张patch内部的空间权重，将注意力加权均值与原始全局均值作小幅残差融合。假设：避免所有空间位置等权。借鉴通用空间注意力池化，与注意力式实例聚合思想有关；不属于患者/WSI级MIL，也没有注意力区域正确性的监督或保证。

三个组件使用1e-3初始化的残差系数（LayerScale/CaiT, ICCV 2021 风格），保留原始基线路径。TEP 表达式为 GAP(F) + γ×(AttentionPool(F)-GAP(F))；其γ为0时严格回到GAP。所有模块及主干共同训练，仅使用原有CE损失。没有新增正交损失、困难类别锚点或测试时自适应。

结构：EfficientNet末级特征 → MCR → SRC → TEP → 原dropout概率 → Linear八分类。

局限：仍只使用最终低分辨率特征图；多尺度分支并非跨主干层融合，不能称为细胞尺度重建。SRC可能增强而不是消除染色敏感性，TEP可能关注伪相关区域，新增小残差也可能学不到有效校正；均需消融与可视化验证。

## 训练与恢复

```bash
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL-CAP50" --model efficientnet_b0 --enhancer prism --run-name cap50_10ep --epochs 10 --batch-size 8 --lr 0.0001 --weight-decay 0.00001 --save-every-batches 100
```

续跑同一指令加 `--resume`；常规测试同一数据/模型/enhancer/run-name 加 `--evaluate-test`，不要加resume。新结构从相同ImageNet预训练开始，不能加载 MORPH 或裸基线的训练断点。输出 `outputs/experiments/efficientnet_b0_prism/cap50_10ep/`。染色评估从 best.pth 自动恢复结构，三种 method 保持相同的强度/种子。

## 必须安排的消融

| enhancer | 组件 | 目的 |
|---|---|---|
| none | 原始基线 | 比较总体收益 |
| morph | 旧 MORPH | 检验新增设计是否优于旧版 |
| prism | MCR + SRC + TEP | 完整组合 |
| prism_no_local | SRC + TEP | 去掉 MCR |
| prism_no_stats | MCR + TEP | 去掉 SRC |
| prism_no_pool | MCR + SRC + GAP | 去掉 TEP |

只需替换 enhancer，各自输出目录独立。上述去一项消融只能显示组合内边际作用，若声称每个单独模块均有效，还应增加基线+各单模块实验；当前CLI未提供该组实验。去模块的随机初始化消耗不同，因此即使固定种子，分类头/主干随机掩码也不保证逐步一致。

先跑完整组合与基线的验证比较，再决定是否完成全部消融。相同划分、训练增强、学习率、batch、预算和权重选择标准；不要针对每个模型根据测试集调参。报告每类F1、均值指标、参数量、吞吐、原始及染色偏移下指标下降；单个随机种子的小幅差异不能证明显著性。当前训练器固定seed42，多种子统计尚未实现。以后若做置信区间，应按WSI重采样，不能把高度相关patch当作独立个体。

新算法设计已参考同一数据集的实验，最终独立结论宜增加未用于设计的新WSI或外部中心验证。现阶段可以称为三个候选技术组件，论文原创性需文献检索、清晰区别和重复实验支撑，不能通过重命名掩盖已有模块来源。
