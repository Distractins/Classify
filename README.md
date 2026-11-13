# README.md
> 项目名称：基于HMU‑CRC‑Hist550K结直肠癌微环境8分类的MSI状态预测完整实验流水线
> 复现论文：Large‑Scale Histological Image Dataset with Metadata for Colorectal Cancer Microenvironment
> 整体实验分为两大阶段：
> **阶段一：Patch级8类肿瘤微环境组织分类模型训练（不使用MSI标签）**
> **阶段二：患者级TME微环境特征统计 + 融合临床特征，构建MSI‑H/MSS二分类预测模型**
>
<AI_META>
project_name: HMU‑CRC‑Hist550K Full Pipeline: 8‑class TME classification → MSI‑H/MSS prediction
total_stage: 2
stage1_task: patch 8‑class histopathology image classification
stage1_classes: ["ADI","DEB","LYM","MUC","MUS","NORM","STR","TUM"]
stage1_input: 224×224 RGB病理patch图像
stage1_label: 8类组织标签；**本阶段不读取、不使用MSI/MSH‑H标签**
stage1_model_backbone_list: ["ViT‑base‑patch16‑224‑in21k","ResNet101","EfficientNet‑B0"]
stage1_loss_baseline: CrossEntropyLoss；可扩展FocalLoss用于难样本优化
stage1_optimizer: ViT使用AdamW；ResNet/EfficientNet使用Adam
stage1_output_artifacts: 模型权重pth文件，分类报告文本，归一化混淆矩阵图，多分类ROC‑AUC曲线图，每类Precision/Recall/F1柱状图

stage2_task: patient‑level MSI‑H / MSS binary classification
stage2_input_source:
  1. 阶段一训练完成的8分类权重，推理全部patch，输出每一张patch预测类别
  2. HMU‑CRC‑Clinical.xlsx临床元数据表，PatientID、年龄、性别、MSI状态标签
stage2_aggregate_logic: 根据PatientID聚合同一患者全部patch，统计8类组织占比百分比(TUM%,LYM%,STR%,MUC%,MUS%,NORM%,ADI%,DEB%)
stage2_feature: 8项微环境占比特征 + 年龄 + 性别
stage2_label: MSI‑H=1，MSS=0（二分类）
stage2_model_option: XGBoost / MLP全连接网络
stage2_output_artifacts: 患者特征csv表格；MSI预测ROC曲线；混淆矩阵；特征重要性柱状图；TME组分与MSI相关性热力图

critical_constraint: **全部数据集划分必须为Patient‑Level患者级别划分，禁止图片级随机划分，避免数据泄露。阶段1和阶段2必须复用同一套患者划分，不可重新划分数据集。**
known_bug_legacy_code: 原始ViTmodle.py、Resnet101.py、EfficientNet.py为图像级随机划分，存在数据泄露；硬编码Windows绝对磁盘路径，运行前必须修改为项目相对路径。
</AI_META>

## 目录完整结构
├── README.md                           # 本项目说明文档（AI 读取入口）
├── requirements.txt                    # python 依赖包
├── .gitignore                          # git 忽略配置文件
├── src/                                # 全部源代码
│   ├── stage1_train_8cls/              # 【阶段一】patch 八分类训练代码
│   │   ├── ViTmodle.py                 # ViT‑base‑patch16‑224‑in21k 训练评估
│   │   ├── Resnet101.py                # ResNet101 训练评估
│   │   ├── EfficientNet.py             # EfficientNet‑B0 训练评估
│   │   ├── patient_dataset_split.py    # 【核心】读取临床表，实现患者级数据集划分，防止数据泄露
│   │   └── focal_loss.py               # 可选扩展：FocalLoss 损失函数实现，用于难样本优化
│   └── stage2_msi_prediction/          # 【阶段二】MSI 预测全流程代码
│       ├── patch_infer_all.py          # 使用阶段一权重推理全部 patch，输出每张 patch 预测结果 csv（patient_id, image_path, pred_class）
│       ├── patient_tme_agg_stat.py     # 根据 patient_id 聚合 patch 预测结果，统计每个患者 8 类组织占比，输出 patient_tme_feature.csv
│       └── train_msi_model.py          # 读取 patient_tme_feature.csv + 临床表，拼接特征，训练 MSI 二分类模型，绘图评估
├── datasets/
│   ├── HMU‑CRC‑Hist550K/
│   │   ├── ADI/
│   │   ├── DEB/
│   │   ├── LYM/
│   │   ├── MUC/
│   │   ├── MUS/
│   │   ├── NORM/
│   │   ├── STR/
│   │   ├── TUM/
│   │   └── HMU‑CRC‑Clinical.xlsx      # 临床元数据表，PatientID、年龄、性别、TNM、分化、MSI‑H/MSS、生存信息
│   └── dataset_readme.md              # 数据集下载来源说明
├── outputs/
│   ├── weights/                        # 模型权重存放目录
│   │   ├── stage1_8cls/                # 阶段一八分类权重
│   │   │   ├── vit_best.pth
│   │   │   ├── resnet101_best.pth
│   │   │   └── efficientnet_best.pth
│   │   └── stage2_msi/                 # 阶段二 MSI 预测模型权重
│   │       └── msi_xgb_model.pkl
│   ├── figures/                         # 全部输出图片：混淆矩阵、ROC、柱状图、热力图
│   │   ├── stage1_8cls/                # 阶段一产出图
│   │   │   ├── *.png  归一化混淆矩阵、多分类 ROC 曲线、各类 Precision‑Recall‑F1 柱状图
│   │   └── stage2_msi/                 # 阶段二产出图
│   │       ├── msi_roc_curve.png       # MSI 二分类 ROC‑AUC 曲线
│   │       ├── msi_confusion.png       # MSI 预测混淆矩阵
│   │       ├── feature_importance.png  # 特征重要性柱状图（哪些 TME 组分影响 MSI 预测）
│   │       └── tme_msi_corr_heatmap.png # TME 微环境特征与 MSI 标签相关性热力图
│   └── intermediate_csv/               # 中间产出 csv 表格文件
│       ├── patch_infer_result.csv      # 所有 patch 推理结果：patient_id, 图片路径，预测类别
│       └── patient_tme_feature.csv     # 每个患者一行：8 类占比、年龄、性别、真实 MSI 标签
└── logs/                                # 训练日志、打印报告文本
├── stage1_log.txt
└── stage2_log.txt


## 实验完整流程说明（两大阶段）
### 📌阶段一：Patch 8分类模型训练
> 目标：训练模型识别单张224×224病理patch属于8种微环境组织中哪一类。**此阶段完全不使用MSI标签，只使用patch图像+组织类别标签**
1. 输入：`datasets/HMU‑CRC‑Hist550K`下ADI~TUM的patch图像
2. 数据划分规则：
    读取`HMU‑CRC‑Clinical.xlsx`获取PatientID，执行**患者级别划分**：把全部500位患者划分为训练患者集合、验证患者集合。同一患者全部patch只能全部属于训练集或者全部属于验证集。**禁止按图片随机划分**。
3. 模型可选：ViT / ResNet101 / EfficientNet‑B0；损失默认CrossEntropyLoss；可替换FocalLoss优化STR/MUS、LYM/DEB难分类别。
4. 训练、验证，保存最优权重到 `outputs/weights/stage1_8cls/`
5. **阶段一需要生成的统计图**
    1. 归一化混淆矩阵png：观察STR与MUS、LYM与DEB混淆程度
    2. 多分类ROC曲线，输出每一类AUC、macro‑AUC
    3. 各类Precision、Recall、F1分数柱状图，重点观察难分类别指标
    4. classification_report文本保存到logs

> 产物：8分类最优模型权重；阶段一各类评估图表；训练日志

### 📌阶段二：患者级TME特征统计 + MSI‑H/MSS预测
> 前提：阶段一训练完成，得到8分类权重；复用阶段一的【患者划分】，**禁止重新划分数据集**
> 分为3个子步骤

#### 子步骤2‑1：对全部patch做推理
1. 加载阶段一保存的8分类模型权重
2. 遍历全部patch图像，每一张图片经过模型推理，得到预测组织类别
3. 关联每一张patch归属的PatientID（需要建立图片文件名与PatientID映射关系）
4. 输出中间文件：`outputs/intermediate_csv/patch_infer_result.csv`
> 字段：`patient_id, image_file_path, predict_class`

#### 子步骤2‑2：按PatientID聚合统计患者级微环境TME占比特征
读取patch_infer_result.csv，按patient_id分组聚合：
统计该患者所有patch中8个类别的占比（百分比）
> `TUM%、LYM%、STR%、MUC%、MUS%、NORM%、ADI%、DEB%`

将统计得到的8项百分比特征，与`HMU‑CRC‑Clinical.xlsx`做merge拼接：
增加临床特征：年龄、性别；标签：MSI状态（MSI‑H=1，MSS=0）

输出中间表格：`outputs/intermediate_csv/patient_tme_feature.csv`
> 每一行代表一位患者；字段：patient_id，8类占比，年龄，性别，msi_label

#### 子步骤2‑3：训练MSI二分类预测模型
输入特征：`[TUM%,LYM%,STR%,MUC%,MUS%,NORM%,ADI%,DEB%,年龄,性别]`
标签：msi_label（MSI‑H / MSS二分类）
数据集划分：严格复用阶段一的患者划分，训练集患者用于训练MSI模型，验证集患者用于评估MSI模型，**不能打乱重新划分**。
模型可选：XGBoost（推荐，方便输出特征重要性）或者简单MLP。

#### ✅阶段二需要生成的论证统计图（全部保存至`outputs/figures/stage2_msi/`）
1. MSI预测ROC曲线，输出AUC
2. MSI任务混淆矩阵
3. **特征重要性柱状图**：观察哪一个TME组分对MSI预测贡献最大（例如LYM淋巴细胞占比）
4. **TME特征‑MSI标签相关性热力图**：展示8个微环境占比特征、年龄性别与MSI标签之间相关系数，用于论文可解释论证。

> 消融实验组设置：
> ①仅临床特征（年龄+性别）做MSI预测（基线）
> ②仅TME微环境占比特征做MSI预测
> ③TME占比特征 + 年龄性别融合（本实验完整方案）
> 对比三组AUC，证明加入精准微环境特征可以提升MSI预测性能。

## 运行顺序
```bash
# 阶段一：八分类训练
python src/stage1_train_8cls/patient_dataset_split.py   # 执行患者级划分
python src/stage1_train_8cls/EfficientNet.py             # 可选 ViTmodle.py / Resnet101.py

# 阶段二
python src/stage2_msi_prediction/patch_infer_all.py      # 全部patch推理
python src/stage2_msi_prediction/patient_tme_agg_stat.py  # 聚合得到患者TME特征表
python src/stage2_msi_prediction/train_msi_model.py       # 训练MSI预测模型、绘图
