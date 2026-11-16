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

## 当前工作：数据探查与整患者子集

数据已解压。本阶段仅做探查、统计和子集构造，**不训练模型**。先在 Windows VSCode 调试，再迁移到 4080Super 机器。上方 `<AI_META>` 保持原文；下文更新了当前数据准备流程，替代旧的逐 patch 限额采样与全量训练步骤。

**子集是从约 50G 全量数据中挑选的小规模版本。后续八分类训练只使用子集，不读取全集训练。患者是不可拆分单元：一位患者的所有类别、所有 patch 全部复制，或者完全不进入子集。** 原始数据只读，复制不会移动或删除原图；因此选中患者的完整副本仍存在于原始数据中。

当前不创建训练/验证划分。后续只对选中患者生成一次患者级划分，阶段一和阶段二复用，同一患者不得跨训练/验证集。MSI 仅用于本次子集患者分层和后续 MSI 任务，不作为八类组织训练的标签或输入。

## 目录与路径配置

```text
Classify/
├── README.md
├── data_explore.py
├── patient_stratified_sampling.py
├── utils/
│   └── path_config.py
├── datasets/
│   ├── HMU‑CRC‑Hist550K/                  # 原始全量数据，不改动
│   │   ├── ADI/
│   │   ├── DEB/
│   │   ├── LYM/
│   │   ├── MUC/
│   │   ├── MUS/
│   │   ├── NORM/
│   │   ├── STR/
│   │   ├── TUM/
│   │   └── HMU‑CRC‑Clinical.xlsx
│   └── HMU‑CRC‑Hist550K-SUBSET/           # 整患者小规模子集，自动生成
│       ├── ADI/ ... TUM/                 # 所选患者的全部 patch
│       ├── HMU‑CRC‑Clinical.xlsx         # 仅所选患者，保留原表字段
│       ├── patch_patient_map.csv
│       ├── selected_patients.csv
│       └── sampling_report.json
├── output/                               # 本阶段产物
│   ├── stat/                             # 全量数据探查 CSV 和图片
│   ├── sampling/                         # 采样计划与报告
│   └── .matplotlib/                      # 项目内绘图缓存
├── pretrained/                           # 已下载的预训练模型
├── src/stage1_train_8cls/                 # 已有八分类训练代码，本阶段不运行
├── src/stage2_msi_prediction/             # 已有 MSI 代码，本阶段不运行
├── outputs/                              # 后续训练权重、CSV、评估图
└── tests/
```

只修改 `utils/path_config.py` 中的根路径：

```python
ROOT = Path(r"D:\xcjhhh\Classify")
# Linux 机器示例：ROOT = Path(r"/data/Classify")
```

两个处理脚本都 import 此配置，不依赖 VSCode 终端当前目录。命令行 `--root` 可临时覆盖根路径。数据目录使用固定名称中的不换行连字符 `‑`（U+2011）；请保持磁盘名称与配置一致。若实际数据使用普通 `-`，在路径配置文件内统一修改 `DATASET_NAME` 和 `CLINICAL_NAME` 即可。

## 1. 环境与本地检查

在 VSCode 选择 `YuHou` 解释器，终端进入项目根目录：

```powershell
conda activate YuHou
python -m pip install pandas matplotlib "opencv-python<4.12" pillow tqdm openpyxl
python utils/path_config.py
python data_explore.py --help
python patient_stratified_sampling.py --help
```

`pathlib`、`argparse`、`random`、`shutil` 等属于 Python 标准库。Excel 读写需要 `openpyxl`。当前两个数据处理步骤使用 CPU，不要求 CUDA 或任何预训练权重。

## 2. 数据探查

推荐分批扫描，每条命令可以单独运行，中断后重复同一条即可继续：

```powershell
python -X faulthandler -u data_explore.py --scan-class ADI --max-files 5000 --checkpoint-every 200
```

重复运行直到 ADI 进度达到 100%，然后将 ADI 依次改为 DEB、LYM、MUC、MUS、NORM、STR、TUM。也可不加 `--max-files`，一次处理完一个类别。每批仍枚举当前类别文件名，但最多解析指定数量的新图片，不同时扫描其他类别。分批扫描不读取临床表、不生成不完整的全量映射。

八类完成后，只读取数据库生成报告和图：

```powershell
python data_explore.py --images-only --summarize
```

任一类别未完成时会拒绝汇总。已有旧断点可继续使用，但每类至少用 `--scan-class` 执行一次以核验列表和写入完成标记；已缓存图片不会重复解析。分批期间请保持原图目录不变；`--summarize` 统计的是已扫描的快照，不检查后续增删。仍支持下面的一次性扫描方式。

```powershell
python data_explore.py
# 临床表存在冲突、暂时只需统计图片规模时，不读取临床表。
python data_explore.py --images-only
# 默认断点续跑：中断后再次执行同一命令即可。
# 更频繁保存断点（每 200 张），减少进程被强制终止时的重复工作。
python data_explore.py --images-only --checkpoint-every 200
# 只有原始文件列表或解析规则变化、或需要重新扫描时才使用：
python data_explore.py --images-only --restart-scan
# 可选：解码全部图片排查损坏文件；会增加磁盘读取时间。
python data_explore.py --check-images pil
# 或使用 OpenCV，支持 Windows 中文路径。
python data_explore.py --check-images opencv
```

按八个组织文件夹识别类别。真实文件名如 `1302097_0_19264_ADI.png`，取首段字符串 `1302097` 为患者 ID，并核对末段组织类别与文件夹一致；也兼容此前示例 `patient_00101_001.png`，取 `00101`。支持 png、jpg、jpeg、tif、tiff、bmp，不把患者 ID 转成整数，不自动补零。若 Excel 将 ID 实际存为数值 `101`，即使显示为 `00101` 也不会自动匹配，应核查原始 ID 后修正临床表。

临床患者列支持 Patient、PatientID 等，MSI 列优先识别 MSI、MSI status 等；若缺少这些列，则从 immunohistochemistry 中仅识别明确写出的 MSI-H/MSS，空白及 IHC 描述统一记为 UNKNOWN，不根据染色阳性比例推断标签。若没有可用列，仍可探查图片，但无法平衡 MSI-H/MSS。若列名无法识别，可在两个脚本上都指定：

```powershell
python data_explore.py --id-column "患者编号" --msi-column "MSI状态"
```

MSI-H 和 MSS 分别统计；MSI-L、空值及其他未明确编码统一记为 UNKNOWN，保留原始状态，绝不擅自归入 MSS。输出临床表全部患者分布及实际有 patch 的患者分布，避免把无图像患者算入可采样人数。

`output/stat/` 产物：

扫描断点保存在 `output/stat/scan_checkpoint.sqlite3`。默认每 1000 张提交一次，Ctrl+C 或解析异常也会保存此前成功解析的进度。中断后重新执行同一命令，复用已完成类别和当前类别已保存的图片；进程被强制关闭时，只需重做尚未提交的部分。请保留该断点文件。恢复时仍会读取目录文件列表以检查新增或删除，不会重新解析已缓存图片。修改文件列表或解析规则后需加 `--restart-scan`，该参数只重建扫描缓存，不改动原始数据。统计、绘图以及可选的图片解码检查会在扫描完成后重新运行，本轮未对这些步骤增加断点。正式三列映射表仅在八类扫描全部完成后生成，采样脚本仍独立完整复扫，不把扫描断点当作完整映射。

| 文件 | 内容 |
| --- | --- |
| patch_patient_map.csv | img_name,patient_id,cls；img_name 为类别内相对路径 |
| class_patch_counts.csv | 八类 patch 总数 |
| patient_patch_stats.csv | 每患者总数、八类计数、临床匹配及 MSI 状态 |
| msi_patient_counts.csv | MSI-H/MSS/UNKNOWN 患者数量 |
| unmatched_patients.csv | 无临床匹配或无 patch 的患者 |
| summary.json | 全量统计摘要 |
| class_patch_counts.png | 类别数量柱状图 |
| patient_patch_histogram.png | 每患者 patch 数量直方图 |
| patient_class_heatmap.png | 患者 × 八类 patch 计数热力图 |
| invalid_images.csv | 仅启用图片验证时生成，列出解码失败图片 |

默认热力图包含所有有图片的患者，标签过多时只稀疏显示部分标签，CSV 仍完整。可用 `--heatmap-max-patients 100` 仅绘制 patch 最多的前 100 位，图标题会标明范围。损坏图片不会被静默剔除，请修复后重跑。

## 3. 患者级分层抽样

先根据统计结果设定预算。下面的 20000 和 5000 是示例上限，并非必须凑满的目标：

```powershell
# 先预览：不复制图片。
python patient_stratified_sampling.py --max-total-patches 20000 --max-per-class 5000 --seed 42 --dry-run

# 确认计划后正式复制。
python patient_stratified_sampling.py --max-total-patches 20000 --max-per-class 5000 --seed 42

# 个别类别可覆盖统一上限，--class-cap 可重复。
python patient_stratified_sampling.py --max-total-patches 20000 --max-per-class 5000 --class-cap LYM=3000 --class-cap TUM=4000 --dry-run
```

策略是整患者贪心选择：先检查加入该患者全部 patch 后是否满足总量和全部类别上限；优先增加尚未覆盖的组织类别，并考虑稀有类别，再优先 MSI 少数患者层，随后优先较小患者包，相同优先级用固定种子打破平局。UNKNOWN 患者仍可为组织覆盖提供样本，不参与 MSI-H/MSS 平衡计数。

**总量和每类上限是硬约束；八类覆盖、MSI-H/MSS 数量平衡是尽量满足的目标。** 这是启发式策略，不保证全局最优、恰好等量或用满预算。若单个患者已超过某项上限，该患者整位排除并记录；如果没有患者能满足限制，明确报错，绝不截取患者的一部分 patch。限制过严导致类别缺失时会报告，需调整预算。

抽样前完整复扫并核对映射，防止旧表、缺行或新图片造成患者数据不完整；复制后再次扫描核对。原图始终不动。先写入 `.building-*` 临时目录，全部成功后才发布正式子集；失败时保留临时目录并打印位置。

`output/sampling/` 输出 `patient_selection.csv`、`planned_patch_patient_map.csv`、`sampling_report.json`，包括每患者是否选择、计划内全部图片、最终八类计数、MSI 分布、缺失类别和超限患者。正式子集另附三列映射和所选临床表。已有正式子集时拒绝覆盖，重新选参数前请自行备份并移走旧子集；预览报告会被最新一次运行更新。

图片患者未匹配临床 ID 时抽样停止，请根据 `unmatched_patients.csv` 修正匹配，不会静默丢弃患者或猜测前导零。

已确认患者 `563385` 有两条临床冲突记录，按用户授权隔离处理：原始 Excel 不修改，两条记录仍完整导出到 `output/stat/clinical_id_issues.csv`；患者只计一次，MSI 设为 UNKNOWN，`clinical_conflict=True`、`msi_training_eligible=False`。该患者仍可作为整患者参与组织子集选择，不强制选中，若选中则复制全部图片。子集 Excel 为其生成唯一占位行，除 ID 与标记外临床取值留空，不采用任何一条冲突记录。后续有监督 MSI 训练必须排除该患者；当前不运行训练。其他重复 ID 或空 ID 仍会报错，不能静默合并。

## 4. 本地调试 → 4080Super 机器

1. 本地修改 `path_config.py` 的 ROOT，按固定结构放好解压数据和临床表。
2. 运行探查，检查统计、前导零、临床关联；先用 `--dry-run` 调整小规模预算。
3. 正式构建子集，确认患者集合、完整性、每类数量和 MSI 分布。
4. 将源码和所需数据上传到 4080Super 机器；Windows 虚拟环境不要直接复制到 Linux。
5. 只修改该机器 `utils/path_config.py` 的 ROOT；保持子目录结构不变，重新安装上述依赖。
6. 按相同命令进行探查和抽样；相同数据、参数和种子得到同一患者选择。也可直接上传本地已生成的完整子集。

可运行合成数据测试，不需要真实数据和 GPU：

```powershell
python -m unittest discover -s tests -p test_data_preparation.py -v
```

后续训练数据入口应明确指向 `datasets/HMU‑CRC‑Hist550K-SUBSET/`，临床入口指向该子集内的 `HMU‑CRC‑Clinical.xlsx`。不要使用旧的全量患者划分名单，子集实验应在独立输出目录中创建属于该子集的一套患者名单并让两个阶段复用；旧实验名单保留备份。本次不修改或运行已有训练代码，接入子集后再开展训练。

旧方案 `build_patch_map.py`、`patient_based_sampling.py`、`ptach.py` 已删除；当前仅使用 `data_explore.py` 和 `patient_stratified_sampling.py` 准备数据。

## 后续完整实验流水线与产物

Patch 八分类（组织标签）→ 选中患者全部 patch 推理 → 每患者八类占比 → 加入年龄/性别 → MSI-H/MSS 二分类。由于本次复制整患者，选中患者的组织占比不会被患者内限额抽样改变。未来训练/验证也必须以患者为单元。

`outputs/weights/stage1_8cls/` 保存训练生成的 `vit_best.pth`、`resnet101_best.pth` 或 `efficientnet_best.pth`，无需提前下载这些八分类权重。`pretrained/` 保存已下载的通用预训练参数。

`outputs/weights/stage2_msi/msi_xgb_model.pkl` 保存 MSI 模型；`outputs/intermediate_csv/` 保存患者名单、patch 推理和 TME 特征。`outputs/figures/stage1_8cls/` 保存组织分类评估图；`outputs/figures/stage2_msi/` 保存 MSI ROC、混淆矩阵、特征重要性和相关性热力图；`logs/` 保存训练报告。本阶段只生成单数 `output/` 下的准备产物及子集，不启动上述训练流水线。
