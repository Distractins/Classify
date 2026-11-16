# HMU‑CRC‑Hist550K：Patch 级八分类数据准备

本项目仅开展 ADI、DEB、LYM、MUC、MUS、NORM、STR、TUM 八类组织的 patch 分类实验。因缺少公开的 slide–patient 对照关系，放弃患者级 MSI 预测。本流程不读取、不复制、不使用临床 Excel，不包含训练模型代码。

<AI_META>
project_name: HMU-CRC-Hist550K patch 8-class classification
current_scope: statistics, sampling, WSI dataset splitting
classes: [ADI, DEB, LYM, MUC, MUS, NORM, STR, TUM]
group_id: slideID = filename.split("_")[0], preserved as string
patient_id: unavailable; never infer from slideID
clinical_excel: unused; no MSI task
seed: 42
fast_mode: per-class patch sampling for debugging only; no formal split
paper_mode: select WSI groups, retain all PNG patches, split WSI groups 70/15/15
critical_constraint: paper_mode train/val/test slideID sets must be disjoint
paths: utils/path_config.py
</AI_META>

## 目录结构

```text
Classify/
├── dataset_stat.py
├── dataset_sampling.py
├── utils/path_config.py
├── datasets/
│   ├── HMU‑CRC‑Hist550K/             # 原始目录，只读
│   │   ├── ADI/ DEB/ LYM/ MUC/ MUS/ NORM/ STR/ TUM/
│   │   └── HMU-CRC-Clinical.xlsx     # 不读取
│   ├── HMU‑CRC‑Hist550K-FAST/        # 调试子集
│   │   ├── ADI/ ... TUM/
│   │   ├── mapping.csv
│   │   ├── split_mapping.csv
│   │   └── slide_splits.csv
│   └── HMU‑CRC‑Hist550K-PAPER/       # 正式 WSI 子集
│       ├── train/                  # 八类目录及 mapping.csv
│       ├── val/                    # 同上
│       ├── test/                   # 同上
│       ├── mapping.csv
│       ├── split_mapping.csv
│       └── slide_splits.csv
└── output/
    ├── stat/
    │   ├── slide_scan_cache/        # 逐类别映射缓存
    │   ├── mapping.csv
    │   ├── class_patch_counts.csv
    │   ├── slide_patch_counts.csv
    │   └── class_patch_counts.png
    └── .matplotlib/
```

原有 data_explore.py、patient_stratified_sampling.py 和患者/MSI 脚本属于旧流程，不再使用；旧路径配置接口已退出本流程。新脚本不依赖旧 SQLite 断点，不删除原始数据或历史报告。

## 环境与路径

Python 3.8 或更新版本。仅需安装 pandas、matplotlib、tqdm；其余依赖为标准库。不需要 torch、CUDA、模型权重或 Excel 读取库。

在目标环境安装（当前使用 xcjenv；如使用 YuHou，先激活 YuHou）：

```powershell
conda activate xcjenv
python -m pip install pandas matplotlib tqdm
```

路径只在 utils/path_config.py 的 CONFIG 区域配置：

```python
ROOT = Path(r"D:\xcjhhh\Classify")
```

其他脚本顶部 CONFIG 导入这一配置，不重复硬编码路径。目录名使用 U+2011 连字符 `‑`；若磁盘使用普通 `-`，在同一个配置文件修改 DATASET_NAME。

```powershell
python utils/path_config.py
```

## 第一步：数据统计

```powershell
python dataset_stat.py
```

仅扫描 PNG（大小写不敏感，可递归），不解码图片。示例 `1302097_896_14784_ADI.png` 的首段是 slideID，保留前导零。支持同类重复后缀，例如 `1434704_1568_21728_ADI_ADI.png`。目录不存在、无 PNG、文件名或类别异常时会报错，不静默漏图。

分批运行时，依次执行：

```powershell
python dataset_stat.py --scan-class ADI
python dataset_stat.py --scan-class DEB
python dataset_stat.py --scan-class LYM
python dataset_stat.py --scan-class MUC
python dataset_stat.py --scan-class MUS
python dataset_stat.py --scan-class NORM
python dataset_stat.py --scan-class STR
python dataset_stat.py --scan-class TUM
python dataset_stat.py --summarize
```

每类成功后保存缓存；中断只需重跑未完成类别，不支持类别内部逐文件断点。`--summarize` 仅汇总八类缓存，任何类别未完成会拒绝汇总。扫描期间保持原图不变，移动项目或修改数据后重新扫描。分批不能修复 Python 环境的 Illegal instruction 等原生崩溃。

若已将旧 output 改名为 output_legacy，可直接复用已完成的图片映射：

```powershell
python dataset_stat.py --import-legacy-map "output_legacy/stat/patch_patient_map.csv"
```

这会逐行核对旧 patient_id 是否等于文件名首段，再将其作为 slideID，检查类别、重复记录和路径。只有八类都存在且全部记录通过验证，才生成新缓存并汇总，不读取临床文件、不修改旧 CSV、不重新扫描原图。导入的是旧文件清单快照，不能证明磁盘数据未变化；采样前仍会复扫当前原图清单。建议核对本次总数是否与之前的 547,971 张、500 个编号一致。

产物位于 output/stat：

| 文件 | 内容 |
| --- | --- |
| mapping.csv | img_name,slideID,cls；img_name 相对于类别目录 |
| class_patch_counts.csv | 每类 PNG 总数 |
| slide_patch_counts.csv | 每张 WSI 的八类数量和 total |
| class_patch_counts.png | 类别数量柱状图 |

控制台打印每类数量、唯一 slide 数和每张 slide 的 patch 数分布，据此选择子集规模；不将 slideID 解释为患者编号。

## 第二步：采样与划分

子集通过复制原图生成，原图不剪切、不修改。后续训练只读取生成的子集。采样前复扫八类文件清单并核对 mapping.csv，拒绝使用增图、缺图或编号变化后的过期映射。

### fast_mode：仅快速调试

```powershell
python dataset_sampling.py --mode fast_mode --sample-per-class 200 --dry-run
python dataset_sampling.py --mode fast_mode --sample-per-class 200
```

每类独立随机抽取指定数量，seed=42；数量不足报错，不放回重复采样、不静默减量。输出 FAST 下八类目录，不生成正式 train/val/test。

此模式拆开 WSI 的 patch；后续若按图片随机划分，可能出现 slide 跨集合和染色风格泄露。仅用于加载、显存和超参数流程调试，不能报告正式性能。split_mapping.csv 的 split 为 debug，slide_splits.csv 不代表正式划分。

### paper_mode：正式 WSI 分组实验

```powershell
python dataset_sampling.py --mode paper_mode --select-slide-num 50 --dry-run
python dataset_sampling.py --mode paper_mode --select-slide-num 50
```

先对排序后的全部 slideID 用 seed=42 无放回选择 50 张 WSI，保留它们在八类目录的全部 PNG；再固定种子打乱选中 WSI，按 70%/15%/15% 分配 train/val/test。所有类别共用同一份 WSI 分配。

用最大余数法取整，小子集比例为近似值；50 张对应 35/8/7。至少选择 7 张且不超过实际总数。脚本检查 WSI 不跨集合，并打印各集合 slide 数和八类数量。若某集合缺少类别，复制前报错，请增加 WSI 数，不通过补抽单张 patch 破坏分组。

不保证类别计数均衡。参数控制 WSI 数，不能精确控制 patch 总数；选中 WSI 的全部 PNG 都保留。paper_mode 是本项目的正式方案，不表示论文原文采用完全相同的三集合比例。

## 第三步：检查输出

| 文件 | 内容 |
| --- | --- |
| mapping.csv | 子集全部图片，严格三列 img_name,slideID,cls |
| split_mapping.csv | 映射加 split 列，记录 train/val/test 或 debug |
| slide_splits.csv | 每张选中 WSI 的集合归属 |
| train/val/test/mapping.csv | paper_mode 各集合的三列映射 |

正式模式图片路径为 `PAPER/<split>/<cls>/<img_name>`。img_name 始终相对于类别目录，根 mapping.csv 需结合 split_mapping.csv 定位图片。

复制使用 shutil.copy2，并检查副本字节大小，不解码验证图片内容。已有输出拒绝覆盖。先写入同级 `.building` 目录，全部成功后重命名；中断保留 `.building`，不发布半成品。重跑前请确认后手动移走未完成副本，或在路径配置中设置新的输出位置。同一输入和参数下选择与划分可复现。

## 迁移到 4080Super

1. 本地运行统计，再运行 `--dry-run` 检查计划。
2. 复制项目和 datasets 到目标机器，安装上述三个依赖。
3. 只修改 utils/path_config.py 的 ROOT，其余脚本无需修改。
4. 运行路径预览、统计，再按选定参数采样。新机器建议重建统计缓存。
5. 保留正式 slide_splits.csv；后续训练直接使用既定三个集合，不重新随机拆分。

Linux 示例为 `ROOT = Path(r"/data/Classify")`；也可用脚本的 `--root` 临时覆盖。数据准备仅用 CPU 和磁盘，无需 GPU。

## 论文方法描述

“本研究以全切片图像 WSI（slideID）为划分最小单元，固定随机种子选择子集，保留所选 WSI 的全部已发布 patch，按约 7:1.5:1.5 划分训练、验证和测试集。同一 WSI 的样本仅属于一个集合，避免同一切片跨集合导致的染色风格及局部图像信息泄露。”

这一保证限于 WSI 层面，不能声称消除全部染色偏差或患者级泄露；没有 slide–patient 映射，无法排除同一患者拥有多张不同 WSI。已发布 patch 经过组织采样，不用于推断真实切片的组织面积占比。
