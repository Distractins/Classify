# HMU-CRC-Hist550K：WSI 分组的 Patch 八分类

仅做八类组织分类，不做患者 MSI 预测，不读取临床 Excel。slideID 为文件名首段，不代表患者编号；同一 WSI 不跨 train/val/test。WSI 隔离不能替代患者级隔离。

<AI_META>
task: patch eight-class classification only
checkpoint_class_order: [ADI, LYM, DEB, MUC, MUS, NORM, STR, TUM]
split_unit: slideID as string
seed: 42
clinical_excel: unused
selection: validation macro-F1; test only after model selection
entry: src/stage1_train_8cls/train_wsi_baseline.py
root_argument: PAPER dataset directory, NOT project directory
output: outputs/experiments/MODEL/RUN_NAME
</AI_META>

## 保留的目录与历史结果

```text
datasets/HMU-CRC-Hist550K-PAPER/
  train/ADI ... TUM/
  val/ADI ... TUM/
  test/ADI ... TUM/
  mapping.csv / split_mapping.csv / slide_splits.csv
pretrained/                  # 输入：下载的预训练权重
src/stage1_train_8cls/train_wsi_baseline.py
src/experiment_reports.py
utils/path_config.py
output/stat/                 # 数据准备结果
outputs/
  experiments/
    efficientnet_b0/legacy_baseline/   # 已完成的服务器基线
    efficientnet_b0/baseline/          # 新实验
    resnet50/baseline/
    resnet101/baseline/
    uni/baseline/
  archive/                   # 早期权重、旧报告和退役代码
```

`output`（单数）只放数据准备结果；`outputs/experiments` 放各个训练实验。原 `output/weights` 和 `output/figures` 已迁到 `outputs/experiments/efficientnet_b0/legacy_baseline/weights` 与 `reports`。原 `outputs/weights/stage1_8cls/efficientnet_wsi` 单独归档，不混入已完成基线。

`output_legacy` 已清理：完整图片映射保留为 `output/stat/legacy_patch_map.csv`，类别数量表/图也保留。小报告、退役代码备份在 outputs/archive；旧扫描 SQLite、崩溃日志、重复扫描缓存已移除。原图、已划分子集及预训练/训练权重没有删除。当前本地 datasets 未发现图片，需要同步服务器 PAPER 数据集才能本地训练。

用户报告 EfficientNet-B0 约 0.78，需要区分验证 accuracy、验证 macro-F1 和最终 test 指标。本地旧日志只有 11 轮，不能据此宣称训练已完成 20 轮。旧权重类别顺序是 ADI、LYM、DEB、MUC、MUS、NORM、STR、TUM，新脚本保留此顺序，不能按字母排序解释旧权重。

## 环境

先保留已工作的 torch/torchvision CUDA 组合，不要为了安装其他依赖覆盖它们：

```bat
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -m pip install numpy pillow tqdm matplotlib
```

仅统计工具需 pandas；UNI 额外安装 `python -m pip install "timm>=1.0,<2"`。XGBoost、openpyxl、transformers 不是当前训练必需。项目默认随代码目录迁移，路径在 utils/path_config.py 集中配置，优先 ASCII 目录名，兼容已有 U+2011 数据目录。

## 训练模型对比

`--root` 保持服务器已修改版本的语义：指向含 train/val/test 的 PAPER 数据集。`--project-root` 才是项目目录，默认由代码位置推断。相对数据路径与预训练路径相对于项目目录。

先试跑训练/验证各两批，不保存正式权重：

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-PAPER" --model efficientnet_b0 --batch-size 8 --smoke-test
```

正式训练：

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-PAPER" --model efficientnet_b0 --run-name baseline --epochs 20 --batch-size 8
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-PAPER" --model resnet50 --run-name baseline --epochs 20 --batch-size 8
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-PAPER" --model resnet101 --run-name baseline --epochs 20 --batch-size 8
```

UNI 显存要求较高，先试跑：

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-PAPER" --model uni --batch-size 2 --smoke-test
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-PAPER" --model uni --run-name baseline --epochs 20 --batch-size 2
```

| model | 本地预训练文件 |
| --- | --- |
| efficientnet_b0 | pretrained/efficientnet_b0_rwightman-3dd342df.pth |
| resnet50 | pretrained/resnet50-0676ba61.pth |
| resnet101 | pretrained/resnet101-63fe2227.pth |
| uni | pretrained/UNI/pytorch_model.bin |

不自动下载权重，严格检查结构匹配。`--pretrained` 可指定其他本地输入文件。CTransPath.pth 和其他 ViT 文件保留，但未适配其架构，不能直接作为普通 ViT 权重加载。UNI 与 CNN 都做全参数微调，使用相同划分和预处理；比较时记录预训练来源、batch size 和训练预算差异。

## 自动保存和断点恢复

### 冻结 UNI：Linear、MLP 和 AETHER 对照

现有 `--model uni` 默认为 `--uni-mode finetune`，行为不变；CNN 的训练与旧断点保持兼容。新增 `--uni-mode linear/mlp/aether`：UNI 参数 requires_grad=False，始终 eval，forward_features 在 no_grad 内执行。优化器仅更新分类头，以及 AETHER 模式的增强模块。仍需完整加载本地 UNI .bin；训练中不缓存特征。

三种冻结模式均移除 CLS/其他前缀 token，对空间 tokens 做均值聚合，避免基线使用 CLS 而方法使用空间 tokens 的混淆。AETHER 在聚合前增强 token；Linear 是 1024→8，MLP 默认 1024→256→GELU→Dropout(0.1)→8。AETHER 分类头仍为 Linear。AETHER 的 hard_cls_idx 保留原模块默认 (2,6,4)，在当前类别顺序下对应 DEB、STR、MUS。sim_matrix 暂不加入额外监督；不跨 patch 或 WSI 聚合。

AETHER 总损失为 CE + `--ortho-weight` × ortho_reg（默认 0.01，经验起点，不代表最优值）；Linear/MLP 是 CE。AETHER 的 loss 曲线为含正则的总损失，不能与 CE-only 模式当作同一损失直接比较；accuracy/F1/AUC 计算一致。

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL" --model uni --uni-mode linear --run-name frozen_5ep --epochs 5 --batch-size 2 --lr 0.0001 --weight-decay 0.00001 --save-every-batches 100
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL" --model uni --uni-mode mlp --run-name frozen_5ep --epochs 5 --batch-size 2 --lr 0.0001 --weight-decay 0.00001 --save-every-batches 100
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL" --model uni --uni-mode aether --run-name frozen_5ep --epochs 5 --batch-size 2 --lr 0.0001 --weight-decay 0.00001 --ortho-weight 0.01 --save-every-batches 100
```

先在对应命令末尾加 --smoke-test 试跑；正式命令移除该项。恢复加 --resume；最终评估加 --evaluate-test（不要同时加 resume）。输出分别为 `outputs/experiments/uni_frozen_linear/frozen_5ep/`、`uni_frozen_mlp/frozen_5ep/`、`uni_frozen_aether/frozen_5ep/`。头配置及模式写入断点，模式不能相互续训；hidden_dim/dropout/ortho_weight 也需与原断点一致。现有 UNI 全参数微调结果和 CNN 结果无需重跑。

冻结模式断点仍包含完整 UNI，便于独立恢复且不依赖后来被替换的预训练文件；因此冻结后断点仍较大，保存耗时并不会按可训练参数量同比减少。可调大 save-every-batches 减少磁盘开销。小模型测试可验证冻结及梯度逻辑，真实 UNI 的显存/速度和完整训练需在服务器试跑确认。

每个实验自动生成独立目录，已有实验不覆盖：

```text
outputs/experiments/<model>/<run-name>/
  config.json
  weights/best.pth         # val macro-F1 最佳的完整断点
  weights/last.pth         # 最近训练批次或完整轮次的恢复断点
  reports/history.csv
  reports/loss_curve.png
  reports/accuracy_curve.png
  reports/macro_f1_curve.png
  reports/best_val_metrics.json
  reports/best_val_per_class.csv
  reports/best_val_confusion_matrix.csv
  reports/best_val_confusion_matrix.png
```

默认每 200 个训练 batch 原子保存 last.pth（`--save-every-batches 200`），训练轮次开始前和最后一批后也保存。记录模型、优化器、下一批位置、该轮累计 loss/混淆矩阵及随机状态。固定每轮 shuffle 顺序，恢复直接读取剩余图片，不重新训练已保存的批次。每轮验证完成后保存整轮历史并更新曲线；best.pth 仍只按验证 macro-F1 选择。恢复例子：

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-PAPER" --model resnet50 --run-name baseline --epochs 20 --batch-size 8 --resume
```

批次恢复需 workers=0（默认）；多进程预取的随机增强状态没有保存，所以 workers>0 时必须用 `--save-every-batches 0`，仅整轮保存。epochs 为目标总轮数。model、run-name、batch-size、lr、weight-decay、workers 必须与断点一致，保存间隔可改。一般训练中断最多重做当前保存间隔内未提交的批次；验证中断会重新验证，但训练已保存完成则无需重训。旧整轮断点仍可读取，但此前没有保存的批次不能追溯恢复。磁盘需容纳原文件和临时断点两份，保存频率越高磁盘开销越大。非法内存访问或硬件问题仍可能破坏计算，断点不能修复其根因。

保存类别和文件清单指纹（路径、标签、大小；不是逐图内容校验）。默认 workers=0，跨硬件不保证数值完全相同。读图失败会明确报错，不再用黑图替代。

旧服务器 checkpoint 缺少完整配置及随机状态，且 best_f1 保存顺序落后一轮。新脚本不直接续训旧断点，保留旧脚本备份在 outputs/archive/retired_patient_msi_code.zip。新实验从预训练权重开始，按新格式续训；旧训练结果原样保留。

旧日志的 loss/accuracy/F1 曲线已根据现存 11 轮历史重建到 legacy_baseline/reports，不能补回未记录的轮次。

## 最终测试

### 无法修复的验证集坏图

原图也损坏而无法恢复时，可显式加 `--exclude-val-image "val/TUM/1432083_10080_4480_TUM.png"`，只允许排除实际无法解码的验证图片，不替换黑图、不删除原文件、不改训练样本或 WSI 归属。该选项可重复；所有后续模型对比需使用相同清单，报告验证样本数量变化。排除名单写入配置及断点，恢复/测试需传相同选项。原始文件清单指纹仍核对全部文件，不会因此绕过缺图或大小变化校验。

已有无排除清单的断点仅允许通过 resume 显式迁移到新验证清单；保留历史指标并标记 val_excluded_count，重置最佳验证分数。旧 best.pth 如存在则备份为 best_before_val_exclusion.pth；新最佳权重在下一次完整验证后产生，不能拿旧验证分数与新分数直接比较。已完成所有目标轮次时需另行规划验证或延长训练，不自动将旧 best 宣称为新验证集选出的最佳模型。

调参只用 val，设置确定后评估 test：

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-PAPER" --model resnet50 --run-name baseline --batch-size 8 --evaluate-test
```

输出 test_metrics.json、test_per_class.csv、test_confusion_matrix.csv/png。指标为八分类 loss、accuracy、macro-F1，不是 MSI 指标。不要根据 test 结果反复调参。

### 完整评估产物（最佳验证集和最终测试集）

图片标题及说明使用中文，Precision、Recall、F1、AUC、ROC 等指标保留英文。Windows 优先用微软雅黑/黑体；Linux 推荐安装 Noto CJK（Ubuntu/Debian 可用 `sudo apt-get install fonts-noto-cjk`），然后重启绘图进程。无管理员权限时，设置环境变量 `HMU_FONT_PATH` 指向已有中文字体的 .ttf/.ttc/.otf 文件。没有可用中文字体时会提示，避免把缺字方框误认为图片内容问题。

每次验证 macro-F1 改善时输出 `best_val_*`；单独执行 --evaluate-test 时输出 `test_*`，保存在各模型/实验的 reports 目录。原来已有每类 Precision/Recall/F1 CSV，现在补全柱状图与概率评估。

| 文件后缀 | 内容 |
| --- | --- |
| _metrics.json | loss、accuracy、macro-P/R/F1、weighted-F1、balanced accuracy、Cohen's κ、Mean-AUC、micro-AUC、mean Average Precision |
| _per_class.csv | 每类 Precision、Recall、F1、support |
| _precision_recall_f1.png | 八类 P/R/F1 分组柱状图 |
| _auc_per_class.csv | 每类 OvR AUC、Average Precision、是否可计算 |
| _roc_ovr.png | 八类 OvR ROC、插值宏平均 ROC、micro ROC，标题标注 Mean-AUC |
| _confusion_matrix.csv | 原始错分计数 |
| _confusion_matrix.png | 按真实类别行归一化的混淆矩阵 |
| _confusion_matrix_counts.png | 原始计数混淆矩阵 |
| _confused_pairs.csv | 按数量排序的定向错分对，含占真实类比例，例如 STR→MUS 与 MUS→STR |
| _predictions.csv | 图片相对路径、真实类别、预测类别、八类 softmax 概率，方便复核与重绘 |

Mean-AUC 是八个 OvR AUC 的等权算术平均，不是 accuracy，也不是把硬预测标签当概率；micro-AUC 为展开 one-hot 标签与概率后的汇总 AUC。ROC 图中的宏平均曲线通过插值绘制，其曲线下面积不作为另一个 Mean-AUC。每类没有正例或负例时 AUC 不可定义，CSV 留空、JSON 使用 null，完整八类 Mean-AUC 也设为 null，不用 0 或 1 代替。balanced accuracy 为有样本类别 Recall 均值，weighted-F1 按 support 加权。Precision/Recall/F1 的零分母按 0 处理。

Average Precision 是 PR 曲线的汇总指标，对类别不平衡提供补充；它不是 ROC-AUC，也不等于梯形积分的 PR-AUC。本入口暂不绘制 PR 曲线。概率保存在 CPU，仅验证/测试收集，不参与反向传播。

新增 AUC 需要 scikit-learn：`python -m pip install scikit-learn`。已有 best 权重无需重训，执行 --evaluate-test 即可生成这些测试结果；原始旧服务器 best.pth 使用下方 --legacy-weights 命令。已经生成的旧混淆矩阵无法还原 ROC，必须重新推理获得概率。选择最佳模型仍只用 val macro-F1；测试指标不反馈到训练。逐轮 history 保留 train/val loss、accuracy、macro-F1，新增完整指标写入最佳验证/测试报告。

这些是 patch 级汇总指标，patch 来自同一 WSI，不能将 patch 当作独立患者声称统计显著。正式模型比较应使用相同 WSI 划分，尽可能补充多随机种子实验；置信区间若后续添加，应按 WSI 聚类重采样，不按单张 patch 随机 bootstrap。

旧服务器 EfficientNet 原始 best.pth（不是更早的 efficientnet_wsi 权重）可单独评估：

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-PAPER" --model efficientnet_b0 --run-name legacy_baseline --evaluate-test --legacy-weights "outputs/experiments/efficientnet_b0/legacy_baseline/weights/best.pth" --batch-size 8
```

## 数据准备与可删除内容

### 每张 WSI、每类限额 50 张：中等规模数据集

新入口 `wsi_capped_sampling.py` 沿用已生成 FULL 中的 train/val/test 归属，不重新划分 WSI；在每个 WSI 自己的每个类别中随机保留最多 50 张 PNG，不足则全部保留。seed=42，各组使用稳定的独立种子。所有原 WSI 必须至少保留一张，且三集合 WSI 交集为空；未选中 patch 不参与本次实验，不放进其他集合。此步骤明确替代旧的“所选 WSI 全部 patch 必须保留”要求，仍保证 WSI 隔离，不保证未知患者身份的隔离。

```bat
python wsi_capped_sampling.py --max-per-wsi-class 50 --exclude-image "val/TUM/1432083_10080_4480_TUM.png" --dry-run
python wsi_capped_sampling.py --max-per-wsi-class 50 --exclude-image "val/TUM/1432083_10080_4480_TUM.png"
```

默认源目录为 datasets/HMU-CRC-Hist550K-FULL，输出独立 FULL-CAP50，可通过 --source/--output 指定路径。只用标准库，不读取临床表，不解码图片；exclude-image 为用户已确认的坏图路径，必须在源目录存在，记录到 excluded_images.csv。其他未识别的坏图仍可能需要另行检查，不会静默用黑图替代。

train/val/test 都采样，500 个 WSI 的总图片上限为 500×8×50=200,000 张，训练上限 140,000 张。实际更少；每类数量、集合图片数和 WSI 数会打印。验证和测试样本发生变化，不直接与旧 FULL/PAPER 测试成绩混同；所有后续对比模型应共用 CAP50。原 FULL 及其映射/图片不修改。

中断后重复同一正式命令，保留 FULL-CAP50.building；脚本重新枚举并生成相同计划，通过源目录、参数和计划标识检查后，对已有副本计算 SHA256，一致则跳过，否则重复制。每张图通过临时文件原子发布；写完根映射、各集合映射、WSI 归属与排除清单后，才改为最终目录。输入不变、参数不变才能续跑，已有最终输出拒绝覆盖。无法修复机器非法内存访问等底层异常。

新数据集必须新建训练实验，不能用 FULL 的 last.pth 直接续训：

采样计划现在保存到 `output/capped_sampling_checkpoint/`，在打印统计和复制之前原子保存。中断后同一命令可复用完整计划，并继续校验/复制已有副本。首次生成计划尚未保存时仍需重新扫描；dry-run 也会写入计划缓存，但不复制图片。缓存按源路径、限额、种子和排除清单区分，并校验缓存 SHA256。恢复时必须保持源数据不变；若增删或修改源图，加 `--rebuild-plan` 重新生成，已有 .building 若计划不同会拒绝混用。缓存不是实时源目录清单，不能自动发现新增或删除图片。断点不能修复非法指令、内存访问异常或运行时损坏。

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL-CAP50" --model efficientnet_b0 --run-name cap50_5ep --epochs 5 --batch-size 8 --lr 0.0001 --weight-decay 0.00001 --save-every-batches 100 --smoke-test
```

试跑后去掉 smoke-test 正式训练，续训加 resume。已排除坏图不在新数据集中，因此新实验不要再带旧的 exclude-val-image 参数。

### 全量数据的独立实验

优先复用完整映射：脚本依次查找 output/stat/legacy_patch_map.csv、output_legacy/stat/patch_patient_map.csv、output/stat/patch_patient_map.csv、output/stat/mapping.csv。都不存在时，自动从原始八类 PNG 文件名生成 output/stat/mapping.csv，不需要 pandas、不读取临床表、不解码图片。显式 --map 指定的文件缺失则报错，不自动改用其他文件。原始八类目录必须仍在 datasets/HMU-CRC-Hist550K，脚本会核对映射与实际文件清单，不能用 PAPER 子集映射冒充全集。首次生成映射中断需重新枚举，成功保存后可继续使用后续校验/复制断点。先预览，再复制全部 WSI 并固定划分：

```bat
python -X faulthandler -u wsi_subset_stdlib.py --all-slides --recheck-files --dry-run
python -X faulthandler -u wsi_subset_stdlib.py --all-slides
```

500 个 WSI 时对应 train/val/test=350/75/75；不是每类图片各自随机划分。输出 datasets/HMU-CRC-Hist550K-FULL，保留原 PAPER 子集和结果。该操作复制全部约 547,971 张 PNG，需要额外容纳一份全量数据的磁盘空间。断点保存、SHA256 校验及恢复规则同子集入口；中断重复第二条命令，不删除 FULL.building。--all-slides 优先于 --select-slide-num；可用 --output-name 指定另一个独立目录。

建议先保持当前优化设置仅增大数据规模，观察 5 轮的收敛情况：

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL" --model efficientnet_b0 --run-name full_adamw_lr1e4_5ep --epochs 5 --batch-size 8 --lr 0.0001 --weight-decay 0.00001 --smoke-test
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL" --model efficientnet_b0 --run-name full_adamw_lr1e4_5ep --epochs 5 --batch-size 8 --lr 0.0001 --weight-decay 0.00001
```

用户提供的论文 CNN 参数可另开实验，不覆盖上面的控制组：

```bat
python src/stage1_train_8cls/train_wsi_baseline.py --root "datasets/HMU-CRC-Hist550K-FULL" --model efficientnet_b0 --run-name full_paper_cnn_5ep --epochs 5 --batch-size 8 --lr 0.01 --weight-decay 0.0001
```

ResNet50/101 可替换 model，所有主干比较使用同一 FULL 划分。AdamW、lr=0.01、weight_decay=1e-4、5 epochs 仅匹配用户提供的部分设置，不代表完整论文复现；ImageNet 权重版本、batch size、划分、预处理和调度器均需核对。更高学习率不保证更好，应根据 val loss/F1 判断。

UNI 若实验采用 lr=3e-4、weight_decay=0.05、5 epochs，可显式指定，但 UNI 是病理预训练 ViT，不能标注为论文 ImageNet ViT 的直接复现。预训练模型的迁移学习率可不同，需要记录差异。5 轮是训练预算，不保证所有主干收敛；不得只因原文使用 5 轮就断言已经最佳。

续训在原命令末尾加 --resume，必须保持 lr/weight-decay/batch-size/workers 不变。已有旧格式新入口断点未记录 weight_decay 时按先前默认 1e-5 兼容，不可用更改优化器的方式伪装续训。子集与 FULL 的测试 WSI 不同，指标不能用于单独归因数据规模效果；若正式分析数据规模，应先固定同一个测试集，仅在训练池中取不同规模。

本方案仍是 WSI 70/15/15 留出法，不是 patient-level 7:3，也不是 10 折交叉验证。缺少 slide–patient 映射不能声称复现患者级划分。10 折可作为后续独立且计算量约十倍的协议；若做最终测试集加交叉验证，交叉验证只能在 train+val 池内部，test 不参与选参。当前不生成或运行 10 折，避免将不兼容评估协议混在同一实验。

保留 dataset_stat.py、dataset_sampling.py、wsi_subset_stdlib.py，已成功划分的数据训练时不用重建。标准库入口默认使用 output/stat/legacy_patch_map.csv，断点仍在 output/wsi_subset_checkpoint。恢复时保持原图不变，变化后加 --recheck-files。

确认无需重建子集后可删除 output 中的扫描缓存；outputs/archive 另有备份后可手动删除。必须保留正式子集的三个集合及划分 CSV、需要比较的预训练权重，以及实验的 config/history/best/last。不要跨 Windows/Linux 拷贝虚拟环境。
# 染色扰动鲁棒性测试（无需重新训练）

训练结束并根据验证集选定 `weights/best.pth` 后，可以用独立脚本一次完成原始测试、Macenko 染色变化、HED 染色扰动三组评估。训练脚本与原有 `--evaluate-test` 不变；新脚本会重新计算原始测试作为同次运行的参照，因此不要求提前单独测试。仅加载模型参数，不训练、不更新权重，不读取临床文件。

EfficientNet-B0 CAP50 示例（在项目根目录运行）：

```bash
python src/stage1_train_8cls/evaluate_stain_robustness.py --root "datasets/HMU-CRC-Hist550K-FULL-CAP50" --checkpoint "outputs/experiments/efficientnet_b0/cap50_5ep/weights/best.pth" --batch-size 8 --strength 0.1 --seed 42
```

冻结 UNI + Linear 示例：

```bash
python src/stage1_train_8cls/evaluate_stain_robustness.py --root "datasets/HMU-CRC-Hist550K-FULL-CAP50" --checkpoint "outputs/experiments/uni_frozen_linear/cap50_10ep/weights/best.pth" --batch-size 2 --strength 0.1 --seed 42
```

模型架构和 UNI 模式从 checkpoint 自动恢复，不需要额外指定 `--model`、`--uni-mode` 或预训练权重。仅加载本项目可信 checkpoint。使用训练时的完整数据集目录，脚本检查类别顺序、WSI 隔离及路径/标签/大小清单 fingerprint；该 fingerprint 不等同于图片内容哈希。

默认输出到该实验的 `reports/stain_s0.1_seed42/`：

- `original/`、`macenko/`、`hed/`：各组独立的指标 JSON、逐类 Precision/Recall/F1 CSV 和柱状图、混淆矩阵、ROC、逐类 AUC/AP、逐图概率预测及八类示例图片。
- `comparison.csv`、`comparison.png`：三组指标对比；`*_drop_pp` 是原始值减扰动值后乘100，单位为百分点，负值表示扰动组指标更高。
- `protocol.json`：权重 SHA256、数据清单 fingerprint、权重轮次、种子、强度、染色算法细节等记录。

输出目录已存在时拒绝覆盖，可用 `--output "另一个新目录"` 指定新位置。三组使用完全相同的测试样本和标签，固定种子结合图片相对路径生成扰动，所有模型保持相同的 `--strength` 和 `--seed`。图片先按原有评估预处理缩放到224，再施加染色变换，最后执行 ImageNet 输入归一化；源图片不修改。解码失败或 Macenko 无法估计染色基底时会明确终止，不静默跳过样本或使用黑图。

Macenko 条件是自定义的 **Macenko 基底估计 + 染色基底/浓度随机扰动**，不是染色归一化：OD 阈值0.15、SVD平面及角度1/99百分位估计两个基底；基底施加标准差 `strength/5` 的噪声并归一化，浓度分别乘以 `[1-strength, 1+strength]` 的均匀随机系数，保留重建残差。HED 条件用标准 H/E/D 向量进行有符号 OD 分解，仅扰动 H、E 系数，保留第三通道。`strength=0.1` 表示浓度缩放范围0.9～1.1，不表示保证视觉变化恰为10%。

默认强度只是预设的实验起点，并非经本数据集验证的标准强度。应先用训练/验证图片检查形态与颜色是否合理，确定协议后再评估测试集；可预先设置0.05/0.1/0.2强度分别运行，所有模型采用一致协议，不能根据测试得分挑选强度。预览图片仅作展示，不用于测试集调参。单次固定种子的结果不代表统计显著性，不能等同于跨中心泛化证明；真实跨中心测试需要独立外部数据。

