"""快速图片采样，或完整 WSI 子集采样与互斥划分；不读取临床表。"""
import argparse
import random
import shutil
import pandas as pd
from tqdm import tqdm
from utils import path_config as config
from dataset_stat import read_mapping, scan_dataset, save_csv

# ==================== CONFIG ====================
CONFIG = config.paths()  # 路径统一在 utils/path_config.py 修改
SEED = config.SEED
CLASSES = config.CLASSES
SPLITS = ("train", "val", "test")
# =================================================


def split_slides(slides):
    """最大余数法按 70/15/15 取整；少于七张不做正式划分。"""
    if len(slides) < 7:
        raise ValueError("paper_mode 至少选择 7 张 WSI，保证三个集合非空")
    shuffled = sorted(slides)
    random.Random(SEED).shuffle(shuffled)
    weights = (70, 15, 15)
    sizes = [len(slides) * w // 100 for w in weights]
    order = sorted(range(3), key=lambda i: (-(len(slides) * weights[i] % 100), i))
    for i in order[:len(slides) - sum(sizes)]:
        sizes[i] += 1
    result, start = {}, 0
    for split, size in zip(SPLITS, sizes):
        for slide in shuffled[start:start + size]:
            result[slide] = split
        start += size
    return result


def build_plan(mapping, mode, sample_per_class, select_slide_num):
    slides = sorted(mapping["slideID"].unique())
    if mode == "fast_mode":
        if sample_per_class < 1:
            raise ValueError("sample_per_class 必须为正整数")
        chunks = []
        rng = random.Random(SEED)
        for cls in CLASSES:
            group = mapping.loc[mapping["cls"].eq(cls)].sort_values("img_name")
            if len(group) < sample_per_class:
                raise ValueError(f"{cls} 只有 {len(group)} 张，少于要求的 {sample_per_class} 张")
            indices = rng.sample(range(len(group)), sample_per_class)
            chunks.append(group.iloc[sorted(indices)])
        selected = pd.concat(chunks, ignore_index=True)
        selected["split"] = "debug"
        return selected
    if not 7 <= select_slide_num <= len(slides):
        raise ValueError(f"select_slide_num 必须在 7 至 {len(slides)} 之间")
    chosen = random.Random(SEED).sample(slides, select_slide_num)
    assignments = split_slides(chosen)
    selected = mapping.loc[mapping["slideID"].isin(chosen)].copy()
    selected["split"] = selected["slideID"].map(assignments)
    if selected.groupby("slideID")["split"].nunique().ne(1).any():
        raise AssertionError("发现 WSI 跨集合")
    return selected.reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="临时覆盖根目录")
    parser.add_argument("--mode", required=True, choices=("fast_mode", "paper_mode"))
    parser.add_argument("--sample-per-class", "--sample_per_class", type=int, default=500)
    parser.add_argument("--select-slide-num", "--select_slide_num", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="验证并打印计划，不复制")
    args = parser.parse_args(argv)
    p = config.paths(args.root)
    mapping = read_mapping(p["stat"] / "mapping.csv")
    print("核对原图与映射表，确保所选 WSI 的全部 PNG 都在映射中……")
    current = scan_dataset(p["raw"]).sort_values(["cls", "img_name"]).reset_index(drop=True)
    if not current.equals(mapping):
        raise ValueError("原图与 mapping.csv 不一致，请重新运行 dataset_stat.py")
    plan = build_plan(mapping, args.mode, args.sample_per_class, args.select_slide_num)
    labels = SPLITS if args.mode == "paper_mode" else ("debug",)
    print(f"模式：{args.mode}；seed={SEED}；选中 slide 数：{plan['slideID'].nunique()}")
    for split in labels:
        part = plan.loc[plan["split"].eq(split)]
        print(f"\n{split}: {part['slideID'].nunique()} slides / {len(part):,} patches")
        print(part.groupby("cls").size().reindex(CLASSES, fill_value=0).to_string())
    if args.mode == "fast_mode":
        print("警告：按图片采样，仅用于调试；不保证完整 WSI，不提供正式实验划分。")
    else:
        coverage = pd.crosstab(plan["split"], plan["cls"]).reindex(
            index=SPLITS, columns=CLASSES, fill_value=0)
        if coverage.eq(0).any().any():
            raise ValueError("某集合缺少组织类别，请增加 select_slide_num；尚未复制图片")
    if args.dry_run:
        print("dry-run 完成，未复制图片。")
        return
    target = p["paper"] if args.mode == "paper_mode" else p["fast"]
    staging = target.with_name(target.name + ".building")
    if target.exists() or staging.exists():
        raise FileExistsError(f"目标或未完成副本已存在，拒绝覆盖：{target} / {staging}")
    for split in labels:
        base = staging / split if args.mode == "paper_mode" else staging
        for cls in CLASSES:
            (base / cls).mkdir(parents=True, exist_ok=True)
    # 中断保留 .building，不将不完整副本发布为最终数据集。
    for row in tqdm(plan.itertuples(index=False), total=len(plan), desc="复制 PNG", unit="patch"):
        source = p["raw"] / row.cls / row.img_name
        base = staging / row.split if args.mode == "paper_mode" else staging
        destination = base / row.cls / row.img_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.stat().st_size != source.stat().st_size:
            raise IOError(f"副本大小不一致：{destination}")
    columns = ["img_name", "slideID", "cls"]
    save_csv(plan[columns], staging / "mapping.csv")
    save_csv(plan, staging / "split_mapping.csv")
    save_csv(plan[["slideID", "split"]].drop_duplicates().sort_values("slideID"),
             staging / "slide_splits.csv")
    for split in labels:
        if args.mode == "paper_mode":
            save_csv(plan.loc[plan["split"].eq(split), columns], staging / split / "mapping.csv")
    staging.rename(target)
    print(f"\n生成完成：{target}；原图未修改。")


if __name__ == "__main__":
    main()
