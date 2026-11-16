"""直接复用旧映射生成正式 WSI 子集；仅使用标准库，不读取临床表。"""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import random
import shutil
import sys
from utils import path_config as config

# ==================== CONFIG ====================
CONFIG = config.paths()
SEED = config.SEED
CLASSES = config.CLASSES
SPLITS = ("train", "val", "test")
# =================================================


def read_legacy(source):
    rows, seen = [], set()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == ["img_name", "patient_id", "cls"]:
            id_column = "patient_id"
        elif reader.fieldnames == ["img_name", "slideID", "cls"]:
            id_column = "slideID"
        else:
            raise ValueError("映射必须为 img_name,patient_id,cls 或 img_name,slideID,cls 三列")
        for line, record in enumerate(reader, 2):
            if None in record or any(value is None for value in record.values()):
                raise ValueError(f"CSV 第 {line} 行列数错误")
            name = record["img_name"].replace("\\", "/")
            slide, cls = record[id_column], record["cls"]
            parts = name.split("/")
            if (cls not in CLASSES or any(p in ("", ".", "..") for p in parts)
                    or ":" in name or not name.lower().endswith(".png")
                    or config.parse_slide_id(parts[-1], cls) != slide):
                raise ValueError(f"CSV 第 {line} 行编号、类别或路径异常：{record}")
            key = (cls, name)
            if key in seen:
                raise ValueError(f"CSV 图片重复：{key}")
            seen.add(key)
            rows.append((name, slide, cls))
            if len(rows) % 100000 == 0:
                print(f"已核对 CSV：{len(rows):,} 行", flush=True)
    if {row[2] for row in rows} != set(CLASSES):
        raise ValueError("映射为空或未包含全部八类")
    return sorted(rows, key=lambda r: (r[2], r[0])), seen


def verify_files(raw, expected):
    """只核对文件清单，不读取像素；防止旧表漏掉所选 WSI 的新增图片。"""
    found = set()
    for cls in CLASSES:
        folder = raw / cls
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"类别目录不存在或是符号链接：{folder}")
        count = 0
        for current, dirs, files in os.walk(folder, followlinks=False):
            if any((Path(current) / name).is_symlink() for name in dirs):
                raise ValueError(f"类别目录内有符号链接目录：{current}")
            for name in files:
                if not name.lower().endswith(".png"):
                    continue
                image = Path(current) / name
                if image.is_symlink() or not image.is_file():
                    raise ValueError(f"异常图片或符号链接：{image}")
                key = (cls, image.relative_to(folder).as_posix())
                if key not in expected:
                    raise ValueError(f"原图中存在旧映射未记录的 PNG：{key}")
                found.add(key)
                count += 1
        print(f"核对 {cls} 文件清单完成：{count:,} 张", flush=True)
    if found != expected:
        raise ValueError(f"旧映射有 {len(expected - found):,} 张 PNG 在原图中不存在")


def make_plan(rows, number):
    pool = sorted({row[1] for row in rows})
    if not 7 <= number <= len(pool):
        raise ValueError(f"选择数量应在 7 至 {len(pool)} 之间")
    selected = sorted(random.Random(SEED).sample(pool, number))
    random.Random(SEED).shuffle(selected)
    weights = (70, 15, 15)
    sizes = [number * w // 100 for w in weights]
    order = sorted(range(3), key=lambda i: (-(number * weights[i] % 100), i))
    for i in order[:number - sum(sizes)]:
        sizes[i] += 1
    assignments, start = {}, 0
    for split, size in zip(SPLITS, sizes):
        for slide in selected[start:start + size]:
            assignments[slide] = split
        start += size
    plan = [(*row, assignments[row[1]]) for row in rows if row[1] in assignments]
    counts = Counter((row[3], row[2]) for row in plan)
    print(f"全量：{len(rows):,} patches / {len(pool)} slides；选中：{number} slides；seed={SEED}")
    for split in SPLITS:
        print(f"{split}: {sum(value == split for value in assignments.values())} slides；"
              + ", ".join(f"{cls}={counts[split, cls]}" for cls in CLASSES))
    if any(counts[split, cls] == 0 for split in SPLITS for cls in CLASSES):
        raise ValueError("某集合缺少组织类别，请增加选择 WSI 数；尚未复制")
    return plan, assignments


def write_csv(target, header, rows):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(target)


def main(argv=None):
    # Windows GBK 终端可能无法输出数据目录里的 U+2011 连字符。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="临时覆盖根目录")
    parser.add_argument("--map", default="output_legacy/stat/patch_patient_map.csv")
    parser.add_argument("--select-slide-num", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="仅验证和打印，不写文件或复制")
    args = parser.parse_args(argv)
    p = config.paths(args.root)
    source = Path(args.map)
    if not source.is_absolute():
        source = p["root"] / source
    print(f"读取映射：{source}（不加载 pandas/NumPy）", flush=True)
    rows, expected = read_legacy(source)
    verify_files(p["raw"], expected)
    plan, assignments = make_plan(rows, args.select_slide_num)
    if args.dry_run:
        print("预览成功：没有写入文件或复制图片。", flush=True)
        return
    target = p["paper"]
    staging = target.with_name(target.name + ".building")
    if target.exists() or staging.exists():
        raise FileExistsError(f"拒绝覆盖已有目标或半成品：{target} / {staging}")
    for split in SPLITS:
        for cls in CLASSES:
            (staging / split / cls).mkdir(parents=True, exist_ok=True)
    for index, (name, slide, cls, split) in enumerate(plan, 1):
        original = p["raw"] / cls / name
        destination = staging / split / cls / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        if original.stat().st_size != destination.stat().st_size:
            raise IOError(f"副本大小异常：{destination}")
        if index % 1000 == 0:
            print(f"已复制 {index:,}/{len(plan):,}", flush=True)
    columns = ["img_name", "slideID", "cls"]
    write_csv(staging / "mapping.csv", columns, (row[:3] for row in plan))
    write_csv(staging / "split_mapping.csv", columns + ["split"], plan)
    write_csv(staging / "slide_splits.csv", ["slideID", "split"], sorted(assignments.items()))
    for split in SPLITS:
        write_csv(staging / split / "mapping.csv", columns,
                  (row[:3] for row in plan if row[3] == split))
    staging.rename(target)
    print(f"完成：{target}；原始数据及旧统计未修改。", flush=True)


if __name__ == "__main__":
    main()
