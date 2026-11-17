"""直接复用旧映射生成正式 WSI 子集；仅使用标准库，不读取临床表。"""
import argparse
import csv
import hashlib
import json
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


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_checkpoint(source, raw, directory, recheck=False):
    """缓存完整 CSV 校验和逐类别文件清单校验；源表变化自动失效。"""
    identity = {"version": 1, "source": str(source.resolve()),
                "sha256": file_hash(source), "raw": str(raw.resolve()),
                "parser": file_hash(Path(config.__file__)), "classes": list(CLASSES)}
    state_path = directory / "state.json"
    rows_path = directory / "validated_rows.json"
    state = {}
    if state_path.exists():
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
    if (state.get("identity") == identity and rows_path.exists()
            and state.get("rows_sha256") == file_hash(rows_path)):
        print("恢复断点：旧 CSV 校验已完成，读取已验证映射。", flush=True)
        with rows_path.open(encoding="utf-8") as handle:
            rows = [tuple(row) for row in json.load(handle)]
        expected = {(row[2], row[0]) for row in rows}
    else:
        rows, expected = read_legacy(source)
        save_json(rows_path, rows)
        state = {"identity": identity, "rows_sha256": file_hash(rows_path), "verified": {}}
        save_json(state_path, state)
        print("CSV 校验断点已保存。", flush=True)
    if recheck:
        state["verified"] = {}
        save_json(state_path, state)
    return rows, expected, state, state_path


def verify_files(raw, expected, state=None, state_path=None):
    """只核对文件清单，不读取像素；防止旧表漏掉所选 WSI 的新增图片。"""
    if state is None:
        state = {"verified": {}}
    for cls in CLASSES:
        if cls in state["verified"]:
            print(f"恢复断点：{cls} 已核对 {state['verified'][cls]:,} 张，跳过。", flush=True)
            continue
        wanted = {key for key in expected if key[0] == cls}
        found = set()
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
                if key not in wanted:
                    raise ValueError(f"原图中存在旧映射未记录的 PNG：{key}")
                found.add(key)
                count += 1
        if found != wanted:
            raise ValueError(f"{cls} 有 {len(wanted - found):,} 张 PNG 在原图中不存在")
        state["verified"][cls] = count
        if state_path is not None:
            save_json(state_path, state)
        print(f"核对 {cls} 文件清单完成：{count:,} 张，断点已保存。", flush=True)


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
    counts = {(split, cls): 0 for split in SPLITS for cls in CLASSES}
    for row in plan:
        counts[row[3], row[2]] += 1
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
    parser.add_argument("--dry-run", action="store_true", help="仅验证和打印，保存断点但不复制图片")
    parser.add_argument("--recheck-files", action="store_true", help="原图发生变化时，重新核对八类文件清单")
    args = parser.parse_args(argv)
    p = config.paths(args.root)
    source = Path(args.map)
    if not source.is_absolute():
        source = p["root"] / source
    print(f"读取映射：{source}（不加载 pandas/NumPy）", flush=True)
    checkpoint = p["root"] / "output" / "wsi_subset_checkpoint"
    rows, expected, state, state_path = load_checkpoint(
        source, p["raw"], checkpoint, args.recheck_files)
    print("恢复期间请勿增删或修改原图；变化后必须加 --recheck-files。", flush=True)
    verify_files(p["raw"], expected, state, state_path)
    plan, assignments = make_plan(rows, args.select_slide_num)
    if args.dry_run:
        print("预览成功：断点已保存，没有复制图片。", flush=True)
        return
    target = p["paper"]
    staging = target.with_name(target.name + ".building")
    if target.exists():
        raise FileExistsError(f"拒绝覆盖已有最终数据集：{target}")
    owner = staging / "resume_identity.json"
    copy_identity = {"source": state["identity"], "seed": SEED,
                     "plan_sha256": hashlib.sha256(json.dumps(plan).encode("utf-8")).hexdigest()}
    if staging.exists():
        if not owner.exists():
            raise ValueError(f"半成品缺少续跑标识，不能自动复用：{staging}")
        with owner.open(encoding="utf-8") as handle:
            if json.load(handle) != copy_identity:
                raise ValueError("半成品属于其他参数或数据，拒绝复用")
        print("恢复复制：逐文件校验已有副本 SHA256，匹配则跳过。", flush=True)
    else:
        save_json(owner, copy_identity)
    for split in SPLITS:
        for cls in CLASSES:
            (staging / split / cls).mkdir(parents=True, exist_ok=True)
    for index, (name, slide, cls, split) in enumerate(plan, 1):
        original = p["raw"] / cls / name
        destination = staging / split / cls / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_hash = file_hash(original)
        if not destination.exists() or file_hash(destination) != original_hash:
            temporary = destination.with_name(destination.name + ".copying")
            shutil.copy2(original, temporary)
            if file_hash(temporary) != original_hash:
                raise IOError(f"副本 SHA256 异常：{temporary}")
            temporary.replace(destination)
        if index % 1000 == 0:
            print(f"已完成或复用 {index:,}/{len(plan):,}", flush=True)
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
