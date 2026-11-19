"""按既有 WSI 划分限额采样：每个 WSI/组织类别最多 N 张，只复制不改原图。"""
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
DEFAULT_CAP = 50
# =================================================


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, header, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def make_plan(source, cap, exclusions=(), report=True):
    """归属来自实际 FULL 子目录；不重新 shuffle 划分 WSI。"""
    if cap < 1:
        raise ValueError("每个 WSI/类别的限额必须为正数")
    excluded = {name.replace("\\", "/") for name in exclusions}
    seen_excluded, assignments, groups = set(), {}, {}
    for split in SPLITS:
        for cls in CLASSES:
            folder = source / split / cls
            if not folder.is_dir() or folder.is_symlink():
                raise ValueError(f"缺少类别目录或存在符号链接：{folder}")
            for current, dirs, files in os.walk(folder, followlinks=False):
                dirs.sort()
                if any(os.path.islink(os.path.join(current, d)) for d in dirs):
                    raise ValueError(f"不接受符号链接目录：{current}")
                for name in sorted(files):
                    if not name.lower().endswith(".png"):
                        continue
                    image = Path(current) / name
                    if image.is_symlink() or not image.is_file():
                        raise ValueError(f"异常图片路径：{image}")
                    slide = config.parse_slide_id(name, cls)
                    if slide in assignments and assignments[slide] != split:
                        raise ValueError(f"源数据存在 WSI 跨集合：{slide}")
                    assignments[slide] = split
                    relative = os.path.relpath(image, folder).replace("\\", "/")
                    key = f"{split}/{cls}/{relative}"
                    if key in excluded:
                        seen_excluded.add(key)
                        continue
                    groups.setdefault((split, slide, cls), []).append((relative, image.stat().st_size))
            print(f"已扫描 {split}/{cls}", flush=True)
    if excluded != seen_excluded:
        raise ValueError(f"排除清单有图片不存在或不是 PNG：{sorted(excluded - seen_excluded)}")
    plan = []
    for (split, slide, cls), candidates in sorted(groups.items()):
        candidates.sort()
        # 每组独立随机种子；不受枚举顺序或其他类别的图片数量影响。
        group_seed = hashlib.sha256(f"{SEED}/{split}/{slide}/{cls}".encode()).digest()
        selected = random.Random(int.from_bytes(group_seed, "big")).sample(candidates, min(cap, len(candidates)))
        for name, size in sorted(selected):
            plan.append((name, slide, cls, split, size))
    selected_assignments = {row[1]: row[3] for row in plan}
    if selected_assignments != assignments:
        raise ValueError("采样会丢失某个 WSI（可能该 WSI 图片被全部排除），拒绝生成")
    if report:
        print_summary(plan, assignments, excluded)
    return plan, assignments


def print_summary(plan, assignments, excluded):
    counts_by_split = {split: dict.fromkeys(CLASSES, 0) for split in SPLITS}
    slides_by_split = dict.fromkeys(SPLITS, 0)
    for row in plan:
        counts_by_split[row[3]][row[2]] += 1
    for split in assignments.values():
        slides_by_split[split] += 1
    for split in SPLITS:
        counts = counts_by_split[split]
        if not all(counts.values()):
            raise ValueError(f"{split} 某组织类别没有采样图片：{counts}")
        print(f"{split}: {slides_by_split[split]} slides / "
              f"{sum(counts.values()):,} patches；{counts}", flush=True)
    print(f"共保留 {len(assignments)} 个 WSI、{len(plan):,} 张图片；排除 {len(excluded)} 张。", flush=True)


def cached_plan(project, source, cap, exclusions, rebuild=False):
    identity = {"source": str(source), "cap": cap, "seed": SEED,
                "excluded": exclusions, "version": 1}
    cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    directory = project / "output/capped_sampling_checkpoint" / cache_id
    plan_path, state_path = directory / "plan.json", directory / "state.json"
    if not rebuild and plan_path.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") == identity and state.get("sha256") == sha256(plan_path):
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            assignments = {}
            for name, slide, cls, split, size in plan:
                if slide in assignments and assignments[slide] != split:
                    raise ValueError("缓存存在 WSI 跨集合")
                assignments[slide] = split
            print("恢复采样计划：跳过源目录重新扫描。源数据变化时必须加 --rebuild-plan。", flush=True)
            print_summary(plan, assignments, exclusions)
            return plan, assignments
    plan, assignments = make_plan(source, cap, exclusions, report=False)
    directory.mkdir(parents=True, exist_ok=True)
    for path, value in ((plan_path, plan),):
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    temporary = state_path.with_name(state_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"identity": identity, "sha256": sha256(plan_path)}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(state_path)
    print(f"完整采样计划断点已保存：{directory}", flush=True)
    print_summary(plan, assignments, exclusions)
    return plan, assignments


def copy_plan(source, target, plan, assignments, cap, exclusions):
    if target.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖：{target}")
    staging = target.with_name(target.name + ".building")
    identity = {"source": str(source), "cap": cap, "seed": SEED,
                "excluded": sorted(set(exclusions)),
                "plan_sha256": hashlib.sha256(json.dumps(plan).encode()).hexdigest()}
    owner = staging / "sampling_config.json"
    if staging.exists():
        if not owner.is_file() or json.loads(owner.read_text(encoding="utf-8")) != identity:
            raise ValueError("半成品与当前计划不一致，不能自动复用")
        print("恢复复制：SHA256 一致的已有副本将跳过。", flush=True)
    else:
        staging.mkdir(parents=True)
        temporary = owner.with_name(owner.name + ".tmp")
        temporary.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        temporary.replace(owner)
    expected = set()
    for split in SPLITS:
        for cls in CLASSES:
            (staging / split / cls).mkdir(parents=True, exist_ok=True)
    for index, (name, slide, cls, split, size) in enumerate(plan, 1):
        relative = f"{split}/{cls}/{name}"
        expected.add(relative)
        original, destination = source / relative, staging / relative
        if original.stat().st_size != size:
            raise ValueError(f"原图大小变化：{original}")
        digest = sha256(original)
        if not destination.is_file() or sha256(destination) != digest:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".copying")
            shutil.copy2(original, temporary)
            if sha256(temporary) != digest:
                raise IOError(f"复制校验失败：{temporary}")
            temporary.replace(destination)
        if index % 1000 == 0:
            print(f"已完成或复用 {index:,}/{len(plan):,}", flush=True)
    actual = {image.relative_to(staging).as_posix() for split in SPLITS
              for image in (staging / split).rglob("*") if image.is_file() and image.suffix.lower() == ".png"}
    if actual != expected:
        raise ValueError("半成品图片清单与计划不同，拒绝发布")
    fields = ["img_name", "slideID", "cls"]
    write_csv(staging / "mapping.csv", fields, (row[:3] for row in plan))
    write_csv(staging / "split_mapping.csv", fields + ["split"], (row[:4] for row in plan))
    write_csv(staging / "slide_splits.csv", ["slideID", "split"], sorted(assignments.items()))
    write_csv(staging / "excluded_images.csv", ["img_name"], ([name] for name in sorted(set(exclusions))))
    for split in SPLITS:
        write_csv(staging / split / "mapping.csv", fields, (row[:3] for row in plan if row[3] == split))
    staging.rename(target)
    print(f"完成：{target}。WSI 归属不变，源图片未修改。", flush=True)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(config.ROOT))
    parser.add_argument("--source", help="已有 train/val/test 的 FULL 目录，默认使用配置路径")
    parser.add_argument("--output", help="新数据集目录，默认为 FULL-CAP限额")
    parser.add_argument("--max-per-wsi-class", type=int, default=DEFAULT_CAP)
    parser.add_argument("--exclude-image", action="append", default=[], help="已确认坏图，例如 val/TUM/文件名.png")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-plan", action="store_true", help="源数据变化后重新扫描和生成计划")
    args = parser.parse_args(argv)
    project = Path(args.project_root).resolve()
    source = Path(args.source) if args.source else config.paths(project)["raw"].with_name(config.paths(project)["raw"].name + "-FULL")
    if not source.is_absolute():
        source = project / source
    source = source.resolve()
    target = Path(args.output) if args.output else source.with_name(source.name + f"-CAP{args.max_per_wsi_class}")
    if not target.is_absolute():
        target = project / target
    target = target.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("输出与源目录不能相同或互相包含")
    exclusions = sorted(set(name.replace("\\", "/") for name in args.exclude_image))
    plan, assignments = cached_plan(project, source, args.max_per_wsi_class, exclusions, args.rebuild_plan)
    if args.dry_run:
        print("预览完成；计划断点已保存，未复制图片或修改源数据。")
        return
    copy_plan(source, target, plan, assignments, args.max_per_wsi_class, exclusions)


if __name__ == "__main__":
    main()
