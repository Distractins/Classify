"""Verify a copied WSI dataset and finish its CSVs without copying images again."""
import argparse
import csv
import os
from pathlib import Path
from utils import path_config as config

SPLITS = ("train", "val", "test")
FIELDS = ["img_name", "slideID", "cls"]


def read_rows(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f"Invalid columns: {path}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Incomplete CSV row: {path}")
            rows.append(tuple(row[key] for key in fields))
        return rows


def write_rows(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    # Write plain UTF-8 (fully compatible with utf-8-sig readers).
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def finalize(staging, raw, expected_count=None, check_only=False):
    if not staging.name.endswith(".building"):
        raise ValueError("Input directory must end with .building")
    target = staging.with_name(staging.name[:-len(".building")])
    if target.exists():
        raise FileExistsError(f"Final dataset already exists: {target}")
    plan = read_rows(staging / "split_mapping.csv", FIELDS + ["split"])
    full = read_rows(staging / "mapping.csv", FIELDS)
    if not plan or (expected_count is not None and len(plan) != expected_count):
        raise ValueError(f"Unexpected patch count: {len(plan)}")
    planned = [row[:3] for row in plan]
    if len(set(planned)) != len(planned) or set(planned) != set(full) or len(full) != len(plan):
        raise ValueError("Top-level mapping files disagree or contain duplicates")
    declared = read_rows(staging / "slide_splits.csv", ["slideID", "split"])
    if len({row[0] for row in declared}) != len(declared):
        raise ValueError("Duplicate slide IDs in slide_splits.csv")
    assignments, expected = {}, set()
    subsets = {split: [] for split in SPLITS}
    for index, (name, slide, cls, split) in enumerate(plan, 1):
        relative = name.replace("\\", "/")
        if (split not in SPLITS or cls not in config.CLASSES or ":" in relative
                or any(part in ("", ".", "..") for part in relative.split("/"))
                or not relative.lower().endswith(".png")
                or config.parse_slide_id(relative.rsplit("/", 1)[-1], cls) != slide):
            raise ValueError(f"Invalid mapping row: {(name, slide, cls, split)}")
        if slide in assignments and assignments[slide] != split:
            raise ValueError(f"WSI crosses splits: {slide}")
        assignments[slide] = split
        key = f"{split}/{cls}/{relative}"
        if key in expected:
            raise ValueError(f"Duplicate image path: {key}")
        expected.add(key)
        destination, original = staging / key, raw / cls / relative
        if (not destination.is_file() or destination.is_symlink()
                or not original.is_file() or original.is_symlink()
                or destination.stat().st_size != original.stat().st_size):
            raise ValueError(f"Missing image or size mismatch: {key}")
        subsets[split].append((name, slide, cls))
        if index % 100000 == 0:
            print(f"Verified image existence and size: {index:,}/{len(plan):,}", flush=True)
    if dict(declared) != assignments:
        raise ValueError("slide_splits.csv disagrees with split_mapping.csv")
    actual = set()
    for split in SPLITS:
        if not subsets[split] or {r[2] for r in subsets[split]} != set(config.CLASSES):
            raise ValueError(f"Missing split or class: {split}")
        for current, dirs, files in os.walk(staging / split, followlinks=False):
            if any(os.path.islink(os.path.join(current, d)) for d in dirs):
                raise ValueError(f"Symbolic link directory: {current}")
            for name in files:
                if name.lower().endswith(".png"):
                    actual.add(os.path.relpath(os.path.join(current, name), staging).replace("\\", "/"))
    if actual != expected:
        raise ValueError("Actual image list differs from mappings")
    for split in SPLITS:
        print(f"{split}: {len(subsets[split]):,} patches / {sum(s == split for s in assignments.values())} slides", flush=True)
    print("Verified mappings, WSI isolation, image lists and byte sizes (not a new SHA256 check).", flush=True)
    if check_only:
        return
    for split in SPLITS:
        write_rows(staging / split / "mapping.csv", subsets[split])
    staging.rename(target)
    print(f"Completed: {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Project directory")
    parser.add_argument("--building", help="Incomplete dataset path, relative to project")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    p = config.paths(args.root)
    staging = Path(args.building) if args.building else p["raw"].with_name(p["raw"].name + "-FULL.building")
    if not staging.is_absolute():
        staging = p["root"] / staging
    finalize(staging, p["raw"], args.expected_count, args.check_only)
