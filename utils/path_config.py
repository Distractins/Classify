"""仅用于 patch 八分类；迁移机器时只修改 ROOT，不读取临床表。"""
from pathlib import Path
import argparse
import os

# ==================== CONFIG ====================
ROOT = Path(r"D:\xcjhhh\Classify")
DATASET_NAME = "HMU‑CRC‑Hist550K"  # U+2011 连字符，须与实际目录一致
CLASSES = ("ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
SEED = 42
# =================================================


def paths(root=None):
    base = Path(root if root is not None else ROOT).expanduser().resolve()
    return {
        "root": base,
        "raw": base / "datasets" / DATASET_NAME,
        "stat": base / "output" / "stat",
        "fast": base / "datasets" / (DATASET_NAME + "-FAST"),
        "paper": base / "datasets" / (DATASET_NAME + "-PAPER"),
    }


def prepare_plotting(root=None):
    cache = paths(root)["root"] / "output" / ".matplotlib"
    os.makedirs(cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["MPLBACKEND"] = "Agg"


def parse_slide_id(filename, expected_class):
    """首段为 slideID，保留前导零；允许重复的同类后缀。"""
    parts = Path(filename).stem.split("_")
    end = len(parts)
    while end and parts[end - 1] in CLASSES:
        end -= 1
    numbers, labels = parts[:end], parts[end:]
    if (len(numbers) < 3 or not labels
            or any(not (s.isascii() and s.isdigit()) for s in numbers)
            or any(label != expected_class for label in labels)):
        raise ValueError(f"图片名或类别不符合规则：{filename}；目录类别：{expected_class}")
    return parts[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="临时覆盖项目根路径")
    args = parser.parse_args()
    for key, value in paths(args.root).items():
        print(f"{key}: {value}")
