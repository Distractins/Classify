"""仅用于 patch 八分类；迁移机器时只修改 ROOT，不读取临床表。"""
from pathlib import Path
import argparse
import os

# ==================== CONFIG ====================
ROOT = Path(__file__).resolve().parents[1]  # 默认随项目迁移；可改为 Path(r"/data/Classify")
DATASET_NAME = "HMU-CRC-Hist550K"  # 新目录统一使用 ASCII 短横杠
CLASSES = ("ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
SEED = 42
# =================================================


def paths(root=None):
    base = Path(root if root is not None else ROOT).expanduser().resolve()
    # 已有 Unicode 目录仍可使用，不自动重命名原图。
    dataset_name = DATASET_NAME
    legacy_name = "HMU‑CRC‑Hist550K"
    if not any((base / "datasets" / (dataset_name + suffix)).exists() for suffix in ("", "-PAPER")):
        if any((base / "datasets" / (legacy_name + suffix)).exists() for suffix in ("", "-PAPER")):
            dataset_name = legacy_name
    return {
        "root": base,
        "raw": base / "datasets" / dataset_name,
        "stat": base / "output" / "stat",
        "fast": base / "datasets" / (dataset_name + "-FAST"),
        "paper": base / "datasets" / (dataset_name + "-PAPER"),
    }


def prepare_plotting(root=None):
    cache = paths(root)["root"] / "output" / ".matplotlib"
    os.makedirs(cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["MPLBACKEND"] = "Agg"


def parse_slide_id(filename, expected_class):
    """首段为 slideID，保留前导零；允许重复的同类后缀。"""
    # 批量处理仅做字符串解析，避免每张图片重复创建 Path。
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    parts = stem.split("_")
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
