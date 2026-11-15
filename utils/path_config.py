"""本阶段统一路径配置；迁移机器时只修改 ROOT 字符串。"""
from pathlib import Path
import os

# ==================== CONFIG ====================
ROOT = Path(r"D:\xcjhhh\Classify")  # Linux 示例：Path(r"/data/Classify")
CLASSES = ("ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DATASET_NAME = "HMU‑CRC‑Hist550K"  # 与固定目录一致，使用不换行连字符 U+2011
CLINICAL_NAME = "HMU-CRC-Clinical.xlsx"  # 实际 Excel 文件名使用普通连字符 -
# =================================================


def paths(root=None):
    """返回一套路径；--root 可临时覆盖 ROOT，不受终端工作目录影响。"""
    base = Path(root if root is not None else ROOT).expanduser().resolve()
    raw = base / "datasets" / DATASET_NAME
    subset = base / "datasets" / (DATASET_NAME + "-SUBSET")
    return {
        "root": base, "raw": raw, "clinical": raw / CLINICAL_NAME,
        "stat": base / "output" / "stat", "subset": subset,
        "sampling": base / "output" / "sampling",
    }


def prepare_plotting(root=None):
    """让 Matplotlib 缓存在项目内，适用于 Windows 和无桌面的服务器。"""
    cache = paths(root)["root"] / "output" / ".matplotlib"
    os.makedirs(cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["MPLBACKEND"] = "Agg"


def parse_patient_id(filename, expected_class=None):
    """兼容 patient_00101_001.png 和 1302097_0_19264_ADI.png；保留字符串 ID。"""
    parts = Path(filename).stem.split("_")
    if (len(parts) == 3 and parts[0] == "patient"
            and parts[1] and parts[2] and parts[1] == parts[1].strip()):
        return parts[1]
    # 兼容重复添加类别后缀，如 1434704_1568_21728_ADI_ADI.png。
    suffix_start = len(parts)
    while suffix_start > 0 and parts[suffix_start - 1] in CLASSES:
        suffix_start -= 1
    labels = parts[suffix_start:]
    numeric_parts = parts[:suffix_start]
    if (labels and len(numeric_parts) >= 2
            and all(part.isascii() and part.isdigit() for part in numeric_parts)):
        if len(set(labels)) != 1:
            raise ValueError(f"图片包含冲突的组织类别后缀：{filename}")
        if expected_class is not None and labels[0] != expected_class:
            raise ValueError(f"图片类别与文件夹不一致：{filename}，所在文件夹：{expected_class}")
        return parts[0]
    raise ValueError(
        f"不支持的图片名：{filename}；支持 patient_患者ID_序号 或 患者ID_数字序号_组织类别。"
    )


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="预览另一台机器的根目录配置")
    args = parser.parse_args()
    print(json.dumps({key: str(value) for key, value in paths(args.root).items()},
                     indent=2, ensure_ascii=False))
