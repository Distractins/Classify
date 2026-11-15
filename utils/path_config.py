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


def parse_patient_id(filename):
    """patient_00101_001.png -> '00101'，原样保留前导零。"""
    parts = Path(filename).stem.split("_")
    if (len(parts) != 3 or parts[0] != "patient"
            or not parts[1] or not parts[2] or parts[1] != parts[1].strip()):
        raise ValueError(f"图片名不符合 patient_患者ID_patch序号 格式：{filename}")
    return parts[1]


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="预览另一台机器的根目录配置")
    args = parser.parse_args()
    print(json.dumps({key: str(value) for key, value in paths(args.root).items()},
                     indent=2, ensure_ascii=False))
