"""逐患者、逐组织类别限额采样；复制图片，保留原始数据。"""
from pathlib import Path
import random
import shutil

import pandas as pd

# ==================== CONFIG ====================
RAW_ROOT = Path(r"D:\dataset\HMU-CRC-ORIGIN")
SUBSET_ROOT = Path(r"D:\dataset\HMU-CRC-SUBSET")
MAP_CSV_PATH = Path(r"D:\dataset\patch_patient_map.csv")
CLINICAL_EXCEL = Path(r"F:\new\data\HMU-CRC-Clinical.xlsx")  # 仅供后续 MSI 阶段使用
CLASSES = ("ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
IMAGE_SUFFIXES = {".png", ".jpg"}
max_patch_per_patient_per_class = 8
SEED = 42
# =================================================


def is_inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main():
    if (isinstance(max_patch_per_patient_per_class, bool)
            or not isinstance(max_patch_per_patient_per_class, int)
            or max_patch_per_patient_per_class < 1):
        raise ValueError("max_patch_per_patient_per_class 必须是正整数，确保每位患者保留 patch。")
    raw_root, subset_root = RAW_ROOT.resolve(), SUBSET_ROOT.resolve()
    if is_inside(subset_root, raw_root) or is_inside(raw_root, subset_root):
        raise ValueError("原始目录和子集目录不能相同或互相包含。")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在：{raw_root}")
    for cls in CLASSES:
        if not (raw_root / cls).is_dir():
            raise FileNotFoundError(f"原始类别目录不存在：{raw_root / cls}")
    if not MAP_CSV_PATH.is_file():
        raise FileNotFoundError(f"映射 CSV 不存在，请先运行 build_patch_map.py：{MAP_CSV_PATH}")

    # 始终按字符串读取，保留患者 ID 的前导零；不读取临床 Excel。
    table = pd.read_csv(MAP_CSV_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if list(table.columns) != ["img_name", "patient_id", "cls"]:
        raise ValueError("映射 CSV 必须依次包含三列：img_name,patient_id,cls。")
    if table.empty or table.apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("映射 CSV 为空，或存在空文件名/患者 ID/类别。")
    table["patient_id"] = table.patient_id.str.strip()
    if not set(table.cls).issubset(CLASSES):
        raise ValueError(f"存在非法类别：{sorted(set(table.cls) - set(CLASSES))}")

    # 先核查全部映射，再采样；缺文件不能静默跳过，否则可能丢失患者。
    names = []
    source_keys = set()
    for img_name, cls in zip(table.img_name, table.cls):
        relative = Path(img_name.replace("\\", "/"))
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError(f"img_name 必须为类别目录内的相对路径：{img_name}")
        if relative.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"映射中有非 png/jpg 图片：{img_name}")
        source = (raw_root / cls / relative).resolve()
        if not is_inside(source, (raw_root / cls).resolve()) or not source.is_file():
            raise FileNotFoundError(f"映射对应图片不存在或越出类别目录：{cls}/{img_name}")
        if source in source_keys:
            raise ValueError(f"映射 CSV 重复引用同一张图片：{cls}/{img_name}")
        source_keys.add(source)
        names.append(relative.as_posix())
    table["img_name"] = names

    patients = sorted(table.patient_id.unique())
    print(f"映射图片数：{len(table):,}")
    print(f"总患者数：{len(patients)}")
    print(f"随机种子：{SEED}；每位患者每类上限：{max_patch_per_patient_per_class}")
    rng = random.Random(SEED)
    selected = []
    # 排序保证相同 CSV 内容即使行顺序改变，也得到相同的抽样结果。
    for patient_id, patient_table in table.groupby("patient_id", sort=True):
        for cls in CLASSES:
            candidates = sorted(patient_table.loc[patient_table.cls == cls, "img_name"].tolist())
            if not candidates:
                continue
            take = min(max_patch_per_patient_per_class, len(candidates))
            selected.extend((patient_id, cls, name) for name in rng.sample(candidates, take))
    kept_patients = {pid for pid, _, _ in selected}
    if kept_patients != set(patients):
        raise RuntimeError("患者保留检查失败，停止复制。")

    # 不清空已有文件；发现旧采样残留时中止，防止重复运行后突破每组上限。
    subset_root.mkdir(parents=True, exist_ok=True)
    selected_paths = {(subset_root / cls / name).resolve() for _, cls, name in selected}
    for cls in CLASSES:
        folder = subset_root / cls
        if not is_inside(folder.resolve(), subset_root):
            raise ValueError(f"子集类别目录不能链接到项目外：{folder}")
        folder.mkdir(parents=True, exist_ok=True)
        existing = {p.resolve() for p in folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        extras = existing - selected_paths
        if extras:
            raise ValueError(
                f"子集目录含本次采样以外的旧图片，例如：{next(iter(extras))}。"
                "请使用空的 SUBSET_ROOT 或先自行整理旧子集，脚本不会删除文件。")

    counts = {cls: 0 for cls in CLASSES}
    for index, (_, cls, img_name) in enumerate(selected, start=1):
        source = raw_root / cls / img_name
        destination = subset_root / cls / img_name
        if not is_inside(destination.resolve(), subset_root):
            raise ValueError(f"目标路径越出子集目录：{destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)  # 复制，不剪切、不删除原图。
        counts[cls] += 1
        if index % 1000 == 0:
            print(f"已复制：{index:,}/{len(selected):,}")

    print("\n==== 采样完成 ====")
    print(f"采样前患者数：{len(patients)}；采样后患者数：{len(kept_patients)}（全部保留）")
    for cls in CLASSES:
        print(f"{cls}: {counts[cls]:,} 张")
    print(f"总计：{len(selected):,} 张")
    print(f"子集目录：{subset_root}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"[错误] {error}")
