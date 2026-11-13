"""扫描八类 patch，按文件名生成 img_name,patient_id,cls 映射。"""
from pathlib import Path

import pandas as pd

# ==================== CONFIG ====================
RAW_ROOT = Path(r"D:\dataset\HMU-CRC-ORIGIN")
SUBSET_ROOT = Path(r"D:\dataset\HMU-CRC-SUBSET")
MAP_CSV_PATH = Path(r"D:\dataset\patch_patient_map.csv")
CLINICAL_EXCEL = Path(r"F:\new\data\HMU-CRC-Clinical.xlsx")  # 本脚本不读取
CLASSES = ("ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
IMAGE_SUFFIXES = {".png", ".jpg"}

# 已确认命名：patient_00101_001.png -> "00101"；末段仅为 patch 序号。
PATIENT_ID_SEPARATOR = "_"
PATIENT_ID_FIELD_INDEX = 1
# =================================================


def parse_patient_id(img_name):
    """仅从文件名提取患者 ID，不读取临床表，不把切片号自动当成患者。"""
    if not PATIENT_ID_SEPARATOR:
        raise ValueError("PATIENT_ID_SEPARATOR 不能为空。")
    if (isinstance(PATIENT_ID_FIELD_INDEX, bool)
            or not isinstance(PATIENT_ID_FIELD_INDEX, int)
            or PATIENT_ID_FIELD_INDEX < 0):
        raise ValueError("PATIENT_ID_FIELD_INDEX 必须是从 0 开始的非负整数。")
    parts = Path(img_name).stem.split(PATIENT_ID_SEPARATOR)
    if (len(parts) != 3 or parts[0] != "patient" or not parts[2]
            or PATIENT_ID_FIELD_INDEX >= len(parts)):
        raise ValueError(f"文件名不符合配置的患者 ID 规则：{img_name}，请检查 CONFIG。")
    # 保留字符串原样，禁止 int()，例如 "00101" 不得变成 101。
    patient_id = parts[PATIENT_ID_FIELD_INDEX]
    if (not patient_id or patient_id != patient_id.strip()
            or patient_id.upper() in CLASSES):
        raise ValueError(f"解析出空 ID 或组织类别：{img_name} -> {patient_id!r}，请检查 CONFIG。")
    return patient_id


def main():
    raw_root = RAW_ROOT.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在，请检查 RAW_ROOT：{raw_root}")
    # 输入目录缺失应提示错误，避免误将空目录作为完整数据集。
    for cls in CLASSES:
        if not (raw_root / cls).is_dir():
            raise FileNotFoundError(f"缺少类别目录：{raw_root / cls}")

    records = []
    seen_sources = set()
    print(f"扫描原始目录：{raw_root}")
    print(f"解析规则：去扩展名后按 {PATIENT_ID_SEPARATOR!r} 分隔，取第 {PATIENT_ID_FIELD_INDEX + 1} 段")
    for cls in CLASSES:
        folder = raw_root / cls
        count = 0
        # 普通平铺目录的 img_name 就是文件名；有子目录时记录类别目录内相对路径。
        for image in sorted(folder.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = image.resolve()
            try:
                resolved.relative_to(folder.resolve())
            except ValueError as error:
                raise ValueError(f"图片链接指向类别目录之外：{image}") from error
            if resolved in seen_sources:
                raise ValueError(f"同一张源图片被重复引用：{image}")
            seen_sources.add(resolved)
            patient_id = parse_patient_id(image.name)
            records.append({"img_name": image.relative_to(folder).as_posix(),
                            "patient_id": patient_id, "cls": cls})
            count += 1
        print(f"{cls}: {count:,} 张 png/jpg" + ("（该类别为空）" if count == 0 else ""))
    if not records:
        raise ValueError("八个类别目录中没有 png/jpg 图片，未生成 CSV。")

    table = pd.DataFrame(records, columns=["img_name", "patient_id", "cls"])
    MAP_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MAP_CSV_PATH.with_suffix(MAP_CSV_PATH.suffix + ".tmp")
    table.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(MAP_CSV_PATH)
    print("\n==== 映射完成 ====")
    print(f"总图片数：{len(table):,}")
    print(f"总患者数：{table.patient_id.nunique()}")
    print(f"映射文件：{MAP_CSV_PATH.resolve()}")
    print("文件名解析示例（请确认 patient_id 对应真实患者）：")
    print(table.head(5).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"[错误] {error}")
