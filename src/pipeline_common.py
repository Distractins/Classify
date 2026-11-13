"""两阶段共享的路径、临床字段和患者映射规则。所有路径相对项目根目录。"""
import argparse
import re
from pathlib import Path, PureWindowsPath

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
TME_COLUMNS = [f"{name}%" for name in ("TUM", "LYM", "STR", "MUC", "MUS", "NORM", "ADI", "DEB")]
CSV_DIR = Path("outputs/intermediate_csv")
TRAIN_IDS = CSV_DIR / "train_patient_ids.csv"
VAL_IDS = CSV_DIR / "val_patient_ids.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def project_path(value):
    """CLI 仅接受项目内相对路径；运行位置不影响路径解析。"""
    value = Path(value)
    if value.is_absolute() or PureWindowsPath(str(value)).drive:
        raise ValueError(f"请使用项目相对路径: {value}")
    path = (PROJECT_ROOT / value).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"路径越出项目目录: {value}") from exc
    return path


def relative_path(path):
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def default_dataset():
    # README 使用不换行连字符，也支持下载后常见的 ASCII 连字符目录名。
    for name in ("HMU‑CRC‑Hist550K", "HMU-CRC-Hist550K"):
        value = Path("datasets") / name
        if project_path(value).is_dir():
            return value
    return Path("datasets/HMU‑CRC‑Hist550K")


def clinical_path(dataset, clinical=None):
    if clinical:
        return project_path(clinical)
    for name in ("HMU‑CRC‑Clinical.xlsx", "HMU-CRC-Clinical.xlsx"):
        path = project_path(Path(dataset) / name)
        if path.is_file():
            return path
    raise FileNotFoundError("未找到临床表，请提供 --clinical 项目相对路径。")


def add_data_arguments(parser):
    parser.add_argument("--dataset", default=str(default_dataset()), help="patch 根目录（项目相对路径）")
    parser.add_argument("--clinical", help="临床 xlsx 路径（默认位于数据集目录）")
    parser.add_argument("--id-column", help="临床表患者 ID 列名，默认自动匹配 PatientID")


def add_mapping_arguments(parser):
    parser.add_argument("--patch-map", help="可选 CSV：image_file_path,patient_id；路径相对项目根目录")
    parser.add_argument("--patient-regex", help="相对图像路径的正则；命名组 patient_id 或第一个捕获组")


def normalize_id(value):
    if pd.isna(value) or not str(value).strip():
        raise ValueError("PatientID 不能为空。")
    # Excel 数字型 ID 可能被读取成 1.0；文本型 001 保留前导零。
    text = str(value).strip()
    return re.sub(r"^([0-9]+)\.0+$", r"\1", text)


def find_column(columns, explicit, aliases, description):
    if explicit:
        if explicit not in columns:
            raise ValueError(f"未找到{description}列 {explicit!r}，实际列: {list(columns)}")
        return explicit
    key = lambda s: re.sub(r"[\s_\-‑–/]+", "", str(s)).casefold()
    matched = [c for c in columns if key(c) in {key(a) for a in aliases}]
    if len(matched) != 1:
        raise ValueError(f"无法唯一识别{description}列，请通过参数指定。实际列: {list(columns)}")
    return matched[0]


def read_clinical(path, id_column=None, include_msi=False, age_column=None,
                  sex_column=None, msi_column=None):
    """阶段一只加载 ID 列，绝不加载 MSI 标签或基于 MSI 分层。"""
    columns = pd.read_excel(path, nrows=0).columns
    pid = find_column(columns, id_column, ["PatientID", "patient_id", "患者ID", "患者编号"], "患者 ID")
    selected = {pid: "patient_id"}
    if include_msi:
        selected[find_column(columns, age_column, ["Age", "年龄"], "年龄")] = "age"
        selected[find_column(columns, sex_column, ["Sex", "Gender", "性别"], "性别")] = "sex"
        selected[find_column(columns, msi_column,
                             ["MSI", "MSI status", "MSI-H/MSS", "msi_label", "MSI状态"], "MSI")] = "msi_label"
    table = pd.read_excel(path, usecols=list(selected), dtype={pid: str}).rename(columns=selected)
    table["patient_id"] = table["patient_id"].map(normalize_id)
    if table["patient_id"].duplicated().any():
        raise ValueError("临床表 PatientID 重复；请先核查重复记录，禁止多对多合并。")
    return table


def load_patient_split(expected_ids=None):
    paths = [project_path(TRAIN_IDS), project_path(VAL_IDS)]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("患者划分缺失或不完整，请先运行 patient_dataset_split.py；阶段二不会创建新划分。")
    result = []
    for path in paths:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(table.columns) != ["patient_id"]:
            raise ValueError(f"患者列表必须只有 patient_id 列: {path.name}")
        ids = table["patient_id"].map(normalize_id).tolist()
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"患者列表为空或有重复 ID: {path.name}")
        result.append(ids)
    train, val = result
    if set(train) & set(val):
        raise ValueError("训练和验证患者有交集，停止以防数据泄露。")
    if expected_ids is not None and set(train + val) != set(expected_ids):
        raise ValueError("保存的划分与当前临床患者集合不一致；禁止静默重新划分。")
    return train, val


class PatchPatientResolver:
    """优先显式映射/正则，否则只接受路径中唯一的完整患者 ID。"""
    def __init__(self, patient_ids, patch_map=None, patient_regex=None):
        self.ids = set(patient_ids)
        self.mapping = None
        self.regex = re.compile(patient_regex) if patient_regex else None
        if patch_map and patient_regex:
            raise ValueError("--patch-map 与 --patient-regex 只能选择一种。")
        if patch_map:
            table = pd.read_csv(project_path(patch_map), dtype=str, keep_default_na=False)
            if not {"image_file_path", "patient_id"}.issubset(table.columns):
                raise ValueError("映射 CSV 必须包含 image_file_path,patient_id。")
            self.mapping = {}
            for path, pid in zip(table.image_file_path, table.patient_id):
                key, pid = relative_path(project_path(path)), normalize_id(pid)
                if key in self.mapping or pid not in self.ids:
                    raise ValueError(f"映射重复或患者未登记: {path}, {pid}")
                self.mapping[key] = pid
        self.tokens = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(
            re.escape(pid) for pid in sorted(self.ids, key=lambda s: (-len(s), s))) + r")(?![A-Za-z0-9])")

    def __call__(self, path):
        path = relative_path(path)
        if self.mapping is not None:
            pid = self.mapping.get(path)
        elif self.regex:
            match = self.regex.search(path)
            if not match:
                pid = None
            elif "patient_id" in self.regex.groupindex:
                pid = match.group("patient_id")
            elif self.regex.groups:
                pid = match.group(1)
            else:
                raise ValueError("--patient-regex 必须包含 patient_id 命名组或捕获组。")
        else:
            # 仅匹配文件名及数据集内父目录，避免项目路径意外包含数字 ID。
            parts = Path(path).parts
            matches = set(self.tokens.findall("/".join(parts[2:])))
            pid = next(iter(matches)) if len(matches) == 1 else None
        if pid is None or normalize_id(pid) not in self.ids:
            raise ValueError(f"无法唯一关联 PatientID: {path}；请提供 --patch-map 或 --patient-regex。")
        return normalize_id(pid)


def partition_imagefolder(dataset, resolver):
    """训练脚本共享入口：按已保存 ID 分配所有 patch，绝不随机划分图片。"""
    train, val = load_patient_split(resolver.ids)
    train_ids = set(train)
    train_indices, val_indices = [], []
    for index, (path, _) in enumerate(dataset.samples):
        pid = resolver(path)
        (train_indices if pid in train_ids else val_indices).append(index)
    if not train_indices or not val_indices:
        raise ValueError("训练或验证患者没有 patch，请检查数据目录及患者映射。")
    return train_indices, val_indices


def stage1_data_setup():
    """供现有三个训练入口使用；参数与患者划分、推理脚本一致。"""
    parser = argparse.ArgumentParser(description="8 类组织训练（复用固定患者划分，不读取 MSI）")
    add_data_arguments(parser)
    add_mapping_arguments(parser)
    args = parser.parse_args()
    clinical = read_clinical(clinical_path(args.dataset, args.clinical), args.id_column)
    load_patient_split(clinical.patient_id)
    resolver = PatchPatientResolver(clinical.patient_id, args.patch_map, args.patient_regex)
    return project_path(args.dataset), resolver
