"""统计每位患者的八类 TME 百分比，再合并年龄、性别和 MSI 标签。"""
import argparse
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import (CLASSES, CSV_DIR, TME_COLUMNS, add_data_arguments,
                             clinical_path, load_patient_split, normalize_id,
                             project_path, read_clinical, relative_path)

import numpy as np
import pandas as pd


def encode_msi(value):
    if pd.isna(value) or not str(value).strip():
        return np.nan
    text = re.sub(r"[\s_\-‑–]+", "", str(value)).upper()
    if text in {"MSIH", "1", "1.0"}:
        return 1
    if text in {"MSS", "0", "0.0"}:
        return 0
    # MSI-L、未知、未检测等不自动归为 MSS；非二分类状态保留为缺失。
    warnings.warn(f"MSI 状态 {value!r} 不属于 MSI-H/MSS，标记缺失并在训练前排除。")
    return np.nan


def encode_sex(value):
    if pd.isna(value) or not str(value).strip():
        return np.nan
    text = str(value).strip().casefold()
    if text in {"male", "m", "男", "1", "1.0"}:
        return 1
    if text in {"female", "f", "女", "0", "0.0"}:
        return 0
    raise ValueError(f"不支持性别值 {value!r}；请将临床表编码为 Male/Female 或 1/0。")


def aggregate_patches(path, known_ids, chunksize=100000):
    """分块累计 550K patch，八类即使未出现也保留为 0。"""
    counts = pd.DataFrame(0, index=sorted(known_ids), columns=list(CLASSES), dtype="int64")
    seen_paths = set()
    with pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunksize) as reader:
        for chunk in reader:
            # 兼容 README 两处给出的列名，但本项目统一输出长列名。
            chunk = chunk.rename(columns={"pred_class": "predict_class", "image_path": "image_file_path"})
            if not {"patient_id", "image_file_path", "predict_class"}.issubset(chunk.columns):
                raise ValueError("patch CSV 必须包含 patient_id,image_file_path,predict_class。")
            chunk["patient_id"] = chunk.patient_id.map(normalize_id)
            if not set(chunk.patient_id).issubset(known_ids):
                raise ValueError("patch 中有未登记到临床表/患者名单的 PatientID。")
            if not set(chunk.predict_class).issubset(CLASSES):
                raise ValueError("patch 预测类别不属于 README 定义的八类。")
            paths = chunk.image_file_path.map(lambda p: relative_path(project_path(p)))
            if paths.duplicated().any() or seen_paths.intersection(paths):
                raise ValueError("patch CSV 含重复图像路径，禁止重复计数。")
            seen_paths.update(paths)
            block = pd.crosstab(chunk.patient_id, chunk.predict_class)
            counts = counts.add(block.reindex(index=counts.index, columns=counts.columns, fill_value=0), fill_value=0)
    totals = counts.sum(axis=1)
    missing = totals[totals == 0].index.tolist()
    if missing:
        raise ValueError(f"以下患者没有 patch，无法统计占比（不会静默丢弃）: {missing}")
    result = counts.div(totals, axis=0).mul(100)
    result.columns = [f"{cls}%" for cls in result.columns]
    result.index.name = "patient_id"
    return result[TME_COLUMNS].reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--patch-csv", default=str(CSV_DIR / "patch_infer_result.csv"))
    parser.add_argument("--age-column")
    parser.add_argument("--sex-column")
    parser.add_argument("--msi-column")
    args = parser.parse_args()
    clinical = read_clinical(clinical_path(args.dataset, args.clinical), args.id_column, True,
                             args.age_column, args.sex_column, args.msi_column)
    load_patient_split(clinical.patient_id)
    tme = aggregate_patches(project_path(args.patch_csv), set(clinical.patient_id))
    clinical["age"] = pd.to_numeric(clinical.age, errors="raise")
    if ((clinical.age < 0) | (clinical.age > 120) | np.isinf(clinical.age)).any():
        raise ValueError("年龄值超出 0~120 岁或为无穷大，请核查临床表。")
    clinical["sex"] = clinical.sex.map(encode_sex)
    clinical["msi_label"] = clinical.msi_label.map(encode_msi)
    result = tme.merge(clinical, on="patient_id", how="left", validate="one_to_one")
    output = project_path(CSV_DIR / "patient_tme_feature.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"已输出 {len(result)} 位患者；占比单位为百分数（合计 100）: {relative_path(output)}")
    print(f"缺失/非 MSI-H/MSS 标签: {result.msi_label.isna().sum()} 位；不会改变其患者划分。")


if __name__ == "__main__":
    main()
