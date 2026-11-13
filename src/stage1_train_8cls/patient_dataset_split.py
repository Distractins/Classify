"""一次性按 PatientID 划分患者；重复运行仅验证并复用已有名单。"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import (CSV_DIR, TRAIN_IDS, VAL_IDS, add_data_arguments,
                             clinical_path, load_patient_split, project_path,
                             read_clinical, relative_path)


def create_patient_split(clinical, val_ratio=0.3, seed=42):
    paths = [project_path(TRAIN_IDS), project_path(VAL_IDS)]
    if any(path.exists() for path in paths):
        # 存在半套名单也必须报错，不能重新随机分配患者。
        return (*load_patient_split(clinical.patient_id), True)
    if not 0 < val_ratio < 1 or len(clinical) < 2:
        raise ValueError("val-ratio 必须在 (0,1) 内，且临床表至少有两位患者。")
    ids = sorted(clinical.patient_id.tolist())
    random.Random(seed).shuffle(ids)  # 仅首次对患者 ID 随机划分，不访问 MSI。
    n_val = max(1, min(len(ids) - 1, math.ceil(len(ids) * val_ratio)))
    val, train = sorted(ids[:n_val]), sorted(ids[n_val:])
    project_path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    import pandas as pd
    for path, values in zip(paths, (train, val)):
        pd.DataFrame({"patient_id": values}).to_csv(path, index=False, encoding="utf-8-sig")
    metadata = {"seed": seed, "val_ratio": val_ratio, "n_train": len(train),
                "n_val": len(val), "level": "patient", "uses_msi": False}
    project_path(CSV_DIR / "patient_split_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    load_patient_split(clinical.patient_id)
    return train, val, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    clinical = read_clinical(clinical_path(args.dataset, args.clinical), args.id_column)
    train, val, reused = create_patient_split(clinical, args.val_ratio, args.seed)
    print(f"{'复用已有' if reused else '创建'}患者划分: train={len(train)}, validation={len(val)}")
    print(f"训练患者: {train}\n验证患者: {val}")
    print(f"名单: {relative_path(project_path(TRAIN_IDS))}, {relative_path(project_path(VAL_IDS))}")


if __name__ == "__main__":
    main()
