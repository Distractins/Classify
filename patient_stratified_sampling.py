"""以患者为不可拆分单元，在总量/类别上限内构造小规模子集。"""
import argparse
import json
import os
import random
import shutil
import tempfile

import pandas as pd
from tqdm import tqdm
from utils import path_config as config
from data_explore import find_column, patient_counts, read_clinical, scan_patch_map


def choose_patients(patients, total_cap, class_caps, seed):
    """患者级贪心：优先补齐缺失组织类别，再照顾 MSI 少数层；不拆分 patch。"""
    rng = random.Random(seed)
    rows = patients.sort_values("patient_id").to_dict("records")
    tie_break = {row["patient_id"]: rng.random() for row in rows}
    used = {cls: 0 for cls in config.CLASSES}
    strata = {"MSI-H": 0, "MSS": 0, "UNKNOWN": 0}
    selected, rejected = [], []
    for row in rows[:]:
        if row["total_patches"] > total_cap or any(row[c] > class_caps[c] for c in config.CLASSES):
            rejected.append(row["patient_id"])
    rejected_ids = set(rejected)
    remaining = [row for row in rows if row["patient_id"] not in rejected_ids]
    total = 0
    while remaining:
        feasible = [row for row in remaining if total + row["total_patches"] <= total_cap
                    and all(used[c] + row[c] <= class_caps[c] for c in config.CLASSES)]
        if not feasible:
            break
        scarcity = {c: 1 / max(1, sum(row[c] > 0 for row in feasible)) for c in config.CLASSES}

        def priority(row):
            new = [c for c in config.CLASSES if used[c] == 0 and row[c] > 0]
            label = row["msi_status"]
            balance = -1
            if label != "UNKNOWN":
                other = "MSS" if label == "MSI-H" else "MSI-H"
                balance = 2 if strata[label] < strata[other] else 1 if strata[label] == strata[other] else 0
            return (len(new), sum(scarcity[c] for c in new), balance,
                    -row["total_patches"], tie_break[row["patient_id"]])

        chosen = max(feasible, key=priority)
        selected.append(chosen["patient_id"])
        total += chosen["total_patches"]
        strata[chosen["msi_status"]] += 1
        for cls in config.CLASSES:
            used[cls] += chosen[cls]
        remaining = [row for row in remaining if row["patient_id"] != chosen["patient_id"]]
    return selected, used, strata, rejected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="临时覆盖 path_config.ROOT")
    parser.add_argument("--max-total-patches", type=int, default=20000)
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--class-cap", action="append", default=[], metavar="CLS=N", help="覆盖某类上限，可重复")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-column")
    parser.add_argument("--msi-column")
    parser.add_argument("--dry-run", action="store_true", help="只生成采样计划，不复制图片")
    args = parser.parse_args(argv)
    if args.max_total_patches < 1 or args.max_per_class < 1:
        parser.error("总量和每类上限必须为正整数。")
    caps = dict.fromkeys(config.CLASSES, args.max_per_class)
    for override in args.class_cap:
        cls, separator, value = override.partition("=")
        if separator != "=" or cls not in caps or not value.isdigit() or int(value) < 1:
            parser.error(f"非法类别上限 {override!r}，示例：--class-cap LYM=3000")
        caps[cls] = int(value)
    p = config.paths(args.root)
    map_path = p["stat"] / "patch_patient_map.csv"
    if not map_path.is_file():
        raise FileNotFoundError(f"请先运行 data_explore.py：{map_path}")
    mapping = pd.read_csv(map_path, dtype=str, keep_default_na=False)
    if list(mapping.columns) != ["img_name", "patient_id", "cls"] or mapping.empty:
        raise ValueError("映射必须为非空三列 img_name,patient_id,cls。")

    # 完整复扫核对，防止旧映射/缺行/新增图像导致只复制患者的一部分 patch。
    current = scan_patch_map(p["raw"])
    columns = ["cls", "img_name", "patient_id"]
    canonical = lambda frame: frame.sort_values(columns).reset_index(drop=True)
    if not canonical(mapping).equals(canonical(current)):
        raise ValueError("映射与当前完整数据不一致，请重新运行 data_explore.py；禁止用部分映射抽样。")
    clinical = read_clinical(p["clinical"], args.id_column, args.msi_column, p["stat"])
    patients = patient_counts(mapping).merge(clinical, on="patient_id", how="left", validate="one_to_one")
    if patients.msi_status.isna().any():
        missing = patients.loc[patients.msi_status.isna(), "patient_id"].tolist()
        raise ValueError(f"图片患者无临床匹配，请先核查 ID/前导零：{missing}")
    selected, counts, strata, rejected = choose_patients(patients, args.max_total_patches, caps, args.seed)
    chosen = mapping[mapping.patient_id.isin(selected)].sort_values(["patient_id", "cls", "img_name"])
    report = {"seed": args.seed, "strategy": "whole_patient_greedy", "max_total_patches": args.max_total_patches,
              "class_caps": caps, "selected_patients": len(selected), "selected_patches": len(chosen),
              "class_counts": counts, "msi_patient_counts": strata, "individually_over_cap": rejected,
              "missing_classes": [c for c in config.CLASSES if counts[c] == 0], "dry_run": args.dry_run}
    os.makedirs(p["sampling"], exist_ok=True)
    patients.assign(selected=patients.patient_id.isin(selected)).to_csv(
        p["sampling"] / "patient_selection.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(p["sampling"] / "planned_patch_patient_map.csv", index=False, encoding="utf-8-sig")
    (p["sampling"] / "sampling_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n患者：{len(patients)} -> {len(selected)}；patch：{len(mapping):,} -> {len(chosen):,}")
    print("每类数量：", counts)
    print("MSI 患者数量：", strata)
    print(f"单个患者本身超过上限而不能选择：{len(rejected)} 位")
    if report["missing_classes"]:
        print("当前上限下未覆盖的类别：", report["missing_classes"])
    if not selected:
        raise ValueError("没有整位患者能满足上限，请提高限额；不会退回逐 patch 抽样。")
    if args.dry_run:
        print(f"仅预览，未复制图片；计划：{p['sampling']}")
        return
    if p["subset"].exists():
        raise FileExistsError(f"子集已存在，不合并或清空旧数据：{p['subset']}；请先自行备份移走。")

    # 完整复制到临时目录，成功后发布；失败会保留临时目录便于检查，不发布残缺子集。
    os.makedirs(p["subset"].parent, exist_ok=True)
    staging = p["subset"].parent / tempfile.mkdtemp(prefix=p["subset"].name + ".building-", dir=p["subset"].parent)
    for cls in config.CLASSES:
        os.makedirs(staging / cls, exist_ok=True)
    try:
        for row in tqdm(chosen.itertuples(index=False), total=len(chosen), desc="复制整患者 patch"):
            destination = staging / row.cls / row.img_name
            os.makedirs(destination.parent, exist_ok=True)
            shutil.copy2(p["raw"] / row.cls / row.img_name, destination)
        copied = scan_patch_map(staging)
        if not canonical(copied).equals(canonical(chosen)):
            raise RuntimeError("复制后患者/patch 完整性核查失败。")
        # 输出只含选中患者的临床表，保持原临床列名，ID 以字符串存储。
        full_clinical = pd.read_excel(p["clinical"], dtype=str, keep_default_na=False)
        id_col = find_column(full_clinical.columns, args.id_column,
                             ["Patient", "PatientID", "patient_id", "患者ID", "患者编号"], "患者ID")
        full_clinical.loc[full_clinical[id_col].str.strip().isin(selected)].to_excel(
            staging / config.CLINICAL_NAME, index=False)
        chosen.to_csv(staging / "patch_patient_map.csv", index=False, encoding="utf-8-sig")
        patients[patients.patient_id.isin(selected)].to_csv(staging / "selected_patients.csv", index=False, encoding="utf-8-sig")
        (staging / "sampling_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        staging.rename(p["subset"])
    except Exception:
        print(f"未发布子集；临时数据保留于：{staging}")
        raise
    print(f"已复制完整患者数据到：{p['subset']}；原始数据未改动。")
    print("本阶段不创建训练/验证集。后续只从该子集患者生成一次患者级划分，两阶段复用。")


if __name__ == "__main__":
    main()
