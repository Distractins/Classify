"""探查八类 patch 和临床表，不抽样、不训练。"""
import argparse
import json
import os
import re
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from utils import path_config as config


def scan_patch_map(raw):
    """完整扫描；类别内相对路径写入 img_name，支持嵌套子目录。"""
    records, seen = [], set()
    for cls in config.CLASSES:
        folder = raw / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"类别目录不存在：{folder}")
        for image in tqdm(sorted(folder.rglob("*")), desc=f"扫描 {cls}", unit="file"):
            if not image.is_file() or image.suffix.lower() not in config.IMAGE_SUFFIXES:
                continue
            resolved = image.resolve()
            try:
                resolved.relative_to(folder.resolve())
            except ValueError as error:
                raise ValueError(f"图片链接越出类别目录：{image}") from error
            if resolved in seen:
                raise ValueError(f"重复引用源图：{image}")
            seen.add(resolved)
            records.append((image.relative_to(folder).as_posix(),
                            config.parse_patient_id(image.name), cls))
    if not records:
        raise ValueError("未找到图片。")
    return pd.DataFrame(records, columns=["img_name", "patient_id", "cls"])


def find_column(columns, explicit, aliases, name):
    if explicit:
        if explicit not in columns:
            raise ValueError(f"不存在{name}列 {explicit!r}，实际列：{list(columns)}")
        return explicit
    key = lambda value: re.sub(r"[\s_\-‑–/]+", "", str(value)).casefold()
    matched = [col for col in columns if key(col) in {key(a) for a in aliases}]
    if len(matched) != 1:
        raise ValueError(f"无法唯一识别{name}列，请用命令行指定；实际列：{list(columns)}")
    return matched[0]


def normalize_msi(value):
    key = re.sub(r"[\s_\-‑–]+", "", str(value)).upper()
    return "MSI-H" if key == "MSIH" else "MSS" if key == "MSS" else "UNKNOWN"


def read_clinical(path, id_column=None, msi_column=None):
    if not path.is_file():
        raise FileNotFoundError(f"临床表不存在：{path}")
    columns = pd.read_excel(path, nrows=0).columns
    pid = find_column(columns, id_column, ["Patient", "PatientID", "patient_id", "患者ID", "患者编号"], "患者ID")
    aliases = ["MSI", "MSI status", "MSI-H/MSS", "msi_label", "MSI状态"]
    key = lambda value: re.sub(r"[\s_\-‑–/]+", "", str(value)).casefold()
    has_msi = any(key(col) in {key(alias) for alias in aliases} for col in columns)
    msi = find_column(columns, msi_column, aliases, "MSI") if msi_column or has_msi else None
    selected = [pid, msi] if msi else [pid]
    table = pd.read_excel(path, usecols=selected, dtype=str, keep_default_na=False)
    table = table.rename(columns={pid: "patient_id"})
    if msi:
        table = table.rename(columns={msi: "msi_status_raw"})
    else:
        table["msi_status_raw"] = ""
        warnings.warn(
            "未找到明确的 MSI 标签列，全部记为 UNKNOWN；仍可统计组织和患者数据，"
            "但无法平衡 MSI-H/MSS。不会自动将 immunohistochemistry 转成 MSI。"
        )
    table["patient_id"] = table.patient_id.str.strip()
    if table.patient_id.eq("").any() or table.patient_id.duplicated().any():
        raise ValueError("临床患者ID为空或重复；请先核查，不自动转换数值或补零。")
    table["msi_status"] = table.msi_status_raw.map(normalize_msi)
    return table


def patient_counts(mapping):
    counts = pd.crosstab(mapping.patient_id, mapping.cls).reindex(columns=config.CLASSES, fill_value=0)
    counts.columns.name = None
    counts["total_patches"] = counts.sum(axis=1)
    return counts.reset_index()


def validate_images(mapping, raw, backend):
    """可选读取全部图片检查解码；默认只统计路径，避免额外读取 50G 图像。"""
    bad = []
    if backend == "pil":
        from PIL import Image
    else:
        import cv2
        import numpy as np
    for row in tqdm(mapping.itertuples(index=False), total=len(mapping), desc="验证图片"):
        image = raw / row.cls / row.img_name
        try:
            if backend == "pil":
                with Image.open(image) as opened:
                    opened.load()
            elif cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR) is None:
                raise ValueError("OpenCV 解码失败")
        except Exception as error:
            bad.append((row.img_name, row.patient_id, row.cls, str(error)))
    return pd.DataFrame(bad, columns=["img_name", "patient_id", "cls", "error"])


def save_plots(counts, patients, stat, heatmap_max_patients):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values)
    ax.set(xlabel="Tissue class", ylabel="Patch count", title="All patches by tissue class")
    fig.tight_layout()
    fig.savefig(stat / "class_patch_counts.png", dpi=200)
    plt.close(fig)

    available = patients[patients.total_patches > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(available.total_patches, bins=min(40, len(available)), edgecolor="white")
    ax.set(xlabel="Patches per patient", ylabel="Patient count", title="Patient patch distribution")
    fig.tight_layout()
    fig.savefig(stat / "patient_patch_histogram.png", dpi=200)
    plt.close(fig)

    shown = available.sort_values(["total_patches", "patient_id"], ascending=[False, True])
    if heatmap_max_patients:
        shown = shown.head(heatmap_max_patients)
    fig, ax = plt.subplots(figsize=(10, max(5, min(24, len(shown) * 0.2))))
    image = ax.imshow(shown[list(config.CLASSES)].to_numpy(), aspect="auto", cmap="YlOrRd")
    step = max(1, len(shown) // 40)
    ax.set_xticks(range(8))
    ax.set_xticklabels(config.CLASSES)
    ax.set_yticks(list(range(0, len(shown), step)))
    ax.set_yticklabels(shown.patient_id.iloc[::step])
    ax.set(xlabel="Tissue class", ylabel="Patient ID",
           title=f"Patient / tissue patch counts (shown {len(shown)}/{len(available)})")
    fig.colorbar(image, ax=ax, label="Patch count")
    fig.tight_layout()
    fig.savefig(stat / "patient_class_heatmap.png", dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="临时覆盖 path_config.ROOT")
    parser.add_argument("--id-column")
    parser.add_argument("--msi-column")
    parser.add_argument("--check-images", choices=["none", "pil", "opencv"], default="none")
    parser.add_argument("--heatmap-max-patients", type=int, default=0, help="0 展示全部患者，否则仅绘制 patch 最多的前 N 位")
    args = parser.parse_args(argv)
    if args.heatmap_max_patients < 0:
        parser.error("heatmap-max-patients 不能为负。")
    p = config.paths(args.root)
    config.prepare_plotting(args.root)
    os.makedirs(p["stat"], exist_ok=True)
    clinical = read_clinical(p["clinical"], args.id_column, args.msi_column)
    mapping = scan_patch_map(p["raw"])
    counts = mapping.cls.value_counts().reindex(config.CLASSES, fill_value=0)
    patients = patient_counts(mapping).merge(clinical, on="patient_id", how="outer", validate="one_to_one", indicator=True)
    patients["clinical_matched"] = patients["_merge"].ne("left_only")
    patients["has_patches"] = patients["_merge"].ne("right_only")
    patients = patients.drop(columns="_merge")
    patients[list(config.CLASSES) + ["total_patches"]] = patients[list(config.CLASSES) + ["total_patches"]].fillna(0).astype(int)
    patients["msi_status"] = patients.msi_status.fillna("UNKNOWN")
    patients["msi_status_raw"] = patients.msi_status_raw.fillna("")
    patients = patients.sort_values("patient_id").reset_index(drop=True)
    msi_counts = pd.DataFrame({
        "clinical_patients": clinical.msi_status.value_counts(),
        "patch_patients": patients.loc[patients.has_patches, "msi_status"].value_counts(),
    }).reindex(["MSI-H", "MSS", "UNKNOWN"]).fillna(0).astype(int)
    mapping.to_csv(p["stat"] / "patch_patient_map.csv", index=False, encoding="utf-8-sig")
    counts.rename_axis("cls").rename("patch_count").to_csv(p["stat"] / "class_patch_counts.csv", encoding="utf-8-sig")
    patients.to_csv(p["stat"] / "patient_patch_stats.csv", index=False, encoding="utf-8-sig")
    msi_counts.rename_axis("msi_status").to_csv(p["stat"] / "msi_patient_counts.csv", encoding="utf-8-sig")
    patients.loc[~patients.clinical_matched | ~patients.has_patches].to_csv(
        p["stat"] / "unmatched_patients.csv", index=False, encoding="utf-8-sig")
    if args.check_images != "none":
        bad = validate_images(mapping, p["raw"], args.check_images)
        bad.to_csv(p["stat"] / "invalid_images.csv", index=False, encoding="utf-8-sig")
        if not bad.empty:
            raise ValueError(f"发现 {len(bad)} 张不可解码图片，见 invalid_images.csv；未丢弃任何 patch，请修复后重跑。")
    save_plots(counts, patients, p["stat"], args.heatmap_max_patients)
    summary = {"raw_root": str(p["raw"]), "patches": len(mapping),
               "patch_patients": int(mapping.patient_id.nunique()), "clinical_patients": len(clinical),
               "unmatched_patch_patients": int((~patients.clinical_matched).sum())}
    (p["stat"] / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n==== 数据探查摘要 ====\n" + counts.to_string())
    print(f"总 patch：{len(mapping):,}；图片患者：{summary['patch_patients']}；临床患者：{len(clinical)}")
    print("MSI 患者分布：\n" + msi_counts.to_string())
    print("每患者 patch 数：\n" + patients.loc[patients.has_patches, "total_patches"].describe().to_string())
    print(f"无临床匹配的图片患者：{summary['unmatched_patch_patients']}，请检查前导零及 ID。")
    print(f"统计与三张图已保存：{p['stat']}")


if __name__ == "__main__":
    main()
