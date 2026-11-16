"""探查八类 patch 和临床表，不抽样、不训练。"""
import argparse
import json
import os
import re
import warnings
import hashlib
import inspect
import sqlite3
import faulthandler
import sys
import traceback
from datetime import datetime
from contextlib import ExitStack, closing
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from utils import path_config as config


def list_class_images(folder):
    """用 os.scandir/os.walk 枚举，避免为整个目录树保留大量 Path 对象。"""
    names = []

    def fail(error):
        raise error  # os.walk 默认忽略部分目录访问异常；这里必须显式报错。

    for directory, subdirs, filenames in os.walk(folder, onerror=fail, followlinks=False):
        for name in subdirs:
            if os.path.islink(os.path.join(directory, name)):
                raise ValueError(f"不支持类别内的目录符号链接，请检查：{os.path.join(directory, name)}")
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in config.IMAGE_SUFFIXES:
                relative = os.path.relpath(os.path.join(directory, filename), folder)
                names.append(relative.replace(os.sep, "/"))
                if len(names) % 10000 == 0:
                    print(f"文件列表已读取 {len(names):,} 张……", flush=True)
    return sorted(names)


def scan_patch_map(raw, checkpoint_path=None, checkpoint_every=1000, restart=False):
    """分批提交扫描结果；采样脚本默认不使用缓存，仍独立完整复扫。"""
    if checkpoint_every < 1:
        raise ValueError("checkpoint-every 必须为正整数。")
    raw = Path(raw).resolve()
    records, seen = [], set()
    with ExitStack() as stack:
        database = None
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            os.makedirs(checkpoint_path.parent, exist_ok=True)
            database = stack.enter_context(closing(sqlite3.connect(checkpoint_path)))
            database.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            database.execute("CREATE TABLE IF NOT EXISTS manifests (cls TEXT PRIMARY KEY, digest TEXT NOT NULL)")
            database.execute("CREATE TABLE IF NOT EXISTS patches (cls TEXT, name TEXT, patient_id TEXT, resolved TEXT, PRIMARY KEY(cls, name))")
            signature = hashlib.sha256(json.dumps({
                "version": 1, "raw": str(raw), "classes": config.CLASSES,
                "extensions": sorted(config.IMAGE_SUFFIXES),
                "parser": inspect.getsource(config.parse_patient_id),
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            if restart:
                with database:
                    database.execute("DELETE FROM patches")
                    database.execute("DELETE FROM manifests")
                    database.execute("DELETE FROM metadata")
                print("已重置扫描断点，重新扫描；原始数据不变。")
            saved = database.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            if saved is not None and saved[0] != signature:
                raise ValueError("断点的数据根目录或解析规则已变化，请加 --restart-scan 重建扫描缓存。")
            with database:
                database.execute("INSERT OR REPLACE INTO metadata VALUES ('signature', ?)", (signature,))
            print(f"扫描断点：{checkpoint_path}（每 {checkpoint_every} 张保存）")

        for cls in config.CLASSES:
            folder = raw / cls
            if not folder.is_dir():
                raise FileNotFoundError(f"类别目录不存在：{folder}")
            print(f"正在读取 {cls} 文件列表……", flush=True)
            names = list_class_images(folder)
            print(f"{cls} 文件列表读取完成：{len(names):,} 张。", flush=True)
            cached = {}
            if database is not None:
                digest = hashlib.sha256(json.dumps(names, ensure_ascii=False).encode("utf-8")).hexdigest()
                manifest = database.execute("SELECT digest FROM manifests WHERE cls=?", (cls,)).fetchone()
                if manifest is not None and manifest[0] != digest:
                    raise ValueError(f"{cls} 文件列表与断点不一致，请加 --restart-scan 重建，避免遗漏图片。")
                with database:
                    database.execute("INSERT OR REPLACE INTO manifests VALUES (?, ?)", (cls, digest))
                cached = {name: (pid, resolved) for name, pid, resolved in database.execute(
                    "SELECT name, patient_id, resolved FROM patches WHERE cls=?", (cls,))}
                if not set(cached).issubset(names):
                    raise ValueError("断点记录与文件列表不一致，请加 --restart-scan 重建。")
                if cached:
                    print(f"{cls}：恢复 {len(cached):,}/{len(names):,} 张，继续剩余部分。", flush=True)
            # 缓存也必须纳入全部患者计数与重复源文件检查。
            for name in names:
                if name in cached:
                    pid, resolved = cached[name]
                    if resolved in seen:
                        raise ValueError(f"重复引用源图：{resolved}")
                    seen.add(resolved)
                    records.append((name, pid, cls))
            pending = []

            def save_pending():
                if database is not None and pending:
                    with database:
                        database.executemany("INSERT INTO patches VALUES (?, ?, ?, ?)", pending)
                    pending.clear()

            remaining = ((folder / name, name) for name in names if name not in cached)
            try:
                with tqdm(total=len(names), initial=len(cached), desc=f"扫描 {cls}", unit="file") as progress:
                    for image, name in remaining:
                        resolved = image.resolve()
                        try:
                            resolved.relative_to(folder.resolve())
                        except ValueError as error:
                            raise ValueError(f"图片链接越出类别目录：{image}") from error
                        if str(resolved) in seen:
                            raise ValueError(f"重复引用源图：{image}")
                        pid = config.parse_patient_id(image.name, expected_class=cls)
                        seen.add(str(resolved))
                        records.append((name, pid, cls))
                        if database is not None:
                            pending.append((cls, name, pid, str(resolved)))
                            if len(pending) >= checkpoint_every:
                                save_pending()
                        progress.update(1)
            except BaseException:
                # Ctrl+C 或文件名报错时，保存此前已成功解析的图片；异常图片不会记为完成。
                save_pending()
                raise
            save_pending()
    if not records:
        raise ValueError("未找到图片。")
    return pd.DataFrame(records, columns=["img_name", "patient_id", "cls"]).sort_values(
        ["cls", "img_name"]).reset_index(drop=True)


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


def read_clinical(path, id_column=None, msi_column=None, diagnostic_dir=None):
    if not path.is_file():
        raise FileNotFoundError(f"临床表不存在：{path}")
    columns = pd.read_excel(path, nrows=0).columns
    pid = find_column(columns, id_column, ["Patient", "PatientID", "patient_id", "患者ID", "患者编号"], "患者ID")
    aliases = ["MSI", "MSI status", "MSI-H/MSS", "msi_label", "MSI状态"]
    key = lambda value: re.sub(r"[\s_\-‑–/]+", "", str(value)).casefold()
    has_msi = any(key(col) in {key(alias) for alias in aliases} for col in columns)
    msi = find_column(columns, msi_column, aliases, "MSI") if msi_column or has_msi else None
    if msi is None:
        ihc = [col for col in columns if key(col) == "immunohistochemistry"]
        if len(ihc) == 1:
            msi = ihc[0]
    if msi is not None and key(msi) == "immunohistochemistry":
        warnings.warn(
            "从 immunohistochemistry 仅识别明确的 MSI-H/MSS；空值及 IHC 描述记为 UNKNOWN，"
            "不依据阳性比例推断 MSI。"
        )
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
    original_ids = table.patient_id.copy()
    table["patient_id"] = table.patient_id.str.strip()
    empty = table.patient_id.eq("")
    duplicate = ~empty & table.patient_id.duplicated(keep=False)
    if empty.any() or duplicate.any():
        # 导出完整问题行，Excel 行号 = DataFrame 索引 + 表头行 + 从 1 起算。
        full = pd.read_excel(path, dtype=str, keep_default_na=False)
        issues = pd.DataFrame({
            "excel_row": table.index + 2,
            "patient_id_raw": original_ids,
            "patient_id_normalized": table.patient_id,
            "issue": ["empty_id" if blank else "duplicate_id" if repeated else ""
                      for blank, repeated in zip(empty, duplicate)],
        }).join(full.add_prefix("source_"))
        issues = issues.loc[empty | duplicate]
        destination = Path(diagnostic_dir) if diagnostic_dir is not None else path.parent
        os.makedirs(destination, exist_ok=True)
        report_path = destination / "clinical_id_issues.csv"
        issues.to_csv(report_path, index=False, encoding="utf-8-sig")
        examples = table.loc[duplicate, "patient_id"].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"临床患者ID异常：空ID {int(empty.sum())} 行；"
            f"重复ID {table.loc[duplicate, 'patient_id'].nunique()} 个，"
            f"涉及 {int(duplicate.sum())} 行；示例 {examples}。"
            f"完整问题行及 Excel 行号已保存：{report_path}。"
            "未自动删除、合并或补零，请先核查报告。"
        )
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
    parser.add_argument("--images-only", action="store_true", help="仅统计图像，不读取临床表；MSI 状态均为 UNKNOWN")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="每 N 张图片提交一次扫描断点，默认自动恢复")
    parser.add_argument("--restart-scan", action="store_true", help="清空扫描断点并重新扫描，不删除原始数据")
    parser.add_argument("--check-images", choices=["none", "pil", "opencv"], default="none")
    parser.add_argument("--heatmap-max-patients", type=int, default=0, help="0 展示全部患者，否则仅绘制 patch 最多的前 N 位")
    args = parser.parse_args(argv)
    if args.heatmap_max_patients < 0:
        parser.error("heatmap-max-patients 不能为负。")
    if args.checkpoint_every < 1:
        parser.error("checkpoint-every 必须为正整数。")
    p = config.paths(args.root)
    config.prepare_plotting(args.root)
    os.makedirs(p["stat"], exist_ok=True)
    if args.images_only:
        clinical = pd.DataFrame(columns=["patient_id", "msi_status_raw", "msi_status"])
        print("仅图像探查：未读取临床表，不统计临床患者数量，不进行 MSI 分层。")
    else:
        clinical = read_clinical(p["clinical"], args.id_column, args.msi_column, p["stat"])
    mapping = scan_patch_map(p["raw"], p["scan_checkpoint"], args.checkpoint_every, args.restart_scan)
    print("八类扫描已全部完成，正在生成统计报告和绘图……", flush=True)
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
    summary = {"raw_root": str(p["raw"]), "patches": len(mapping), "clinical_read": not args.images_only,
               "patch_patients": int(mapping.patient_id.nunique()), "clinical_patients": len(clinical),
               "unmatched_patch_patients": int((~patients.clinical_matched).sum())}
    (p["stat"] / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n==== 数据探查摘要 ====\n" + counts.to_string())
    clinical_count = "未读取" if args.images_only else str(len(clinical))
    print(f"总 patch：{len(mapping):,}；图片患者：{summary['patch_patients']}；临床患者：{clinical_count}")
    print("MSI 患者分布：\n" + msi_counts.to_string())
    print("每患者 patch 数：\n" + patients.loc[patients.has_patches, "total_patches"].describe().to_string())
    if args.images_only:
        print("临床匹配未执行；报告中的 UNKNOWN 和未匹配标记不代表真实临床状态。")
    else:
        print(f"无临床匹配的图片患者：{summary['unmatched_patch_patients']}，请检查前导零及 ID。")
    print(f"统计与三张图已保存：{p['stat']}")


if __name__ == "__main__":
    # -X faulthandler 还能捕获脚本导入依赖阶段的原生崩溃；这里另存运行阶段日志。
    diagnostic_parser = argparse.ArgumentParser(add_help=False)
    diagnostic_parser.add_argument("--root")
    diagnostic_args, _ = diagnostic_parser.parse_known_args()
    log_directory = config.paths(diagnostic_args.root)["stat"]
    os.makedirs(log_directory, exist_ok=True)
    crash_path = log_directory / "scan_crash.log"
    with crash_path.open("a", encoding="utf-8") as crash_log:
        crash_log.write(f"\nSTART {datetime.now().isoformat()} python={sys.executable} version={sys.version}\n")
        crash_log.flush()
        faulthandler.enable(file=crash_log, all_threads=True)
        print(f"Python：{sys.executable}\n诊断日志：{crash_path}", flush=True)
        try:
            main()
        except BaseException:
            traceback.print_exc(file=crash_log)
            crash_log.flush()
            raise
        else:
            crash_log.write("COMPLETED\n")
            crash_log.flush()
        finally:
            faulthandler.disable()
