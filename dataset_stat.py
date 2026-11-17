"""扫描 PNG，统计八类数量和 WSI 编号；不读取临床表。"""
import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from utils import path_config as config

# ==================== CONFIG ====================
CONFIG = config.paths()  # 路径统一在 utils/path_config.py 修改
CLASSES = config.CLASSES
# =================================================


def scan_class(raw, cls):
    folder = Path(raw) / cls
    if not folder.is_dir():
        raise FileNotFoundError(f"类别目录不存在：{folder}")
    rows = []
    for current, dirs, files in os.walk(folder, followlinks=False):
        dirs.sort()
        for name in tqdm(sorted(files), desc=f"扫描 {cls}", unit="file"):
            if not name.lower().endswith(".png"):
                continue
            image = Path(current) / name
            if image.is_symlink() or not image.is_file():
                raise ValueError(f"不支持符号链接或异常图片：{image}")
            rows.append((image.relative_to(folder).as_posix(),
                         config.parse_slide_id(name, cls), cls))
    if not rows:
        raise ValueError(f"类别目录没有 PNG：{folder}")
    return pd.DataFrame(rows, columns=["img_name", "slideID", "cls"])


def scan_dataset(raw):
    return pd.concat([scan_class(raw, cls) for cls in CLASSES], ignore_index=True)


def save_csv(frame, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(target)


def read_mapping(path):
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    return validate_mapping(data, path)


def validate_mapping(data, source):
    if list(data.columns) != ["img_name", "slideID", "cls"] or data.empty:
        raise ValueError(f"映射表为空或字段错误：{source}")
    if data.duplicated(["cls", "img_name"]).any():
        raise ValueError(f"存在重复图片记录：{source}")
    for row in data.itertuples(index=False):
        relative = Path(row.img_name)
        if (row.cls not in CLASSES or relative.is_absolute()
                or relative.drive or ".." in relative.parts
                or relative.suffix.lower() != ".png"
                or config.parse_slide_id(relative.name, row.cls) != row.slideID):
            raise ValueError(f"映射记录不合法：{row}")
    return data.sort_values(["cls", "img_name"]).reset_index(drop=True)


def import_legacy_mapping(source, cache):
    """仅接受旧 ID 等于文件名前缀的完整八类映射，不猜测或改写 ID。"""
    data = pd.read_csv(source, dtype=str, keep_default_na=False)
    if list(data.columns) != ["img_name", "patient_id", "cls"]:
        raise ValueError("旧映射必须为 img_name,patient_id,cls 三列")
    data = validate_mapping(data.rename(columns={"patient_id": "slideID"}), source)
    if set(data["cls"]) != set(CLASSES):
        raise ValueError("旧映射缺少类别，不能作为完整八类统计导入")
    for cls in CLASSES:
        save_csv(data.loc[data["cls"].eq(cls)], cache / f"{cls}.csv")
    print(f"已核对并导入旧映射：{len(data):,} 张 PNG / {data['slideID'].nunique()} 个 slideID。")
    print("这是旧文件清单快照；未重新核对磁盘增删，采样前仍会检查当前原图清单。")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="临时覆盖根目录")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--scan-class", choices=CLASSES, help="只扫描并缓存一个类别")
    modes.add_argument("--summarize", action="store_true", help="仅汇总八类缓存")
    modes.add_argument("--import-legacy-map", metavar="CSV", help="核对并导入旧 patch_patient_map.csv，然后汇总")
    args = parser.parse_args(argv)
    p = config.paths(args.root)
    cache = p["stat"] / "slide_scan_cache"
    cache.mkdir(parents=True, exist_ok=True)
    if args.import_legacy_map:
        source = Path(args.import_legacy_map)
        if not source.is_absolute():
            source = p["root"] / source
        import_legacy_mapping(source, cache)
    elif not args.summarize:
        for cls in ([args.scan_class] if args.scan_class else CLASSES):
            save_csv(scan_class(p["raw"], cls), cache / f"{cls}.csv")
        if args.scan_class:
            print(f"{args.scan_class} 缓存完成；八类完成后执行 --summarize。")
            return
    missing = [cls for cls in CLASSES if not (cache / f"{cls}.csv").is_file()]
    if missing:
        raise ValueError(f"尚未扫描类别：{missing}")
    frames = []
    for cls in CLASSES:
        frame = read_mapping(cache / f"{cls}.csv")
        if not frame["cls"].eq(cls).all():
            raise ValueError(f"缓存类别不一致：{cls}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    save_csv(data, p["stat"] / "mapping.csv")
    counts = data.groupby("cls").size().reindex(CLASSES, fill_value=0)
    save_csv(counts.rename("patch_count").rename_axis("cls").reset_index(),
             p["stat"] / "class_patch_counts.csv")
    slides = pd.crosstab(data["slideID"], data["cls"]).reindex(columns=CLASSES, fill_value=0)
    slides["total"] = slides.sum(axis=1)
    save_csv(slides.reset_index(), p["stat"] / "slide_patch_counts.csv")
    config.prepare_plotting(args.root)
    import matplotlib.pyplot as plt
    from utils.plot_style import configure_chinese_font
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot.bar(ax=ax, color="steelblue", rot=0)
    ax.set(xlabel="组织类别", ylabel="Patch 数量", title="HMU-CRC-Hist550K 各组织类别样本数量")
    fig.tight_layout()
    fig.savefig(p["stat"] / "class_patch_counts.png", dpi=180)
    plt.close(fig)
    print("\n=== 数据统计（不读取临床表） ===")
    print(counts.to_string())
    print(f"总 patch：{len(data):,}；唯一 slideID：{len(slides):,}")
    print("每张 WSI 的 patch 数：\n" + slides["total"].describe().to_string())
    print(f"统计保存至：{p['stat']}")


if __name__ == "__main__":
    main()
