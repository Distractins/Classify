"""核对原图并定点修复划分后的图片；不改变标签或集合，不跳过样本。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime
from PIL import Image
from utils import path_config as config


def decode(path):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.convert("RGB").load()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repair(project, dataset, relative, apply=False):
    project, dataset = Path(project).resolve(), Path(dataset).resolve()
    relative = Path(relative)
    if relative.is_absolute() or len(relative.parts) < 3 or ".." in relative.parts:
        raise ValueError("图片路径必须是 split/类别/文件名 的相对路径")
    split, cls = relative.parts[:2]
    if split not in ("train", "val", "test") or cls not in config.CLASSES:
        raise ValueError("无效集合或组织类别")
    destination = dataset / relative
    if not destination.resolve().is_relative_to(dataset):
        raise ValueError("图片路径超出数据集")
    source = config.paths(project)["raw"] / cls / Path(*relative.parts[2:])
    print(f"划分后图片：{destination}", flush=True)
    try:
        decode(destination)
    except Exception as exc:
        print(f"副本解码失败：{type(exc).__name__}: {exc}", flush=True)
    else:
        print("该图片当前能够完整解码，不修改。")
        return False
    print(f"检查原始图片：{source}", flush=True)
    try:
        decode(source)
    except Exception as exc:
        raise RuntimeError("原始图片也缺失或无法解码；请从原始压缩包恢复同名图片，不能用黑图替代。") from exc
    if source.stat().st_size != destination.stat().st_size:
        raise RuntimeError("原图可解码，但与副本大小不同；恢复会改变断点数据指纹。尚未修改文件，请保留此输出以便核查断点兼容性。")
    if sha256(source) == sha256(destination):
        raise RuntimeError("原图与副本字节完全一致却解码结果不同，请检查运行稳定性；不盲目覆盖。")
    if not apply:
        print("原图通过检查且大小相同。加 --apply 可备份并修复。")
        return False
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = project / "outputs/image_repairs" / stamp
    backup.mkdir(parents=True)
    shutil.copy2(destination, backup / (destination.name + ".bad"))
    temporary = destination.with_name(destination.name + ".repairing")
    shutil.copy2(source, temporary)
    decode(temporary)
    if sha256(temporary) != sha256(source):
        raise IOError("修复副本哈希校验失败，未覆盖原文件")
    temporary.replace(destination)
    (backup / "repair.json").write_text(json.dumps({"source": str(source),
        "destination": str(destination), "sha256": sha256(destination)}, indent=2), encoding="utf-8")
    print(f"修复完成；坏副本保存在 {backup}。标签、集合、字节大小未改变。")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(config.ROOT))
    parser.add_argument("--dataset", default="datasets/HMU-CRC-Hist550K-FULL")
    parser.add_argument("--image", required=True, help="例如 val/TUM/1432083_10080_4480_TUM.png")
    parser.add_argument("--apply", action="store_true", help="备份后用通过检查的原图修复")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = project / dataset
    repair(project, dataset, args.image, args.apply)
