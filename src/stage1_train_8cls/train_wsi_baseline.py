"""EfficientNet-B0：复用 WSI 目录划分，不读取临床表。"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import path_config as config
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ==================== CONFIG ====================
CONFIG = config.paths()
CLASSES = config.CLASSES
SEED = config.SEED
# =================================================


def atomic_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # 仅加载自己生成或可信的本地文件。
    return torch.load(path, map_location="cpu", weights_only=False)


def load_data(root):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    evaluation = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    training = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.ToTensor(), normalize,
    ])
    result, groups, manifest = {}, {}, []
    root_string = os.path.abspath(os.fspath(root))
    for split in ("train", "val", "test"):
        print(f"{split}: 正在枚举 PNG 文件……", flush=True)
        result[split] = datasets.ImageFolder(
            root / split, transform=training if split == "train" else evaluation,
            is_valid_file=lambda name: name.lower().endswith(".png"))
        if tuple(result[split].classes) != CLASSES:
            raise ValueError(f"{split} 必须包含固定八类目录")
        groups[split] = set()
        print(f"{split}: 枚举完成，开始核对 {len(result[split])} 张图片的 slideID……", flush=True)
        for index, (filename, label) in enumerate(result[split].samples, 1):
            relative = os.path.relpath(filename, root_string).replace("\\", "/")
            if relative.startswith("../") or not relative.startswith(split + "/"):
                raise ValueError(f"图片路径超出指定集合：{filename}")
            groups[split].add(config.parse_slide_id(relative.rsplit("/", 1)[-1], CLASSES[label]))
            manifest.append((relative, label))
            if index % 10000 == 0:
                print(f"{split}: 已核对 {index:,}/{len(result[split]):,}", flush=True)
        print(f"{split}: {len(result[split])} patches / {len(groups[split])} slides", flush=True)
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("检测到 slideID 跨集合，拒绝训练")
    return result, manifest


def run_epoch(model, loader, device, optimizer=None, limit=0):
    model.train(optimizer is not None)
    total, loss_sum, correct = 0, 0.0, 0
    matrix = torch.zeros((8, 8), dtype=torch.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for step, (images, labels) in enumerate(tqdm(loader, desc="train" if optimizer else "eval")):
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(1)
            total += labels.numel()
            correct += (predictions == labels).sum().item()
            loss_sum += loss.item() * labels.numel()
            matrix += torch.bincount((labels * 8 + predictions).cpu(), minlength=64).reshape(8, 8)
            if limit and step + 1 >= limit:
                break
    f1 = 2 * matrix.diag().float() / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "macro_f1": f1.mean().item()}, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="训练和验证各两批，不保存正式权重，不使用 test")
    parser.add_argument("--evaluate-test", action="store_true", help="仅用最佳权重评估 test；调参期间不要运行")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs、batch-size、lr 必须为正数")
    if sum((args.resume, args.smoke_test, args.evaluate_test)) > 1:
        parser.error("resume、smoke-test、evaluate-test 不能同时使用")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    p = config.paths(args.root)
    data, manifest = load_data(p["paper"])
    output = p["root"] / "outputs/weights/stage1_8cls/efficientnet_wsi"
    last, best_path = output / "last.pth", output / "best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    model = models.efficientnet_b0(weights=None)
    if not args.resume and not args.evaluate_test:
        candidates = [p["root"] / "pretrained" / name for name in (
            "efficientnet_b0_rwightman-3dd342df.pth", "efficientnet_b0_rwightman-7f5810bc.pth")]
        weight = next((path for path in candidates if path.is_file()), None)
        if weight is None:
            raise FileNotFoundError(f"请放置本地 ImageNet 权重：{candidates[0]}")
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best = 0, -1.0
    if args.resume or args.evaluate_test:
        state = load_checkpoint(best_path if args.evaluate_test else last)
        if state["manifest"] != manifest or state["classes"] != list(CLASSES):
            raise ValueError("当前文件划分与权重记录不一致")
        model.load_state_dict(state["model"])
        if args.resume:
            if state["batch_size"] != args.batch_size or state["lr"] != args.lr:
                raise ValueError("恢复时 batch-size 和 lr 必须与保存时一致")
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            if device.type == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            start, best = state["epoch"], state["best"]
    elif not args.smoke_test and (last.exists() or best_path.exists()):
        raise FileExistsError("已有基线权重，请用 --resume，或先手动备份旧实验")
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=split == "train", num_workers=0)
               for split, dataset in data.items()}
    if args.evaluate_test:
        scores, matrix = run_epoch(model, loaders["test"], device)
        report = p["root"] / "outputs/figures/stage1_8cls/efficientnet_wsi"
        report.mkdir(parents=True, exist_ok=True)
        with (report / "test_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scores))
            writer.writeheader()
            writer.writerow(scores)
        with (report / "test_confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true/pred"] + list(CLASSES))
            writer.writerows([cls] + row for cls, row in zip(CLASSES, matrix.tolist()))
        print("test:", scores)
        return
    for epoch in range(start, start + 1 if args.smoke_test else args.epochs):
        train, _ = run_epoch(model, loaders["train"], device, optimizer, 2 if args.smoke_test else 0)
        val, _ = run_epoch(model, loaders["val"], device, limit=2 if args.smoke_test else 0)
        print(f"epoch={epoch + 1} train={train} val={val}", flush=True)
        if args.smoke_test:
            print("小批次试跑完成；未保存正式权重，未评估 test。")
            return
        improved = val["macro_f1"] > best
        best = max(best, val["macro_f1"])
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1, "best": best, "classes": list(CLASSES),
                 "manifest": manifest, "batch_size": args.batch_size, "lr": args.lr,
                 "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                 "train_metrics": train, "val_metrics": val}
        output.mkdir(parents=True, exist_ok=True)
        if improved:
            atomic_save(state, best_path)
        atomic_save(state, last)
    print("训练完成。最佳模型按 val macro-F1 选择；最终评估请单独运行 --evaluate-test。")


if __name__ == "__main__":
    main()
