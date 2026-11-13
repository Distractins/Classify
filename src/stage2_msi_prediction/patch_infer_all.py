"""加载阶段一权重，对全部 patch 推理并输出患者关联结果。"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import (CLASSES, CSV_DIR, IMAGE_EXTENSIONS, PatchPatientResolver,
                             add_data_arguments, add_mapping_arguments, clinical_path,
                             load_patient_split, project_path, read_clinical, relative_path)

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


def load_patch_model(weights, model_name, device, train_ids=None, val_ids=None):
    """兼容原脚本的包装器 state_dict，以及带元数据的 checkpoint；不下载预训练权重。"""
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:  # 兼容不支持 weights_only 的旧版 PyTorch。
        checkpoint = torch.load(weights, map_location="cpu")
    if "state_dict" in checkpoint or "model_state_dict" in checkpoint:
        if "class_names" in checkpoint and list(checkpoint["class_names"]) != list(CLASSES):
            raise ValueError("checkpoint 类别顺序与 README 不符，禁止错误映射组织类别。")
        if "train_patient_ids" in checkpoint and train_ids is not None:
            if (set(checkpoint["train_patient_ids"]) != set(train_ids)
                    or set(checkpoint["val_patient_ids"]) != set(val_ids)):
                raise ValueError("权重记录的患者划分与当前名单不一致。")
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    else:
        state = checkpoint
        print("提示：旧版纯 state_dict 不含划分来源；请确认权重已按当前患者名单重新训练。")
    state = {key[len("module."):] if key.startswith("module.") else key: value
             for key, value in state.items()}
    prefix = {"efficientnet": "efficientnet.", "resnet101": "resnet.", "vit": "vit."}[model_name]
    # HF ViT 自身也含 vit. 前缀，只有外层包装器含 vit.vit. 时才去掉一层。
    wrapped = (any(key.startswith("vit.vit.") for key in state) if model_name == "vit"
               else all(key.startswith(prefix) for key in state))
    if wrapped:
        state = {key[len(prefix):] if key.startswith(prefix) else key: value
                 for key, value in state.items()}
    if model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
    elif model_name == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    else:
        from transformers import ViTConfig, ViTForImageClassification
        model = ViTForImageClassification(ViTConfig(num_labels=len(CLASSES)))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    # CNN 与原阶段一 Resize + ToTensor 一致；ViT 对应 HF 默认 mean/std=0.5。
    steps = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if model_name == "vit":
        steps.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
    return model, transforms.Compose(steps)


class PatchDataset(Dataset):
    def __init__(self, dataset_root, resolver, transform):
        self.samples = []
        self.transform = transform
        for cls in CLASSES:
            folder = dataset_root / cls
            if not folder.is_dir():
                raise FileNotFoundError(f"缺少组织目录: {folder}")
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, resolver(path)))
        if not self.samples:
            raise ValueError("数据目录中没有 patch 图像。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, pid = self.samples[index]
        # 损坏图片直接报错，禁止静默漏掉 patch 而改变患者组织占比。
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, pid, relative_path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_mapping_arguments(parser)
    parser.add_argument("--model", choices=["efficientnet", "resnet101", "vit"], default="efficientnet")
    parser.add_argument("--weights", help="阶段一 pth 的项目相对路径")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 默认使用主进程加载")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size 必须为正，num-workers 不能为负。")
    clinical = read_clinical(clinical_path(args.dataset, args.clinical), args.id_column)
    train, val = load_patient_split(clinical.patient_id)
    resolver = PatchPatientResolver(clinical.patient_id, args.patch_map, args.patient_regex)
    weights = project_path(args.weights or f"outputs/weights/stage1_8cls/{args.model}_best.pth")
    if not weights.is_file():
        raise FileNotFoundError(f"阶段一权重不存在: {relative_path(weights)}")
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    model, transform = load_patch_model(weights, args.model, device, train, val)
    dataset = PatchDataset(project_path(args.dataset), resolver, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    output = project_path(CSV_DIR / "patch_infer_result.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    # 完整推理成功后才替换正式文件，避免中断后误用部分 patch 统计。
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "image_file_path", "predict_class"])
        with torch.inference_mode():
            for images, patient_ids, paths in tqdm(loader, desc="Patch inference"):
                result = model(images.to(device, non_blocking=True))
                logits = result.logits if hasattr(result, "logits") else result
                predictions = logits.argmax(dim=1).cpu().tolist()
                writer.writerows((pid, path, CLASSES[index])
                                 for pid, path, index in zip(patient_ids, paths, predictions))
    temporary.replace(output)
    print(f"已推理全部 {len(dataset)} 张 patch: {relative_path(output)}")


if __name__ == "__main__":
    main()
