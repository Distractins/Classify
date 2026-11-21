"""WSI-disjoint eight-class training with local weights and per-model experiments."""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from utils import path_config as config
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from src.experiment_reports import save_history, save_evaluation

# ==================== CONFIG ====================
# Preserve the class order used by the completed server baseline.
CLASSES = ("ADI", "LYM", "DEB", "MUC", "MUS", "NORM", "STR", "TUM")
SEED = 42
MODEL_NAMES = ("efficientnet_b0", "resnet50", "resnet101", "uni")
ENHANCERS = ("none", "morph", "morph_single_scale", "morph_no_gate",
             "prism", "prism_no_local", "prism_no_stats", "prism_no_pool")
# =================================================


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Load only trusted checkpoints; optimizer/RNG state requires full loading.
    return torch.load(path, map_location="cpu", weights_only=False)


class PatchDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.classes = list(CLASSES)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        try:
            with Image.open(filename) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot decode image; refusing to replace it with a black image: {filename}") from exc
        return self.transform(image), label


def load_data(root):
    root_string = os.path.abspath(os.fspath(root))
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    evaluation = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    training = transforms.Compose([transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(), norm])
    result, groups, manifest = {}, {}, []
    for split in ("train", "val", "test"):
        samples, groups[split] = [], set()
        for label, cls in enumerate(CLASSES):
            folder = os.path.join(root_string, split, cls)
            if not os.path.isdir(folder) or os.path.islink(folder):
                raise ValueError(f"Missing class directory or symbolic link: {folder}")
            before = len(samples)
            for current, dirs, files in os.walk(folder, followlinks=False):
                dirs.sort()
                if any(os.path.islink(os.path.join(current, d)) for d in dirs):
                    raise ValueError(f"Symbolic link directory is not supported: {current}")
                for name in sorted(files):
                    if not name.lower().endswith(".png"):
                        continue
                    filename = os.path.join(current, name)
                    if os.path.islink(filename):
                        raise ValueError(f"Symbolic link image is not supported: {filename}")
                    slide = config.parse_slide_id(name, cls)
                    groups[split].add(slide)
                    relative = os.path.relpath(filename, root_string).replace("\\", "/")
                    manifest.append((relative, label, os.path.getsize(filename)))
                    samples.append((filename, label))
            if len(samples) == before:
                raise ValueError(f"No PNG images in class directory: {folder}")
        result[split] = PatchDataset(samples, training if split == "train" else evaluation)
        print(f"{split}: {len(samples)} patches / {len(groups[split])} slides", flush=True)
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("slideID overlap across train/val/test; refusing to train")
    return result, manifest


def build_model(name, project, pretrained=True, weight_override=None, uni_mode="finetune",
                hidden_dim=256, dropout=0.1, ortho_weight=0.01, enhancer="none"):
    if enhancer != "none" and (name != "efficientnet_b0" or uni_mode != "finetune"):
        raise ValueError("CNN enhancers require efficientnet_b0 with default uni-mode")
    if enhancer not in ENHANCERS:
        raise ValueError(f"Unsupported enhancer: {enhancer}")
    if uni_mode not in ("finetune", "linear", "mlp"):
        raise ValueError("Unsupported UNI mode; use finetune, linear or mlp")
    if name == "uni":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("UNI requires timm; CNN models do not") from exc
        model = timm.create_model("vit_large_patch16_224", pretrained=False,
                                  img_size=224, patch_size=16, init_values=1.0,
                                  num_classes=0, dynamic_img_size=True)
        path = project / "pretrained/UNI/pytorch_model.bin"
    else:
        model = getattr(models, name)(weights=None)
        filenames = {"efficientnet_b0": "efficientnet_b0_rwightman-3dd342df.pth",
                     "resnet50": "resnet50-0676ba61.pth", "resnet101": "resnet101-63fe2227.pth"}
        path = project / "pretrained" / filenames[name]
    if pretrained:
        path = Path(weight_override) if weight_override else path
        if not path.is_file():
            raise FileNotFoundError(f"Local pretrained weights not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    if enhancer.startswith("morph"):
        from network.morph import EfficientNetMORPH
        variant = {"morph": "full", "morph_single_scale": "single_scale", "morph_no_gate": "no_gate"}[enhancer]
        return EfficientNetMORPH(model, variant)
    if enhancer.startswith("prism"):
        from network.prism import EfficientNetPRISM
        variant = "full" if enhancer == "prism" else enhancer.removeprefix("prism_")
        return EfficientNetPRISM(model, variant)
    if name == "uni":
        if uni_mode != "finetune":
            from network.frozen_uni import FrozenUNIClassifier
            return FrozenUNIClassifier(model, uni_mode, hidden_dim, dropout, ortho_weight)
        model.reset_classifier(8)
    elif name == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    else:
        model.fc = nn.Linear(model.fc.in_features, 8)
    return model


def run_epoch(model, loader, device, optimizer=None, limit=0, collect_predictions=False,
              progress=None, checkpoint_callback=None, save_every=200):
    model.train(optimizer is not None)
    total, loss_sum, correct = 0, 0.0, 0
    matrix = torch.zeros((8, 8), dtype=torch.int64)
    batch_cursor = 0
    if progress:
        total, loss_sum, correct = progress["total"], progress["loss_sum"], progress["correct"]
        matrix = progress["matrix"].clone()
        batch_cursor = progress["batch_cursor"]
    targets, probabilities = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for step, (images, labels) in enumerate(tqdm(loader, desc="train" if optimizer else "eval")):
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            regularizer = None
            if isinstance(logits, tuple):
                logits, regularizer = logits
            loss = nn.functional.cross_entropy(logits, labels)
            if regularizer is not None:
                loss = loss + regularizer
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; refusing to save this epoch")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(1)
            if collect_predictions:
                targets.extend(labels.cpu().tolist())
                probabilities.extend(torch.softmax(logits.detach(), dim=1).cpu().tolist())
            total += labels.numel()
            correct += (predictions == labels).sum().item()
            loss_sum += loss.item() * labels.numel()
            matrix += torch.bincount((labels * 8 + predictions).cpu(), minlength=64).reshape(8, 8)
            if checkpoint_callback is not None:
                current = {"total": total, "loss_sum": loss_sum, "correct": correct,
                           "matrix": matrix.clone(), "batch_cursor": batch_cursor + step + 1}
                if (step + 1 == len(loader)) or (save_every and current["batch_cursor"] % save_every == 0):
                    checkpoint_callback(current)
            if limit and step + 1 >= limit:
                break
    f1 = 2 * matrix.diag().float() / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)
    scores = {"loss": loss_sum / total, "accuracy": correct / total,
              "macro_f1": f1.mean().item()}
    if collect_predictions:
        return scores, matrix, (targets, probabilities)
    return scores, matrix


def safe_name(value):
    if not value or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in value):
        raise argparse.ArgumentTypeError("Run names allow only ASCII letters, digits, underscore and hyphen")
    return value


def exclude_validation_images(data, dataset, requested):
    """Explicitly exclude only confirmed undecodable validation images, preserving source files."""
    names = sorted(set(name.replace("\\", "/") for name in requested))
    indexed = {os.path.relpath(filename, dataset).replace("\\", "/"): (filename, label)
               for filename, label in data["val"].samples}
    for name in names:
        if not name.startswith("val/") or name not in indexed:
            raise ValueError(f"Exclusion must name an existing validation image: {name}")
        try:
            with Image.open(indexed[name][0]) as image:
                image.convert("RGB").load()
        except (OSError, ValueError, SyntaxError):
            pass
        else:
            raise ValueError(f"Image is decodable; refusing to exclude it as corrupt: {name}")
    remaining = [sample for name, sample in indexed.items() if name not in names]
    if names:
        if {label for _, label in remaining} != set(range(8)):
            raise ValueError("Exclusions would remove a validation class")
        data["val"].samples = remaining
        print(f"Explicit validation exclusions: {names}; remaining={len(remaining):,}", flush=True)
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--dataset", dest="dataset", help="PAPER dataset directory, matching the server CLI")
    parser.add_argument("--project-root", default=str(PROJECT), help="Project directory containing pretrained and outputs")
    parser.add_argument("--model", choices=MODEL_NAMES, default="efficientnet_b0")
    parser.add_argument("--enhancer", choices=ENHANCERS, default="none",
                        help="Train EfficientNet-B0 with an optional enhancer; default preserves original baseline")
    parser.add_argument("--uni-mode", choices=("finetune", "linear", "mlp"), default="finetune")
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--ortho-weight", type=float, default=0.01)
    parser.add_argument("--run-name", type=safe_name, default="baseline")
    parser.add_argument("--pretrained", help="Local pretrained file, relative to project directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every-batches", type=int, default=200,
                        help="Save last.pth every N completed training batches; 0 means epoch checkpoints only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--legacy-weights", help="Evaluate original server best.pth with ADI,LYM,DEB,... class order")
    parser.add_argument("--exclude-val-image", action="append", default=[],
                        help="Explicit corrupt validation image relative path; repeat for multiple images, keep files on disk")
    args = parser.parse_args(argv)
    if args.enhancer != "none" and (args.model != "efficientnet_b0" or args.uni_mode != "finetune"):
        parser.error("CNN enhancers require --model efficientnet_b0 and default --uni-mode")
    if args.model != "uni" and args.uni_mode != "finetune":
        parser.error("Frozen head modes are available only for UNI")
    if args.head_hidden_dim < 1 or not 0 <= args.head_dropout < 1 or args.ortho_weight < 0:
        parser.error("Invalid head dimensions, dropout or regularization weight")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0 or args.weight_decay < 0:
        parser.error("epochs, batch-size and lr must be positive; workers must be nonnegative")
    if sum((args.resume, args.smoke_test, args.evaluate_test)) > 1:
        parser.error("resume, smoke-test and evaluate-test are mutually exclusive")
    if args.legacy_weights and (not args.evaluate_test or args.model != "efficientnet_b0"):
        parser.error("legacy-weights requires efficientnet_b0 and evaluate-test")
    if args.save_every_batches < 0 or (args.save_every_batches and args.workers != 0):
        parser.error("Batch resume requires workers=0; use save-every-batches=0 for multi-worker epoch resume")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    project = Path(args.project_root).expanduser().resolve()
    dataset = Path(args.dataset).expanduser() if args.dataset else config.paths(project)["paper"]
    if not dataset.is_absolute():
        dataset = project / dataset
    data, manifest = load_data(dataset)
    fingerprint = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    exclusions = exclude_validation_images(data, dataset, args.exclude_val_image)
    experiment_model = args.model if args.uni_mode == "finetune" else "uni_frozen_" + args.uni_mode
    if args.enhancer != "none":
        experiment_model += "_" + args.enhancer
    output = project / "outputs/experiments" / experiment_model / args.run_name
    weights, reports = output / "weights", output / "reports"
    last, best_path = weights / "last.pth", weights / "best.pth"
    if not (args.resume or args.smoke_test or args.evaluate_test) and output.exists():
        raise FileExistsError(f"Experiment exists; choose another run-name or resume: {output}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; experiment={output}", flush=True)
    override = Path(args.pretrained) if args.pretrained else None
    if override and not override.is_absolute():
        override = project / override
    if args.enhancer != "none":
        model = build_model(args.model, project, not (args.resume or args.evaluate_test), override,
                            ortho_weight=args.ortho_weight, enhancer=args.enhancer).to(device)
    elif args.uni_mode == "finetune":
        model = build_model(args.model, project, not (args.resume or args.evaluate_test), override).to(device)
    else:
        model = build_model(args.model, project, not (args.resume or args.evaluate_test), override,
                            args.uni_mode, args.head_hidden_dim, args.head_dropout, args.ortho_weight).to(device)
    head_config = {"hidden_dim": args.head_hidden_dim, "dropout": args.head_dropout,
                   "ortho_weight": args.ortho_weight} if args.uni_mode != "finetune" else None
    print(f"Parameters: trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}, "
          f"total={sum(p.numel() for p in model.parameters()):,}", flush=True)
    if not args.smoke_test:
        weights.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    start, best, history, pending = 0, -1.0, [], None
    if args.resume or args.evaluate_test:
        if args.legacy_weights:
            path = Path(args.legacy_weights)
            if not path.is_absolute():
                path = project / path
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        else:
            state = load_checkpoint(best_path if args.evaluate_test else last)
            if (state["fingerprint"] != fingerprint or state["classes"] != list(CLASSES)
                    or state["model_name"] != args.model
                    or state.get("enhancer", "none") != args.enhancer
                    or state.get("uni_mode", "finetune") != args.uni_mode
                    or state.get("head_config") != head_config):
                raise ValueError("Model, class order or dataset manifest differs from checkpoint")
            model.load_state_dict(state["model"])
            prior_exclusions = state.get("val_exclusions", [])
            if prior_exclusions != exclusions and (not args.resume or prior_exclusions or not exclusions):
                raise ValueError("Validation exclusions differ from checkpoint; use the original exclusion list")
            if args.resume:
                if (state["batch_size"] != args.batch_size or state["lr"] != args.lr
                        or state["workers"] != args.workers
                        or state.get("weight_decay", 1e-5) != args.weight_decay):
                    raise ValueError("Resume requires the same batch-size, lr, weight-decay and workers")
                optimizer.load_state_dict(state["optimizer"])
                torch.set_rng_state(state["torch_rng"])
                random.setstate(state["python_rng"])
                np.random.set_state(state["numpy_rng"])
                if device.type == "cuda" and state["cuda_rng"] is not None:
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
                start, best, history = state["epoch"], state["best"], state["history"]
                if prior_exclusions != exclusions:
                    # Keep prior metrics, but never compare the old validation set's best score with the new one.
                    if best_path.exists():
                        backup = best_path.with_name("best_before_val_exclusion.pth")
                        if not backup.exists():
                            shutil.copy2(best_path, backup)
                    best = -1.0
                    print("Validation set changed explicitly: best-score selection reset; old history retained.", flush=True)
                pending = state.get("train_progress")
                if pending:
                    print(f"Resume epoch {start + 1}, next training batch {pending['batch_cursor'] + 1}", flush=True)
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=device.type == "cuda",
                                generator=torch.Generator().manual_seed(SEED))
               for split, ds in data.items()}
    def checkpoint_state(epoch, train_progress=None):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_name": args.model, "epoch": epoch, "best": best,
            "uni_mode": args.uni_mode, "head_config": head_config, "enhancer": args.enhancer,
            "val_exclusions": exclusions,
            "classes": list(CLASSES), "fingerprint": fingerprint,
            "batch_size": args.batch_size, "lr": args.lr, "workers": args.workers,
            "weight_decay": args.weight_decay, "train_progress": train_progress,
            "history": history, "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
    if not args.smoke_test and not args.evaluate_test:
        (output / "config.json").write_text(json.dumps({**vars(args), "classes": CLASSES,
            "seed": SEED, "dataset_fingerprint": fingerprint, "selection": "val_macro_f1"},
            indent=2, ensure_ascii=False), encoding="utf-8")
    if args.evaluate_test:
        scores, matrix, predictions = run_epoch(model, loaders["test"], device, collect_predictions=True)
        save_evaluation(reports, "test", scores, matrix.tolist(), CLASSES, *predictions,
                        image_names=[os.path.relpath(filename, dataset).replace("\\", "/")
                                     for filename, _ in data["test"].samples])
        print("test:", scores)
        return
    for epoch in range(start, start + 1 if args.smoke_test else args.epochs):
        # Independent epoch permutation and DataLoader RNG avoid consuming restored augmentation/dropout RNG.
        order = torch.randperm(len(data["train"]), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
        cursor = pending["batch_cursor"] if pending else 0
        train_loader = DataLoader(data["train"], batch_size=args.batch_size,
            sampler=order[cursor * args.batch_size:], num_workers=args.workers,
            pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(SEED + epoch))
        callback = None
        if not args.smoke_test and args.save_every_batches:
            def callback(progress):
                atomic_save(checkpoint_state(epoch, progress), last)
                print(f"Checkpoint saved: epoch {epoch + 1}, batch {progress['batch_cursor']}", flush=True)
            if pending is None:
                initial = {"total": 0, "loss_sum": 0.0, "correct": 0,
                           "matrix": torch.zeros((8, 8), dtype=torch.int64), "batch_cursor": 0}
                atomic_save(checkpoint_state(epoch, initial), last)
        train, _ = run_epoch(model, train_loader, device, optimizer, 2 if args.smoke_test else 0,
                             progress=pending, checkpoint_callback=callback, save_every=args.save_every_batches)
        pending = None
        val, matrix, predictions = run_epoch(model, loaders["val"], device,
                                             limit=2 if args.smoke_test else 0, collect_predictions=True)
        print(f"epoch={epoch + 1} train={train} val={val}", flush=True)
        if args.smoke_test:
            print("Smoke test completed; no formal checkpoints saved; test not evaluated.")
            return
        row = {"epoch": epoch + 1}
        if exclusions:
            row["val_excluded_count"] = len(exclusions)
            for previous in history:
                previous.setdefault("val_excluded_count", 0)
        row.update({"train_" + k: v for k, v in train.items()})
        row.update({"val_" + k: v for k, v in val.items()})
        history.append(row)
        improved = val["macro_f1"] > best
        best = max(best, val["macro_f1"])
        state = checkpoint_state(epoch + 1)
        if improved:
            atomic_save(state, best_path)
        atomic_save(state, last)
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps({**vars(args), "classes": CLASSES,
            "seed": SEED, "dataset_fingerprint": fingerprint, "selection": "val_macro_f1"},
            indent=2, ensure_ascii=False), encoding="utf-8")
        save_history(reports, history)
        if improved:
            save_evaluation(reports, "best_val", val, matrix.tolist(), CLASSES, *predictions,
                            image_names=[os.path.relpath(filename, dataset).replace("\\", "/")
                                         for filename, _ in data["val"].samples])
    if history:
        # Rebuild plots from checkpoint history after interrupted reporting.
        save_history(reports, history)
    print("Training finished. Selection uses val macro-F1; evaluate test separately.")


if __name__ == "__main__":
    main()
