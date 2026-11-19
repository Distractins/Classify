"""Evaluate the same trusted best checkpoint on original/Macenko/HED test inputs."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.stage1_train_8cls import train_wsi_baseline as baseline
from src.stain_perturbations import image_rng, perturb_stain
from src.experiment_reports import save_evaluation, plotting


class StainDataset(Dataset):
    def __init__(self, samples, root, method, strength, seed):
        self.samples, self.root = samples, root
        self.method, self.strength, self.seed = method, strength, seed
        self.resize = transforms.Resize((224, 224))
        self.tensor = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        name = os.path.relpath(filename, self.root).replace("\\", "/")
        try:
            with Image.open(filename) as image:
                original = self.resize(image.convert("RGB"))
            shifted = perturb_stain(original, self.method, self.strength,
                                    image_rng(name, self.seed))
        except Exception as exc:
            raise RuntimeError(f"Stain evaluation failed ({self.method}): {filename}; no image skipped") from exc
        return self.tensor(shifted), label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trusted experiment weights/best.pth")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.strength < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Require 0 < strength < 1, batch-size >= 1, workers >= 0")
    root, checkpoint = args.root.resolve(), args.checkpoint.resolve()
    output = args.output or checkpoint.parent.parent / "reports" / f"stain_s{args.strength:g}_seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"Choose a new --output to preserve previous results: {output}")
    state = baseline.load_checkpoint(checkpoint)
    data, manifest = baseline.load_data(root)
    fingerprint = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    if state["classes"] != list(baseline.CLASSES) or state["fingerprint"] != fingerprint:
        raise ValueError("Class order or dataset manifest differs from checkpoint")
    head = state.get("head_config") or {}
    model = baseline.build_model(state["model_name"], PROJECT, pretrained=False,
        uni_mode=state.get("uni_mode", "finetune"), hidden_dim=head.get("hidden_dim", 256),
        dropout=head.get("dropout", 0.1), ortho_weight=head.get("ortho_weight", 0.01))
    model.load_state_dict(state["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    output.mkdir(parents=True)
    checkpoint_hash = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_hash.update(block)
    digest = checkpoint_hash.hexdigest()
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "dataset_fingerprint": fingerprint, "epoch": state["epoch"], "seed": args.seed,
        "strength": args.strength, "classes": list(baseline.CLASSES),
        "model": state["model_name"], "uni_mode": state.get("uni_mode", "finetune"),
        "torch": torch.__version__, "device": str(device),
        "protocol": "Resize224 then stain shift then ImageNet normalization; no test tuning; synthetic shift only",
        "macenko": "OD threshold0.15, SVD plane, angle percentiles1/99; basis noise SD=strength/5, concentration scale U(1-s,1+s); preserve OD residual",
        "hed": "Signed OD decomposition with H=[.65,.70,.29], E=[.07,.99,.11], D=[.27,.57,.78]; scale H/E by U(1-s,1+s); preserve third channel"}
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    samples = data["test"].samples
    names = [os.path.relpath(name, root).replace("\\", "/") for name, _ in samples]
    rows = []
    for method in ("original", "macenko", "hed"):
        directory = output / method
        directory.mkdir()
        dataset = StainDataset(samples, root, method, args.strength, args.seed)
        # Save identical preview selections for visual inspection across conditions.
        preview_indices = [next(i for i, (_, label) in enumerate(samples) if label == cls)
                           for cls in range(len(baseline.CLASSES))]
        for cls, index in zip(baseline.CLASSES, preview_indices):
            filename, _ = samples[index]
            with Image.open(filename) as image:
                original = dataset.resize(image.convert("RGB"))
            preview = perturb_stain(original, method, args.strength, image_rng(names[index], args.seed))
            preview.save(directory / f"preview_{cls}.png")
        print(f"Evaluating {method}: {len(samples)} images", flush=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
        scores, matrix, predictions = baseline.run_epoch(model, loader, device, collect_predictions=True)
        save_evaluation(directory, "test", scores, matrix.tolist(), baseline.CLASSES,
                        *predictions, image_names=names)
        row = {"condition": method, **scores}
        for metric in ("accuracy", "macro_f1", "mean_auc_ovr"):
            reference = rows[0][metric] if rows else row[metric]
            row[metric + "_drop_pp"] = 100 * (reference - row[metric]) if reference is not None and row[metric] is not None else None
        rows.append(row)
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plt = plotting(output)
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, metric in enumerate(("accuracy", "macro_f1", "mean_auc_ovr")):
        ax.bar([i + (offset - 1) * 0.25 for i in range(3)],
               [row[metric] for row in rows], width=0.25, label=metric)
    ax.set(xticks=range(3), xticklabels=["原始测试", "Macenko 染色变化", "HED 染色扰动"],
           ylabel="指标值", ylim=(0, 1), title="染色扰动鲁棒性对比")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=200)
    plt.close(fig)
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()
