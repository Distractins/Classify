import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import (CLASSES, load_patient_split, partition_imagefolder,
                             project_path, stage1_data_setup)
from pretrained_weights import load_cnn_pretrained
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm  # 添加这个
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from torchvision import transforms


# 定义ResNet101模型
class ResNet101Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101Model, self).__init__()
        self.resnet = models.resnet101(weights=None)
        load_cnn_pretrained(self.resnet, "resnet101")
        
        # 替换最后的全连接层
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)



def evaluate_model_on_validation_set(model, validation_loader, device, epoch, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    progress_bar = tqdm(validation_loader, desc=f"🔍 Validating (Epoch {epoch})", leave=False)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    print("\n📊 Classification Report:")
    report = classification_report(all_labels, all_preds, labels=np.arange(len(class_names)),
                                   target_names=class_names, digits=4, zero_division=0)
    print(report)
    report_path = project_path("logs/stage1_log.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nresnet101 epoch={epoch}\n{report}\n")
    scores = classification_report(all_labels, all_preds, labels=np.arange(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    positions = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric in enumerate(("precision", "recall", "f1-score")):
        ax.bar(positions + (offset - 1) * 0.25,
               [scores[label][metric] for label in class_names], width=0.25, label=metric)
    ax.set_xticks(positions)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    score_path = project_path(f"outputs/figures/stage1_8cls/resnet101_epoch{epoch}_prf.png")
    score_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(score_path, dpi=300)
    plt.close(fig)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(class_names)))
    cm_normalized = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)  # 按行归一化为百分比

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar(label="Proportion")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # 在图中添加百分比文本
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")

    plt.tight_layout()
    cm_path = str(project_path(f"outputs/figures/stage1_8cls/resnet101_epoch{epoch}_confusion_matrix.png"))
    Path(cm_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ 归一化混淆矩阵已保存：{cm_path}")


    # ROC 和 AUC
    y_true_bin = label_binarize(all_labels, classes=np.arange(len(class_names)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        if np.unique(y_true_bin[:, i]).size < 2:
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in roc_auc if isinstance(i, int)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in [k for k in roc_auc if isinstance(k, int)]:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= sum(isinstance(k, int) for k in roc_auc)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    for i in [k for k in roc_auc if isinstance(k, int)]:
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot(fpr["micro"], tpr["micro"], linestyle='--', label=f'Micro avg (AUC = {roc_auc["micro"]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], linestyle='--', label=f'Macro avg (AUC = {roc_auc["macro"]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.annotate(f"Macro AUC = {roc_auc['macro']:.4f}\nMicro AUC = {roc_auc['micro']:.4f}",
                 xy=(0.7, 0.1), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle="round", fc="w"))
    roc_path = str(project_path(f"outputs/figures/stage1_8cls/resnet101_epoch{epoch}_roc_curve.png"))
    plt.savefig(roc_path)
    plt.close()
    print(f"✅ ROC图已保存：{roc_path}")

    model.train()
    return 100.0 * (all_preds == all_labels).sum() / len(all_labels)

# 数据集路径
def main():
    dataset_path, patient_resolver = stage1_data_setup() 

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 加载数据集并进行预处理

    # 加载数据集
    full_dataset = ImageFolder(dataset_path, transform=transform)
    class_names = full_dataset.classes  # ['classA', 'classB', ..., 'classH']

    # 收集每个类别的样本索引
    # Reuse saved patient IDs; never randomly split images.
    if tuple(class_names) != CLASSES:
        raise ValueError("Expected the eight README classes in ADI~TUM order.")
    train_indices, val_indices = partition_imagefolder(full_dataset, patient_resolver)

    train_dataset = Subset(full_dataset, train_indices)
    validation_dataset = Subset(full_dataset, val_indices)

    # 数据加载器
    # shuffle 只改变固定训练集内的批次顺序，不重新划分患者。
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=False)


    # 实例化模型
    model = ResNet101Model(num_classes=8) 
    # 指定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    project_path("outputs/weights/stage1_8cls").mkdir(parents=True, exist_ok=True)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 训练模型
    num_epochs = 5
    best_acc = -1.0
    no_improve_count = 0

    #print("Unique labels in batch:", labels.unique())

    # ✅ 替换你的训练循环部分
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
        for i, data in enumerate(progress_bar):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        validation_accuracy = evaluate_model_on_validation_set(model, validation_loader, device, epoch+1, class_names)

        print(f'Epoch {epoch+1}, Validation Accuracy: {validation_accuracy:.2f}%')

        if validation_accuracy > best_acc:
                best_acc = validation_accuracy
                no_improve_count = 0
                torch.save({"state_dict": model.state_dict(), "class_names": list(CLASSES),
                                "train_patient_ids": load_patient_split()[0],
                                "val_patient_ids": load_patient_split()[1]},
                               project_path("outputs/weights/stage1_8cls/resnet101_best.pth"))
                print("✅ Saved Better Model")
        else:
           no_improve_count += 1

           if no_improve_count >= 15:
               print("🛑 Stopping early due to no improvement in validation accuracy.")
               break

        scheduler.step()
    print("🎉 Training Complete")

if __name__ == "__main__":
    main()
