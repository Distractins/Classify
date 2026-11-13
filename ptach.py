import os
import random
import shutil
import pandas as pd

random.seed(42)

# ========== 配置区 ==========
root_original = r"D:\dataset\HMU-CRC-ORIGIN"
root_subset = r"D:\dataset\HMU-CRC-SUBSET"
patch_map_csv = r"D:\dataset\patch_patient_map.csv" # 三列：img_name,patient_id,cls
max_patch_per_patient_per_class = 8  # 单个病人，同一个类别最多抽几张，可调
classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
# =============================

os.makedirs(root_subset, exist_ok=True)
for c in classes:
    os.makedirs(os.path.join(root_subset, c), exist_ok=True)

df = pd.read_csv(patch_map_csv)
patient_list = df["patient_id"].unique()
print(f"总患者数量：{len(patient_list)}")

for pid in patient_list:
    patient_df = df[df["patient_id"] == pid]
    for c in classes:
        cls_df = patient_df[patient_df["cls"] == c]
        if len(cls_df) == 0:
            continue
        img_names = cls_df["img_name"].tolist()
        take_num = min(max_patch_per_patient_per_class, len(img_names))
        selected = random.sample(img_names, take_num)
        for img in selected:
            src = os.path.join(root_original, c, img)
            dst = os.path.join(root_subset, c, img)
            shutil.copy2(src, dst)

# 统计最终每类数量
print("\n==== 子集统计 ====")
for c in classes:
    cnt = len(os.listdir(os.path.join(root_subset, c)))
    print(f"{c}: {cnt}")
