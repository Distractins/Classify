"""合成临床表、图像和权重验证真实流水线，不需要下载 HMU 数据。"""
import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pipeline_common as common
from stage1_train_8cls import patient_dataset_split as split
from stage2_msi_prediction import patch_infer_all as infer
from stage2_msi_prediction import patient_tme_agg_stat as aggregate
from stage2_msi_prediction import train_msi_model as train

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models
from torchvision.datasets import ImageFolder


class PatientPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pipeline_test_", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patch_root = patch.object(common, "PROJECT_ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        torch.set_num_threads(1)
        self.ids = [f"P{i:03d}" for i in range(20)]
        self.clinical = pd.DataFrame({"patient_id": self.ids})

    def run_main(self, module, *arguments):
        with patch.object(sys, "argv", [module.__file__, *arguments]), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_split_is_reused_without_msi_and_rejects_overlap(self):
        xlsx = self.root / "clinical.xlsx"
        pd.DataFrame({"PatientID": self.ids, "MSI": ["invalid"] * 20}).to_excel(xlsx, index=False)
        with patch.object(common.pd, "read_excel", wraps=pd.read_excel) as reader:
            ids_only = common.read_clinical(xlsx)
            self.assertEqual(reader.call_args.kwargs["usecols"], ["PatientID"])
            self.assertEqual(list(ids_only.columns), ["patient_id"])
        first = split.create_patient_split(ids_only)
        before = common.project_path(common.TRAIN_IDS).read_bytes()
        repeated = split.create_patient_split(ids_only, val_ratio=0.5, seed=99)
        self.assertEqual(first[:2], repeated[:2])
        self.assertTrue(repeated[2])
        self.assertEqual(before, common.project_path(common.TRAIN_IDS).read_bytes())
        with self.assertRaises(ValueError):
            common.load_patient_split(self.ids[:-1])
        val_path = common.project_path(common.VAL_IDS)
        val = pd.read_csv(val_path)
        val.loc[0, "patient_id"] = first[0][0]
        val.to_csv(val_path, index=False)
        with self.assertRaises(ValueError):
            common.load_patient_split()

    def test_mapping_is_exact_or_explicit_and_paths_are_relative(self):
        resolver = common.PatchPatientResolver(["P1", "P10"])
        self.assertEqual(resolver(self.root / "dataset/ADI/P10_patch.png"), "P10")
        with self.assertRaises(ValueError):
            resolver(self.root / "dataset/ADI/P1_P10.png")
        with self.assertRaises(ValueError):
            resolver(self.root / "dataset/ADI/unknown.png")
        resolver = common.PatchPatientResolver(["P1"], patient_regex=r"case-(?P<patient_id>P\d+)-")
        self.assertEqual(resolver(self.root / "dataset/ADI/case-P1-patch.png"), "P1")
        pd.DataFrame({"image_file_path": ["dataset/ADI/opaque.png"], "patient_id": ["P1"]}).to_csv(
            self.root / "mapping.csv", index=False)
        resolver = common.PatchPatientResolver(["P1"], patch_map="mapping.csv")
        self.assertEqual(resolver(self.root / "dataset/ADI/opaque.png"), "P1")
        for path in ("../escape", "H:/dataset", "G:\\weights"):
            with self.assertRaises(ValueError):
                common.project_path(path)

    def test_aggregation_counts_missing_classes_and_rejects_duplicate_patch(self):
        path = self.root / "patch.csv"
        patches = pd.DataFrame({"patient_id": ["P1"] * 3,
                                "image_file_path": ["dataset/ADI/P1_a.png", "dataset/TUM/P1_b.png", "dataset/TUM/P1_c.png"],
                                "predict_class": ["ADI", "TUM", "TUM"]})
        patches.to_csv(path, index=False)
        result = aggregate.aggregate_patches(path, {"P1"}, chunksize=1)
        self.assertAlmostEqual(result.loc[0, "TUM%"], 200 / 3)
        self.assertAlmostEqual(result.loc[0, common.TME_COLUMNS].sum(), 100)
        self.assertEqual(result.loc[0, "STR%"], 0)
        with self.assertRaises(ValueError):
            aggregate.aggregate_patches(path, {"P1", "P2"})
        pd.concat([patches, patches.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            aggregate.aggregate_patches(path, {"P1"}, chunksize=1)
        self.assertEqual(aggregate.encode_msi("MSI‑H"), 1)
        self.assertEqual(aggregate.encode_msi("MSS"), 0)
        self.assertTrue(np.isnan(aggregate.encode_msi(None)))

    def test_resnet_and_vit_wrapper_checkpoints_load_offline(self):
        from transformers import ViTConfig, ViTForImageClassification
        for name in ("resnet101", "vit"):
            with self.subTest(model=name):
                if name == "resnet101":
                    source = models.resnet101(weights=None)
                    source.fc = torch.nn.Linear(source.fc.in_features, 8)
                    prefix = "resnet."
                else:
                    source = ViTForImageClassification(ViTConfig(num_labels=8))
                    prefix = "vit."
                state = {prefix + key: value for key, value in source.state_dict().items()}
                checkpoint = {"state_dict": state, "class_names": list(common.CLASSES)}
                with patch.object(infer.torch, "load", return_value=checkpoint):
                    model, transform = infer.load_patch_model("unused.pth", name, torch.device("cpu"))
                tensor = transform(Image.new("RGB", (32, 32))).unsqueeze(0)
                with torch.inference_mode():
                    result = model(tensor)
                logits = result.logits if hasattr(result, "logits") else result
                self.assertEqual(tuple(logits.shape), (1, 8))
                del source, state, checkpoint, model, result, logits

    def test_full_pipeline_and_training_only_imputation(self):
        # 固定患者名单，合成标签不会影响初始患者划分。
        train_ids, val_ids, _ = split.create_patient_split(self.clinical)
        labels = {pid: i % 2 for group in (train_ids, val_ids) for i, pid in enumerate(group)}
        clinical = pd.DataFrame({"PatientID": self.ids, "Age": [40 + i for i in range(20)],
                                 "Sex": ["Male" if i % 2 else "Female" for i in range(20)],
                                 "MSI": ["MSI-H" if labels[pid] else "MSS" for pid in self.ids]})
        clinical.loc[clinical.PatientID == train_ids[0], "Age"] = np.nan
        clinical.to_excel(self.root / "clinical.xlsx", index=False)
        for cls in common.CLASSES:
            (self.root / "dataset" / cls).mkdir(parents=True)
        for i, pid in enumerate(self.ids):
            for j in range(2):
                Image.new("RGB", (32, 32), (i * 10, j * 80, 30)).save(
                    self.root / "dataset" / common.CLASSES[(i + j) % 8] / f"{pid}_patch{j}.png")
        resolver = common.PatchPatientResolver(self.ids)
        imagefolder = ImageFolder(self.root / "dataset")
        train_indices, val_indices = common.partition_imagefolder(imagefolder, resolver)
        self.assertEqual({resolver(imagefolder.samples[i][0]) for i in train_indices}, set(train_ids))
        self.assertEqual({resolver(imagefolder.samples[i][0]) for i in val_indices}, set(val_ids))

        cnn = models.efficientnet_b0(weights=None)
        cnn.classifier[1] = torch.nn.Linear(cnn.classifier[1].in_features, 8)
        state = {f"efficientnet.{key}": value for key, value in cnn.state_dict().items()}
        torch.save({"state_dict": state, "class_names": list(common.CLASSES),
                    "train_patient_ids": train_ids, "val_patient_ids": val_ids}, self.root / "synthetic.pth")
        self.run_main(infer, "--dataset", "dataset", "--clinical", "clinical.xlsx",
                      "--weights", "synthetic.pth", "--batch-size", "8", "--device", "cpu")
        output = common.project_path(common.CSV_DIR)
        patches = pd.read_csv(output / "patch_infer_result.csv")
        self.assertEqual(len(patches), 40)
        self.assertEqual(set(patches.patient_id), set(self.ids))
        self.assertTrue(patches.image_file_path.str.startswith("dataset/").all())
        self.run_main(aggregate, "--dataset", "dataset", "--clinical", "clinical.xlsx")
        features = pd.read_csv(output / "patient_tme_feature.csv")
        self.assertEqual(len(features), 20)
        np.testing.assert_allclose(features[common.TME_COLUMNS].sum(axis=1), 100)
        hashes = [hashlib.sha256(common.project_path(p).read_bytes()).hexdigest()
                  for p in (common.TRAIN_IDS, common.VAL_IDS)]
        self.run_main(train, "--n-estimators", "3", "--ablations")
        self.assertEqual(hashes, [hashlib.sha256(common.project_path(p).read_bytes()).hexdigest()
                                 for p in (common.TRAIN_IDS, common.VAL_IDS)])
        for name in ("msi_roc_curve.png", "msi_confusion.png", "feature_importance.png", "tme_msi_corr_heatmap.png"):
            self.assertGreater((common.project_path(train.FIGURES) / name).stat().st_size, 1000)
        model = joblib.load(common.project_path("outputs/weights/stage2_msi/msi_xgb_model.pkl"))
        expected_median = features.set_index("patient_id").loc[train_ids, "age"].median()
        self.assertEqual(model.named_steps["imputer"].statistics_[-2], expected_median)
        predictions = pd.read_csv(output / "msi_validation_predictions.csv")
        self.assertEqual(set(predictions.patient_id), set(val_ids))
        np.testing.assert_allclose(model.predict_proba(features.set_index("patient_id").loc[
            predictions.patient_id, train.FEATURES])[:, 1], predictions.msi_probability, rtol=1e-6)
        self.assertEqual(len(pd.read_csv(output / "msi_ablation_results.csv")), 3)
        # 失去验证集某个标签类别时应报错，不能重新划分患者来制造 AUC。
        features.loc[features.patient_id.isin(val_ids), "msi_label"] = 0
        features.to_csv(output / "patient_tme_feature.csv", index=False)
        with self.assertRaises(ValueError):
            train.load_features(output / "patient_tme_feature.csv")


if __name__ == "__main__":
    unittest.main()
