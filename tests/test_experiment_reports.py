"""Validate probability-based ROC, undefined classes, and confusion-derived scores."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_reports import evaluation_metrics, save_evaluation


class ReportTests(unittest.TestCase):
    def test_scores_and_probability_auc_match_reference(self):
        targets = np.array([0, 0, 1, 1, 2, 2])
        probabilities = np.array([[.8,.1,.1],[.4,.5,.1],[.2,.6,.2],
                                  [.5,.3,.2],[.1,.3,.6],[.2,.5,.3]])
        predicted = probabilities.argmax(1)
        matrix = np.zeros((3, 3), dtype=int)
        np.add.at(matrix, (targets, predicted), 1)
        scores, p, r, f, _ = evaluation_metrics(matrix)
        rp, rr, rf, _ = precision_recall_fscore_support(targets, predicted, labels=[0,1,2], zero_division=0)
        np.testing.assert_allclose(p, rp)
        np.testing.assert_allclose(r, rr)
        np.testing.assert_allclose(f, rf)
        self.assertAlmostEqual(scores['balanced_accuracy'], balanced_accuracy_score(targets, predicted))
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            save_evaluation(temporary, 'test', {}, matrix.tolist(), ['A','B','C'], targets, probabilities)
            report = json.loads((Path(temporary) / 'test_metrics.json').read_text())
            self.assertAlmostEqual(report['mean_auc_ovr'], roc_auc_score(targets, probabilities, multi_class='ovr', average='macro'))
            self.assertTrue((Path(temporary) / 'test_precision_recall_f1.png').exists())
            self.assertTrue((Path(temporary) / 'test_roc_ovr.png').exists())
            self.assertTrue((Path(temporary) / 'test_predictions.csv').exists())

    def test_absent_class_auc_is_null(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            scores = {}
            save_evaluation(temporary, 'test', scores, [[1,0,0],[0,1,0],[0,0,0]],
                            ['A','B','C'], [0,1], [[.8,.1,.1],[.1,.8,.1]])
            self.assertIsNone(scores['mean_auc_ovr'])
            self.assertEqual(scores['valid_auc_class_count'], 2)


if __name__ == '__main__':
    unittest.main()
