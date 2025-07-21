#!/usr/bin/env python3
"""
Step 08: Model Evaluation + Threshold Optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score, fbeta_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

class ModelEvaluator:
    """Evaluate model and optimize threshold"""

    def __init__(self, model_dir="models", data_dir="data/processed", result_dir="results"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def load_latest_model(self):
        models = sorted(self.model_dir.glob("model_*.pkl"))
        if not models:
            raise FileNotFoundError("No trained model found.")
        latest = models[-1]
        model = joblib.load(latest)
        print(f"📂 Loaded model: {latest.name}")
        return model, latest

    def load_test_data(self):
        data_path = self.data_dir / "data_splits.pkl"
        if not data_path.exists():
            raise FileNotFoundError("No data splits found.")
        splits = joblib.load(data_path)
        print(f"📂 Loaded test set from: {data_path.name}")
        return splits["X_test"], splits["y_test"]

    def evaluate(self, model, X_test, y_test):
        probs = model.predict_proba(X_test)[:, 1]
        report_at_05 = self._evaluate_at_threshold(y_test, probs, threshold=0.5)
        threshold_opt = self._find_best_threshold(y_test, probs)
        report_at_opt = self._evaluate_at_threshold(y_test, probs, threshold=threshold_opt)

        self._save_conf_matrix(y_test, probs, [0.5, threshold_opt])
        self._save_json_reports(report_at_05, report_at_opt, threshold_opt)

        return report_at_opt

    def _evaluate_at_threshold(self, y_true, probas, threshold):
        preds = (probas >= threshold).astype(int)
        report = classification_report(y_true, preds, output_dict=True)
        report["roc_auc"] = roc_auc_score(y_true, probas)
        report["threshold"] = threshold
        report["failure_recall"] = recall_score(y_true, preds, pos_label=0)
        report["failure_precision"] = precision_score(y_true, preds, pos_label=0, zero_division=0)
        print(f"📊 Threshold = {threshold:.2f}")
        print(classification_report(y_true, preds, target_names=["Failure", "Success"]))
        return report

    def _find_best_threshold(self, y_true, probas):
        best_t = 0.5
        best_score = 0
        for t in np.arange(0.1, 0.91, 0.01):
            preds = (probas >= t).astype(int)
            score = fbeta_score(y_true, preds, beta=2.0, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = t
        print(f"🎯 Optimal F2 threshold: {best_t:.2f}, F2 Score: {best_score:.3f}")
        return best_t

    def _save_conf_matrix(self, y_true, probas, thresholds):
        for t in thresholds:
            preds = (probas >= t).astype(int)
            cm = confusion_matrix(y_true, preds)
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=["Failure", "Success"],
                        yticklabels=["Failure", "Success"])
            plt.title(f"Confusion Matrix (Threshold={t:.2f})")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plot_file = self.result_dir / f"conf_matrix_threshold_{t:.2f}.png"
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()
            print(f"✅ Saved plot: {plot_file.name}")

    def _save_json_reports(self, report_05, report_opt, threshold_opt):
        summary = {
            "standard_threshold": 0.5,
            "optimized_threshold": threshold_opt,
            "standard": report_05,
            "optimized": report_opt
        }
        report_file = self.result_dir / "evaluation_report.json"
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Evaluation report saved: {report_file.name}")

def main():
    print("🚀 STEP 08: Evaluate Model")
    evaluator = ModelEvaluator()
    model, _ = evaluator.load_latest_model()
    X_test, y_test = evaluator.load_test_data()
    evaluator.evaluate(model, X_test, y_test)
    print("✅ Step 08 completed ✅")

if __name__ == "__main__":
    main()
