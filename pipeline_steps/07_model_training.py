#!/usr/bin/env python3
"""
Step 07: Train Balanced Random Forest Model (Fixed + Safe)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score
from imblearn.ensemble import BalancedRandomForestClassifier
import sys

class BalancedModelTrainer:
    def __init__(self, input_dir="data/processed", output_dir="models"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.model = BalancedRandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True
        )

    def load_data_splits(self):
        split_file = self.input_dir / "data_splits.pkl"
        if not split_file.exists():
            raise FileNotFoundError("Missing data_splits.pkl — Please run Step 06 first.")
        splits = joblib.load(split_file)
        print(f"📂 Loaded split file: {split_file}")
        return splits

    def check_class_balance(self, y, name="y_train"):
        counts = pd.Series(y).value_counts()
        print(f"📊 {name} distribution:")
        print(counts.to_dict())
        if len(counts) < 2:
            print("❌ Only one class found in training labels. Cannot train classifier.")
            return False
        return True

    def train_model(self, X_train, y_train, X_val, y_val):
        print("🎯 Training BalancedRandomForest Classifier...")
        self.model.fit(X_train, y_train)

        preds_train = self.model.predict(X_train)
        preds_val = self.model.predict(X_val)
        probas_val = self.model.predict_proba(X_val)[:, 1]

        print("\n📈 Training performance:")
        print(classification_report(y_train, preds_train, target_names=["Failure", "Success"]))
        print("\n📉 Validation performance:")
        print(classification_report(y_val, preds_val, target_names=["Failure", "Success"]))

        results = {
            "val_auc": roc_auc_score(y_val, probas_val),
            "val_failure_recall": recall_score(y_val, preds_val, pos_label=0),
            "val_failure_precision": precision_score(y_val, preds_val, pos_label=0, zero_division=0)
        }
        print("\n🎯 Key validation metrics:")
        print(f" - ROC AUC             : {results['val_auc']:.4f}")
        print(f" - Failure Recall      : {results['val_failure_recall']:.4f}")
        print(f" - Failure Precision   : {results['val_failure_precision']:.4f}")

        return results

    def save_model(self, metrics):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / f"model_{timestamp}.pkl"
        meta_path = self.output_dir / f"model_metadata_{timestamp}.json"

        joblib.dump(self.model, model_path)
        with open(meta_path, "w") as f:
            json.dump({
                "model_type": "BalancedRandomForestClassifier",
                "trained_at": timestamp,
                "model_params": self.model.get_params(),
                "validation_metrics": metrics
            }, f, indent=2)

        print(f"✅ Model saved: {model_path}")
        print(f"✅ Metadata saved: {meta_path}")
        return model_path

def main():
    print("🚀 STEP 07: MODEL TRAINING")
    trainer = BalancedModelTrainer()

    try:
        splits = trainer.load_data_splits()
        if not trainer.check_class_balance(splits["y_train"], "y_train"):
            print("⚠️ Please recheck your data or use more samples.")
            sys.exit(1)

        val_ok = trainer.check_class_balance(splits["y_val"], "y_val")
        if not val_ok:
            print("⚠️ Validation set only has one class. Cannot evaluate reliably.")

        metrics = trainer.train_model(
            splits["X_train"], splits["y_train"],
            splits["X_val"], splits["y_val"]
        )
        trainer.save_model(metrics)
        print("✅ Step 07 complete.")

    except Exception as e:
        print(f"🔥 ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
