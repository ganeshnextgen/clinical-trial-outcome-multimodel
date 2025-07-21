#!/usr/bin/env python3
"""
Step 07: Balanced Model Training with Fusion Features
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score
from imblearn.ensemble import BalancedRandomForestClassifier

class BalancedModelTrainer:
    """Balanced model trainer for clinical trial prediction"""

    def __init__(self, input_dir="data/processed", output_dir="models"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = BalancedRandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            sampling_strategy='auto',
            bootstrap=True
        )
        
    def load_training_data(self):
        # Look for the latest data splits
        split_path = self.input_dir / "data_splits.pkl"
        if not split_path.exists():
            raise FileNotFoundError(f"No data splits found at {split_path}. Run step 06 first.")
        splits = joblib.load(split_path)
        print(f"📂 Loaded training splits from: {split_path}")
        print(f"📊 Training set: {splits['X_train'].shape}")
        print(f"📊 Validation set: {splits['X_val'].shape}")
        print(f"📊 Test set: {splits['X_test'].shape}")
        return splits

    def analyze_class_distribution(self, y_train):
        failures = (y_train == 0).sum()
        successes = (y_train == 1).sum()
        ratio = successes / max(failures, 1)
        print("📊 CLASS DISTRIBUTION:")
        print(f" - Failures : {failures}")
        print(f" - Successes: {successes}")
        print(f" - Imbalance ratio (success/failure): {ratio:.2f}")
        return ratio

    def train_model(self, X_train, y_train, X_val, y_val):
        print("🎯 TRAINING MODEL...")
        self.model.fit(X_train, y_train)
        print("✅ Model training complete.")

        print("📈 TRAINING PERFORMANCE:")
        preds_train = self.model.predict(X_train)
        print(classification_report(y_train, preds_train, target_names=["Failure", "Success"]))

        print("📉 VALIDATION PERFORMANCE:")
        preds_val = self.model.predict(X_val)
        print(classification_report(y_val, preds_val, target_names=["Failure", "Success"]))

        proba_val = self.model.predict_proba(X_val)[:, 1]  # Probabilities for Success (positive class)
        metrics = {
            "validation_auc": roc_auc_score(y_val, proba_val),
            "failure_recall": recall_score(y_val, preds_val, pos_label=0),
            "failure_precision": precision_score(y_val, preds_val, pos_label=0, zero_division=0)
        }

        print(f"🔎 ROC AUC         : {metrics['validation_auc']:.4f}")
        print(f"🔎 Failure Recall  : {metrics['failure_recall']:.4f}")
        print(f"🔎 Failure Precision: {metrics['failure_precision']:.4f}")

        return metrics

    def save_model(self, metrics):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / f"model_{timestamp}.pkl"
        joblib.dump(self.model, model_path)

        meta = {
            "model_type": "BalancedRandomForestClassifier",
            "trained_at": timestamp,
            "params": self.model.get_params(),
            "validation_metrics": metrics
        }
        with open(self.output_dir / f"model_metadata_{timestamp}.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Model saved to: {model_path}")
        return model_path

def main():
    print("🚀 STEP 07: Training Balanced Model")
    trainer = BalancedModelTrainer()
    
    splits = trainer.load_training_data()
    trainer.analyze_class_distribution(splits["y_train"])
    
    metrics = trainer.train_model(
        splits["X_train"], splits["y_train"], 
        splits["X_val"], splits["y_val"]
    )
    
    trainer.save_model(metrics)
    print("✅ Step 07 completed ✅")

if __name__ == "__main__":
    main()
