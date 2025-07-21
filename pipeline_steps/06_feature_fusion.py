#!/usr/bin/env python3
"""
Step 06: Multimodal Feature Fusion and Dataset Splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class FeatureFusion:
    def __init__(self, input_file="data/processed/clinical_trials_fused.csv", output_dir="data/processed"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def fuse_and_split(self):
        df = pd.read_csv(self.input_file)
        y = df['outcome']
        text_cols = [c for c in df.columns if c.startswith("text_emb_")]
        struct_cols = [c for c in df.columns if c.endswith("_scaled")]
        X = df[text_cols + struct_cols].values
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        data_splits = {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train.to_numpy(), "y_val": y_val.to_numpy(), "y_test": y_test.to_numpy()
        }
        joblib.dump(data_splits, self.output_dir / "data_splits.pkl")
        print("✅ Data splits saved successfully.")
        print(f"   • Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return data_splits

def main():
    fusion = FeatureFusion()
    fusion.fuse_and_split()

if __name__ == "__main__":
    main()
