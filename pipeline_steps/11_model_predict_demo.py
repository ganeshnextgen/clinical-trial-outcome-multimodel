#!/usr/bin/env python3
"""
Step 11: Demo/Quick Test Inference Script
"""
import joblib
import numpy as np
from pathlib import Path

def demo_predict(model_dir="models", data_dir="data/processed"):
    model_files = sorted(Path(model_dir).glob("model_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No model found.")
    model = joblib.load(model_files[-1])
    splits = joblib.load(Path(data_dir) / "data_splits.pkl")

    # Take a sample from test set
    X_sample = splits["X_test"][:5]
    preds = model.predict(X_sample)
    proba = model.predict_proba(X_sample)
    print(f"Test sample predictions: {preds}")
    print(f"Test sample probabilities:\n{proba}")

def main():
    demo_predict()
    print("🚀 Step 11 (Demo Prediction) completed.")

if __name__ == "__main__":
    main()
