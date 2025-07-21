#!/usr/bin/env python3
"""
Step 10: Export Model, Artifacts, and Demo Package
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
import joblib

def export_all(export_dir="exports", model_dir="models", data_dir="data/processed", results_dir="results"):
    export_dir = Path(export_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy all model files
    for f in Path(model_dir).glob("model_*.pkl"):
        shutil.copy2(f, export_dir)
    for f in Path(model_dir).glob("model_metadata_*.json"):
        shutil.copy2(f, export_dir)
    # Copy scaler, data splits, and results
    for f in Path(data_dir).glob("*.pkl"):
        shutil.copy2(f, export_dir)
    for f in Path(results_dir).rglob("*.json"):
        shutil.copy2(f, export_dir)
    for f in Path(results_dir).rglob("*.png"):
        shutil.copy2(f, export_dir)
    # Copy SHAP results if exists
    shap_dir = Path(results_dir) / "shap_analysis"
    if shap_dir.exists():
        for f in shap_dir.glob("*.*"):
            shutil.copy2(f, export_dir)
    print(f"✅ Exported all artifacts to {export_dir}")

def main():
    export_all()
    print("🚀 Step 10 (Export) completed.")

if __name__ == "__main__":
    main()
