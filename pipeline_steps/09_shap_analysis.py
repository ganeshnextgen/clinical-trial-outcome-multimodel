#!/usr/bin/env python3
"""
Step 09: SHAP Analysis for Clinical Trial Outcome Model
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shap

class SHAPAnalyzer:
    def __init__(self, model_path="models", data_path="data/processed", out_dir="results/shap_analysis"):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.X = None
        self.y = None

    def load_data_and_model(self):
        # Find most recent model
        models = list(self.model_path.glob("model_*.pkl"))
        if not models:
            raise FileNotFoundError("No model found.")
        model_file = sorted(models)[-1]
        self.model = joblib.load(model_file)
        splits = joblib.load(self.data_path / "data_splits.pkl")
        self.X = splits["X_test"]
        self.y = splits["y_test"]
        print(f"✅ Loaded model {model_file.name}. Testing on {self.X.shape[0]} samples.")

    def run_shap(self, max_samples=100):
        bg = shap.utils.sample(self.X, min(100, self.X.shape[0]))
        explainer = shap.TreeExplainer(self.model, bg)
        ns = min(max_samples, self.X.shape[0])
        shap_vals = explainer.shap_values(self.X[:ns])
        self._save_shap_figs(shap_vals, ns)
        np.save(self.out_dir / "shap_values.npy", shap_vals)
        print(f"✅ Saved SHAP analysis for {ns} test samples.")

    def _save_shap_figs(self, shap_vals, nsamples):
        # Beeswarm
        plt.figure()
        shap.summary_plot(shap_vals, self.X[:nsamples], show=False)
        plt.title("SHAP Beeswarm (Test Set)")
        plt.savefig(self.out_dir / "shap_beeswarm.png")
        plt.close()

        # Bar
        plt.figure()
        shap.summary_plot(shap_vals, self.X[:nsamples], plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Test Set)")
        plt.savefig(self.out_dir / "shap_bar.png")
        plt.close()
        print("✅ SHAP plots saved.")

def main():
    analyzer = SHAPAnalyzer()
    analyzer.load_data_and_model()
    analyzer.run_shap()
    print("🚀 Step 09 (SHAP) completed.")

if __name__ == "__main__":
    main()
