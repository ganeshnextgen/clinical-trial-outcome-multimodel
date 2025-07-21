#!/usr/bin/env python3
"""
Step 05: Prepare Structured Features (scaling, normalization)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class StructuredFeatureProcessor:
    def __init__(self, input_file="data/processed/clinical_trials_embedded.csv", output_dir="data/processed"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()

    def identify_features(self, df):
        structured = [c for c in df.columns if
                      c.endswith("_encoded") or
                      "year" in c or
                      "trial_duration" in c or
                      c in ["enrollment_count", "is_multi_phase", "is_international"]]
        return structured

    def scale_and_save(self):
        df = pd.read_csv(self.input_file)

        # Get features
        features = self.identify_features(df)
        print(f"📊 Structured columns to scale: {features}")
        X_structured = df[features].fillna(0)

        X_scaled = self.scaler.fit_transform(X_structured)
        scaled_df = pd.DataFrame(X_scaled, columns=[f"{col}_scaled" for col in features])
        final_df = pd.concat([df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

        out_file = self.output_dir / "clinical_trials_fused.csv"
        final_df.to_csv(out_file, index=False)
        joblib.dump(self.scaler, self.output_dir / "structured_scaler.pkl")
        print(f"✅ Structured features saved to {out_file}")
        return final_df

def main():
    processor = StructuredFeatureProcessor()
    _ = processor.scale_and_save()

if __name__ == "__main__":
    main()
