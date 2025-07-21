#!/usr/bin/env python3
"""
Step 04: Generate Text Embeddings using DistilBERT or BioBERT
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import torch
from tqdm import tqdm

class TextEmbeddingGenerator:
    def __init__(self, model_name="distilbert-base-uncased", input_file="data/processed/clinical_trials_processed.csv", output_dir="data/processed"):
        self.model_name = model_name
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def encode_text(self, texts, batch_size=16, max_len=128):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding text"):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                outputs = self.model(**tokens)
                cls_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()
            all_embeddings.append(cls_embeddings)
        return np.vstack(all_embeddings)

    def generate_and_save(self):
        df = pd.read_csv(self.input_file)
        texts = df['combined_text'].astype(str).tolist()

        print(f"📄 Embedding {len(texts)} records using {self.model_name}...")
        embeddings = self.encode_text(texts)
        emb_df = pd.DataFrame(embeddings, columns=[f"text_emb_{i}" for i in range(embeddings.shape[1])])
        df_combined = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

        out_file = self.output_dir / "clinical_trials_embedded.csv"
        df_combined.to_csv(out_file, index=False)
        print(f"✅ Saved embedded data to {out_file}")

def main():
    embedder = TextEmbeddingGenerator()  # Swap model_name if using BioBERT / ClinicalBERT
    embedder.generate_and_save()

if __name__ == "__main__":
    main()
