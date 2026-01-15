from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import torch

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def generate(self, texts, batch_size=32):
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        return embeddings
