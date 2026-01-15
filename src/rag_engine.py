import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import sys

# Ensure src is in path for local imports if needed
sys.path.append(os.path.dirname(__file__))
from embeddings import EmbeddingGenerator

class RAGEngine:
    def __init__(self, data_path='../data/processed_corpus_balanced.csv', model_name='all-MiniLM-L6-v2'):
        self.data_path = Path(data_path)
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._load_data()
        self._build_index()

    def _load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} not found.")
        
        print("Loading corpus...")
        df = pd.read_csv(self.data_path)
        # Drop nan
        df = df.dropna(subset=['clean_text'])
        
        # We can use a subset for speed if needed, but let's try full
        # For demonstration speedality, let's limit if too large, but 147k is handled fine by FAISS
        self.documents = df['clean_text'].astype(str).tolist()
        # Keep simple metadata
        self.metadata = df[['sentiment_score', 'source']].to_dict('records')
        print(f"Loaded {len(self.documents)} documents.")

    def _build_index(self):
        print("Encoding documents for RAG (this may take a moment)...")
        # Check if we have pre-computed embeddings? 
        # For now, let's compute on fly or load if saved. Saving would be better phase 4 step.
        # To avoid massive re-compute every run, we should check for a saved numpy file.
        emb_path = self.data_path.parent / "corpus_embeddings.npy"
        
        if emb_path.exists():
            print("Loading pre-computed embeddings...")
            embeddings = np.load(emb_path)
        else:
            print("Computing embeddings...")
            embeddings = self.encoder.encode(self.documents, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
            np.save(emb_path, embeddings)
            
        print("Building FAISS index...")
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors.")

    def retrieve(self, query, k=5):
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(distances[0][i])
                })
        return results

    def format_prompt(self, query, results):
        context_str = "\n".join([f"- [{r['metadata']['source']}] {r['text']}" for r in results])
        prompt = f"""You are an assistant for a Smart Urban System. Use the following citizen complaints/feedback context to answer the user's question.

Context:
{context_str}

User Question: {query}

Answer (summarize the sentiment and key issues found in context):"""
        return prompt
