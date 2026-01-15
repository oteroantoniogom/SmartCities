import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from embeddings import EmbeddingGenerator

def main():
    print("Loading data...")
    data_path = Path("data/processed_corpus_balanced.csv")
    if not data_path.exists():
        print("Data not found!")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['clean_text', 'sentiment_score'])
    
    # Sample 5000 for speed
    print("Sampling 2000 points (CPU optimization)...")
    df_sample = df.sample(n=2000, random_state=42)
    texts = df_sample['clean_text'].astype(str).tolist()
    labels = df_sample['sentiment_score'].tolist()
    
    print("Generating S-BERT embeddings...")
    emb_gen = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    embeddings = emb_gen.generate(texts)
    
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    print("Saving plot...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=labels,
        palette='viridis',
        alpha=0.6
    )
    plt.title('UMAP Projection of S-BERT Embeddings')
    plt.savefig('umap_sbert.png')
    print("Plot saved to umap_sbert.png")

if __name__ == "__main__":
    main()
