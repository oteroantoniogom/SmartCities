import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from dl_models import DLManager

def main():
    print("Loading data for DL...")
    data_path = Path("data/processed_corpus_balanced.csv")
    if not data_path.exists():
        print("Data not found!")
        return

    df = pd.read_csv(data_path)
    # Drop rows with NaN
    df = df.dropna(subset=['clean_text', 'sentiment_score'])
    
    # Stratified Split
    X = df['clean_text'].astype(str)
    y = df['sentiment_score']
    
    print(f"Data Loaded: {len(X)} rows")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Initialize Manager
    dl_manager = DLManager(vector_size=100, max_len=100, hidden_dim=128)
    
    # 2. Train Word2Vec
    # Use full corpus (train+test) for better embeddings or just train? 
    # Usually better to use all available text for unsupervised embedding training
    print("Training Word2Vec on full corpus...")
    dl_manager.train_w2v(X)
    
    # 3. Train BiLSTM
    print("Training BiLSTM...")
    # Further split train for validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    model = dl_manager.train_model(X_tr, y_tr, X_val, y_val, epochs=5, batch_size=64, lr=0.001)
    
    # 4. Evaluate
    print("\n--- BiLSTM Evaluation ---")
    report = dl_manager.evaluate(X_test, y_test)
    print(report)

if __name__ == "__main__":
    main()
