import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models import SentimentClassifier

def main():
    print("Loading data...")
    data_path = Path("data/processed_corpus_balanced.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run generate_datasets.py first.")
        return

    df = pd.read_csv(data_path)
    # Drop rows with NaN
    df = df.dropna(subset=['clean_text', 'sentiment_score'])
    
    print(f"Data shape: {df.shape}")
    
    X = df['clean_text']
    y = df['sentiment_score']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = ['logreg', 'svm', 'rf']
    results = {}
    
    for m in models:
        print(f"\nTraining {m.upper()}")
        clf = SentimentClassifier(model_type=m)
        clf.train(X_train, y_train)
        res = clf.evaluate(X_test, y_test)
        results[m] = res['accuracy']
        
    print("\n\n=== SUMMARY ACCURACY ===")
    for m, acc in results.items():
        print(f"{m.upper()}: {acc:.4f}")

if __name__ == "__main__":
    main()
