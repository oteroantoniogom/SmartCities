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
        print("Data not found!")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['clean_text', 'sentiment_score'])
    
    X = df['clean_text']
    y = df['sentiment_score']
    
    # Train final model on full balanced dataset for production?
    # Or stick to train/test split? Let's use all data for the interactive app.
    print("Training production (SVM) model on full balanced corpus...")
    
    clf = SentimentClassifier(model_type='svm')
    clf.train(X, y)
    
    # Save
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    save_path = models_dir / "sentiment_svm.joblib"
    
    clf.save(str(save_path))
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
