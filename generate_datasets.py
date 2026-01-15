import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from config import DATA_PATHS
from preprocessor import SmartUrbanPreprocessor

def main():
    processor = SmartUrbanPreprocessor()
    
    # 1. Load & Normalize
    raw_data = processor.load_data()
    unified_df = processor.normalize(raw_data)
    
    # 2. Clean
    clean_df = processor.process_corpus(unified_df)
    
    print(f"\nTotal Unified Rows: {len(clean_df)}")
    print("Sentiment counts:\n", clean_df['sentiment_score'].value_counts())
    
    # Ensure data directory exists
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # 3. Save Full Corpus
    output_path_full = output_dir / "processed_corpus_full.csv"
    clean_df.to_csv(output_path_full, index=False)
    print(f"\n[SAVED] Full Corpus: {len(clean_df)} rows -> {output_path_full}")
    
    # 4. Create & Save Balanced Corpus (Downsample Financial)
    # Target size: Match Amazon (~30k) or just cap it at 30k
    target_size = 30000 
    
    df_financial = clean_df[clean_df['source'] == 'financial']
    df_others = clean_df[clean_df['source'] != 'financial']
    
    print(f"\nBalancing: {len(df_financial)} financial rows -> target {target_size}")
    
    if len(df_financial) > target_size:
        df_financial_balanced = df_financial.sample(n=target_size, random_state=42)
    else:
        df_financial_balanced = df_financial
        
    balanced_df = pd.concat([df_others, df_financial_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path_balanced = output_dir / "processed_corpus_balanced.csv"
    balanced_df.to_csv(output_path_balanced, index=False)
    print(f"[SAVED] Balanced Corpus: {len(balanced_df)} rows -> {output_path_balanced}")
    print("Balanced Sentiment counts:\n", balanced_df['sentiment_score'].value_counts())

if __name__ == "__main__":
    main()
