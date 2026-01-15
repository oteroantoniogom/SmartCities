from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# File Paths
DATA_PATHS = {
    "stores": DATA_DIR / "stores_complaints.xlsx",
    "financial": DATA_DIR / "financial_complaints.csv",
    "university": DATA_DIR / "university_complaints.csv",
    "amazon": DATA_DIR / "amazon_multilingual_reviews.csv",
    "news": DATA_DIR / "news_sentiments.csv"
}

# Schema Columns
UNIFIED_COLUMNS = [
    "id", "text", "source", "language", "sentiment_score",
]

# Random Seed
SEED = 42
