import pandas as pd
import numpy as np
import re
import spacy
from typing import Literal
from tqdm import tqdm
from config import DATA_PATHS, UNIFIED_COLUMNS

SentimentLabel = Literal["bad", "neutral", "good", "unknown"]


def map_amazon_sentiment(stars: int | float | None) -> SentimentLabel:
    """Map star ratings to sentiment labels.
    
    Args:
        stars: Star rating value (1-5). None or invalid values return 'unknown'.
        
    Returns:
        'bad' for 1-2 stars, 'neutral' for 3 stars, 'good' for 4-5 stars,
        'unknown' for invalid/missing values.
    """
    if stars is None or not isinstance(stars, (int, float)):
        return "unknown"
    
    try:
        star_value = float(stars)
        if star_value <= 2:
            return "bad"
        if star_value <= 3:
            return "neutral"
        return "good"
    except (ValueError, TypeError):
        return "unknown"


class SmartUrbanPreprocessor:
    def __init__(self):
        print("Initializing NLP models...")
        try:
            self.nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self.nlp_es = spacy.load("es_core_news_sm", disable=["parser", "ner"])
        except OSError:
            print("Spacy models not found. Please run: python -m spacy download en_core_web_sm && python -m spacy download es_core_news_sm")
            self.nlp_en = None
            self.nlp_es = None

    def load_data(self):
        """Loads all raw datasets into a dictionary of DataFrames."""
        raw_data = {}
        
        # Stores (Spanish)
        print("Loading Stores Complaints...")
        try:
            raw_data["stores"] = pd.read_excel(DATA_PATHS["stores"])
        except Exception as e:
            print(f"Error loading stores: {e}")

        # Financial (English)
        print("Loading Financial Complaints...")
        try:
            raw_data["financial"] = pd.read_csv(DATA_PATHS["financial"])
        except Exception as e:
            print(f"Error loading financial: {e}")

        # University (English)
        print("Loading University Complaints...")
        try:
            raw_data["university"] = pd.read_csv(DATA_PATHS["university"])
        except Exception as e:
            print(f"Error loading university: {e}")

        # Amazon (Multilingual)
        print("Loading Amazon Reviews...")
        try:
            raw_data["amazon"] = pd.read_csv(DATA_PATHS["amazon"])
        except Exception as e:
            print(f"Error loading amazon: {e}")

        # News (English)
        print("Loading News Sentiments...")
        try:
            # File has no header, col 0 = sentiment, col 1 = text
            raw_data["news"] = pd.read_csv(DATA_PATHS["news"], header=None, names=["sentiment", "text"])
        except UnicodeDecodeError:
            print("Falling back to ISO-8859-1 encoding for news")
            raw_data["news"] = pd.read_csv(DATA_PATHS["news"], header=None, names=["sentiment", "text"], encoding="ISO-8859-1")
            print("Removing non-UTF-8 characters from news text")
            raw_data["news"]["text"] = raw_data["news"]["text"].str.encode('utf-8', errors='ignore').str.decode('utf-8')
            print("Removing non-UTF-8 characters from news sentiment")
            raw_data["news"]["sentiment"] = raw_data["news"]["sentiment"].str.encode('utf-8', errors='ignore').str.decode('utf-8')
            print("Converted news to UTF-8")
        except Exception as e:
            print(f"Error loading news: {e}")
            
        return raw_data


    def normalize(self, raw_data):
        """Unifies all datasets into a single schema."""
        unified_frames = []

        # Amazon
        if "amazon" in raw_data:
            df = raw_data["amazon"].copy()
            
            if 'stars' in df.columns:
                df['sentiment_score'] = df['stars'].apply(map_amazon_sentiment)
            else:
                df['sentiment_score'] = 'unknown'

            # Normalize
            norm_df = pd.DataFrame({
                "id": df.get('review_id', df.index),
                "text": df.get('review_body', "") + " " + df.get('review_title', ""),
                "source": "amazon",
                "language": df.get('language', 'unknown'),
                "sentiment_score": df['sentiment_score'],
            })
            unified_frames.append(norm_df)

        # Stores (Spanish)
        if "stores" in raw_data:
            df = raw_data["stores"].copy()
            text_col = next((c for c in df.columns if 'text' in c.lower() or 'coment' in c.lower() or 'queja' in c.lower()), df.columns[0])
            
            norm_df = pd.DataFrame({
                "id": df.index,
                "text": df[text_col],
                "source": "stores",
                "language": "es",
                "sentiment_score": "bad", # It's a complaint
            })
            unified_frames.append(norm_df)

        # Financial (English)
        if "financial" in raw_data:
            df = raw_data["financial"].copy()
            text_col = 'Consumer complaint narrative' if 'Consumer complaint narrative' in df.columns else df.columns[-1]
            if 'Consumer complaint narrative' in df.columns:
                df = df.dropna(subset=['Consumer complaint narrative'])
            
            norm_df = pd.DataFrame({
                "id": df.get('Complaint ID', df.index),
                "text": df.get('Consumer complaint narrative', df[df.columns[0]]),
                "source": "financial",
                "language": "en",
                "sentiment_score": "bad",
            })
            unified_frames.append(norm_df)

        # University (English)
        if "university" in raw_data:
            df = raw_data["university"].copy()
            text_col = next((c for c in df.columns if 'complaint' in c.lower() or 'text' in c.lower()), df.columns[0])

            norm_df = pd.DataFrame({
                "id": df.index,
                "text": df[text_col],
                "source": "university",
                "language": "en",
                "sentiment_score": "bad",
            })
            unified_frames.append(norm_df)

        # News (English)
        if "news" in raw_data:
            df = raw_data["news"].copy()
            norm_df = pd.DataFrame({
                "id": df.index,
                "text": df["text"],
                "source": "news",
                "language": "en",
                "sentiment_score": df["sentiment"].map({"positive": "good", "negative": "bad", "neutral": "neutral"}).fillna(df["sentiment"]),
            })
            unified_frames.append(norm_df)

        return pd.concat(unified_frames, ignore_index=True)

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # 1. Lowercase
        text = text.lower()
        # 2. HTML (basic)
        text = re.sub(r'<.*?>', '', text)
        # 3. URLs
        text = re.sub(r'http\S+', '', text)
        # 4. Special chars (keep basic punctuation for sentence structure? maybe redundant if lemmatizing)
        text = re.sub(r'[^\w\s\.]', '', text)
        return text.strip()

    def process_corpus(self, df):
        """Applies cleaning and lemmatization."""
        print("Cleaning text...")
        df['clean_text'] = df['text'].apply(self.clean_text)
        
        # Lemmatization (Sample to avoid waiting forever on full corpus if massive)
        # For now, just tokenization logic or basic processing
        
        return df
