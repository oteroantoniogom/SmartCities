import pandas as pd
import numpy as np
import re
import spacy
from tqdm import tqdm
from config import DATA_PATHS, UNIFIED_COLUMNS

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
        
        # 1. Stores (Spanish)
        print("Loading Stores Complaints...")
        try:
            raw_data["stores"] = pd.read_excel(DATA_PATHS["stores"])
        except Exception as e:
            print(f"Error loading stores: {e}")

        # 2. Financial (English)
        print("Loading Financial Complaints...")
        try:
            raw_data["financial"] = pd.read_csv(DATA_PATHS["financial"])
        except Exception as e:
            print(f"Error loading financial: {e}")

        # 3. University (English)
        print("Loading University Complaints...")
        try:
            raw_data["university"] = pd.read_csv(DATA_PATHS["university"])
        except Exception as e:
            print(f"Error loading university: {e}")

        # 4. Amazon (Multilingual)
        print("Loading Amazon Reviews...")
        try:
            raw_data["amazon"] = pd.read_csv(DATA_PATHS["amazon"])
        except Exception as e:
            print(f"Error loading amazon: {e}")

        # 5. News (English)
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

        # --- Amazon ---
        if "amazon" in raw_data:
            df = raw_data["amazon"].copy()
            # Assuming cols: 'review_id', 'product_id', 'reviewer_id', 'stars', 'review_body', 'review_title', 'language', 'product_category'
            # Note: Need to verify actual columns. Based on Kaggle dataset 'amazon_reviews_multi':
            # usually: review_id, stars, review_body, review_title, language, product_category
            
            # Map sentiment
            def map_amazon_sentiment(stars):
                if stars <= 2: return "bad"
                if stars == 3: return "neutral"
                return "good"
            
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
                "category": df.get('product_category', 'product'),
                "sentiment_score": df['sentiment_score'],
                "timestamp": None # Amazon dataset often lacks ts or needs parsing
            })
            unified_frames.append(norm_df)

        # --- Stores (Spanish) ---
        if "stores" in raw_data:
            df = raw_data["stores"].copy()
            # Columns likely: 'complain_id', 'complain_text', ... need to inspect
            # We'll assume first text column is content if names unknown, or 'comentario'
            # For now, let's look for likely text candidates
            text_col = next((c for c in df.columns if 'text' in c.lower() or 'coment' in c.lower() or 'queja' in c.lower()), df.columns[0])
            
            norm_df = pd.DataFrame({
                "id": df.index,
                "text": df[text_col],
                "source": "stores",
                "language": "es",
                "category": "service", # Default
                "sentiment_score": "bad", # It's a complaint
                "timestamp": None
            })
            unified_frames.append(norm_df)

        # --- Financial (English) ---
        if "financial" in raw_data:
            df = raw_data["financial"].copy()
            # Likely cols: 'Date received', 'Product', 'Issue', 'Consumer complaint narrative'
            text_col = 'Consumer complaint narrative' if 'Consumer complaint narrative' in df.columns else df.columns[-1]
            if 'Consumer complaint narrative' in df.columns:
                df = df.dropna(subset=['Consumer complaint narrative'])
            
            norm_df = pd.DataFrame({
                "id": df.get('Complaint ID', df.index),
                "text": df.get('Consumer complaint narrative', df[df.columns[0]]),
                "source": "financial",
                "language": "en",
                "category": df.get('Product', 'finance'),
                "sentiment_score": "bad",
                "timestamp": df.get('Date received', None)
            })
            unified_frames.append(norm_df)

        # --- University (English) ---
        if "university" in raw_data:
            df = raw_data["university"].copy()
            # Inspect structure later, assume 'Complaint' column
            text_col = next((c for c in df.columns if 'complaint' in c.lower() or 'text' in c.lower()), df.columns[0])

            norm_df = pd.DataFrame({
                "id": df.index,
                "text": df[text_col],
                "source": "university",
                "language": "en",
                "category": "education",
                "sentiment_score": "bad",
                "timestamp": None
            })
            unified_frames.append(norm_df)

        # --- News (English) ---
        if "news" in raw_data:
            df = raw_data["news"].copy()
            # We explicitly named columns ["sentiment", "text"] in load_data
            
            norm_df = pd.DataFrame({
                "id": df.index,
                "text": df["text"],
                "source": "news",
                "language": "en",
                "category": "news",
                "sentiment_score": df["sentiment"].map({"positive": "good", "negative": "bad", "neutral": "neutral"}).fillna(df["sentiment"]),
                "timestamp": None
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
