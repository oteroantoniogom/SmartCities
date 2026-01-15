import pandas as pd
import numpy as np
import re
import spacy
from typing import Literal
from tqdm import tqdm
from config import DATA_PATHS, UNIFIED_COLUMNS
from textblob import TextBlob
from collections import Counter

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
    def __init__(self, enable_spell_correction: bool = True, enable_entity_recognition: bool = True):
        print("Initializing NLP models...")
        self.enable_spell_correction = enable_spell_correction
        self.enable_entity_recognition = enable_entity_recognition
        try:
            # Enable NER if entity recognition is needed
            disable_components = ["parser"] if enable_entity_recognition else ["parser", "ner"]
            self.nlp_en = spacy.load("en_core_web_sm", disable=disable_components)
            self.nlp_es = spacy.load("es_core_news_sm", disable=disable_components)
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

    def clean_text(self, text: str) -> str:
        """Text cleaning with noise removal.
        
        Removes HTML, URLs, emails, phone numbers, special characters,
        extra whitespace, and normalizes text.
        
        Args:
            text: Raw text string to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        
        # URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Phone numbers (basic patterns)
        text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' ', text)
        
        # Special characters but keep apostrophes for contractions
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Multiple whitespace to single space
        text = re.sub(r'\s+', ' ', text)

        # Remove emojis
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Normalize case
        text = text.lower()
        
        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def correct_spelling(self, text: str) -> str:
        """Correct spelling errors using TextBlob.
        
        Args:
            text: Text string to correct
            
        Returns:
            Spell-corrected text string
        """
        if not self.enable_spell_correction or not text:
            return text
        
        try:
            blob = TextBlob(text)
            return str(blob.correct())
        except Exception:
            return text

    def tokenize_and_lemmatize(self, text: str, language: str = "en") -> tuple[list[str], list[str], list[str]]:
        """Tokenize and lemmatize text using spaCy.
        
        Args:
            text: Text string to process
            language: Language code ('en' or 'es')
            
        Returns:
            Tuple of (tokens, lemmas, pos_tags)
        """
        if not text:
            return [], [], []
        
        nlp = self.nlp_en if language == "en" else self.nlp_es
        if nlp is None:
            # Fallback to simple tokenization
            tokens = text.split()
            return tokens, tokens, []
        
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_space]
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        
        return tokens, lemmas, pos_tags

    def extract_entities(self, text: str, language: str = "en") -> list[dict[str, str]]:
        """Extract named entities from text.
        
        Args:
            text: Text string to process
            language: Language code ('en' or 'es')
            
        Returns:
            List of dictionaries with entity text and label
        """
        if not self.enable_entity_recognition or not text:
            return []
        
        nlp = self.nlp_en if language == "en" else self.nlp_es
        if nlp is None or "ner" in nlp.disabled:
            return []
        
        doc = nlp(text)
        entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
        return entities

    def remove_duplicates(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Remove duplicate entries based on text content.
        
        Args:
            df: DataFrame to deduplicate
            text_column: Column name containing text to compare
            
        Returns:
            DataFrame with duplicates removed
        """
        original_count = len(df)
        df = df.drop_duplicates(subset=[text_column], keep="first")
        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate entries")
        return df

    def process_corpus(self, df: pd.DataFrame, apply_spell_correction: bool = False) -> pd.DataFrame:
        """Apply complete preprocessing pipeline: cleaning, deduplication, 
        tokenization, lemmatization, and entity recognition.
        
        Args:
            df: DataFrame containing text data
            apply_spell_correction: Whether to apply spell correction (slow)
            
        Returns:
            Processed DataFrame with additional columns
        """
        print(f"Processing corpus with {len(df)} documents...")
        
        # 1. Remove duplicates
        print("Removing duplicates...")
        df = self.remove_duplicates(df, "text")
        
        # 2. Clean text
        print("Cleaning text...")
        df['clean_text'] = df['text'].apply(self.clean_text)
        
        # 3. Spell correction (optional, can be slow)
        if apply_spell_correction:
            print("Applying spell correction...")
            df['clean_text'] = df['clean_text'].apply(self.correct_spelling)
        
        # 4. Tokenization and lemmatization
        print("Tokenizing and lemmatizing...")
        results = []
        entities_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            text = row['clean_text']
            lang = row.get('language', 'en')
            
            tokens, lemmas, pos_tags = self.tokenize_and_lemmatize(text, lang)
            entities = self.extract_entities(text, lang)
            
            results.append({
                'tokens': tokens,
                'lemmas': lemmas,
                'pos_tags': pos_tags
            })
            entities_list.append(entities)
        
        df['tokens'] = [r['tokens'] for r in results]
        df['lemmas'] = [r['lemmas'] for r in results]
        df['pos_tags'] = [r['pos_tags'] for r in results]
        df['entities'] = entities_list
        
        # 5. Create joined lemmas for vectorization
        df['lemmas_text'] = df['lemmas'].apply(lambda x: ' '.join(x) if x else '')
        
        print(f"Processing complete. Final corpus size: {len(df)} documents")
        return df
