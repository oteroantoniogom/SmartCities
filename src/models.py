import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class SentimentClassifier:
    def __init__(self, model_type='logreg', max_features=5000):
        self.model_type = model_type
        self.max_features = max_features
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # 1. Vectorizer
        vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2))
        
        # 2. Classifier
        if self.model_type == 'logreg':
            clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        elif self.model_type == 'svm':
            clf = LinearSVC(class_weight='balanced', random_state=42, dual='auto')
        elif self.model_type == 'rf':
            clf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        return Pipeline([
            ('tfidf', vectorizer),
            ('clf', clf)
        ])

    def train(self, X_train, y_train):
        print(f"Training {self.model_type}...")
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"{self.model_type.upper()} Results")
        return {"accuracy": acc, "report": report, "confusion_matrix": cm}

    def save(self, path):
        import joblib
        print(f"Saving model to {path}...")
        joblib.dump(self.pipeline, path)

    def load(self, path):
        import joblib
        print(f"Loading model from {path}...")
        self.pipeline = joblib.load(path)
