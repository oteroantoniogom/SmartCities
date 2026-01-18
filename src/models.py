from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb

class SentimentClassifier:
    def __init__(self, model_type='logreg', vectorizer_type='tfidf', max_features=5000):
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type 
        self.max_features = max_features
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # Vectorizer
        if self.vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2))
    
        elif self.vectorizer_type == 'count':
            vectorizer = CountVectorizer(max_features=self.max_features, ngram_range=(1, 2), dtype=np.float64)
    
        elif self.vectorizer_type == 'binary':
            vectorizer = CountVectorizer(max_features=self.max_features, ngram_range=(1, 2), binary=True, dtype=np.float64)
    
        elif self.vectorizer_type == 'char_tfidf':
            vectorizer = TfidfVectorizer(max_features=self.max_features, analyzer='char_wb', ngram_range=(3, 5))
    
        elif self.vectorizer_type == 'hashing':
            vectorizer = HashingVectorizer(n_features=self.max_features or 16384, ngram_range=(1, 2), alternate_sign=False)
            
        else:
            raise ValueError(f"Unknown vectorizer_type: {self.vectorizer_type}")
        
        # Classifier
        if self.model_type == 'logreg':
            clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        elif self.model_type == 'svm':
            clf = LinearSVC(class_weight='balanced', random_state=42, dual='auto')
        elif self.model_type == 'rf':
            clf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
        elif self.model_type == 'nb':
            clf = MultinomialNB()
        elif self.model_type == 'gb':
            clf = GradientBoostingClassifier(random_state=42)
        elif self.model_type == 'xgb':
            clf = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        elif self.model_type == 'lgbm':
            clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        return Pipeline([
            ('vect', vectorizer),
            ('clf', clf)
        ])

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return {"accuracy": acc, "y_pred": y_pred}

    def save(self, path):
        joblib.dump(self.pipeline, path)

    def load(self, path):
        self.pipeline = joblib.load(path)