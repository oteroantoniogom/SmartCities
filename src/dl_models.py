import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, classification_report

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2vec_model, max_len=100):
        self.texts = texts
        self.labels = labels
        self.w2v = word2vec_model
        self.max_len = max_len
        self.label_map = {'bad': 0, 'neutral': 1, 'good': 2}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]).split()
        # Pad or truncate
        if len(text) > self.max_len:
            text = text[:self.max_len]
        
        # Convert to vectors
        vectors = []
        for word in text:
            if word in self.w2v.wv:
                vectors.append(self.w2v.wv[word])
            else:
                vectors.append(np.zeros(self.w2v.vector_size))
                
        # Padding
        if len(vectors) < self.max_len:
            padding = [np.zeros(self.w2v.vector_size)] * (self.max_len - len(vectors))
            vectors.extend(padding)
            
        return {
            'features': torch.tensor(np.array(vectors), dtype=torch.float32),
            'label': torch.tensor(self.label_map[self.labels[idx]], dtype=torch.long)
        }

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Global max pooling (or assume last hidden state)
        # Let's take the last output for now (or max pool)
        # lstm_out: [batch, seq_len, hidden*2]
        
        # Max pooling over sequence
        out, _ = torch.max(lstm_out, dim=1)
        out = self.dropout(out)
        return self.fc(out)

class DLManager:
    def __init__(self, vector_size=100, max_len=100, hidden_dim=64):
        self.vector_size = vector_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.w2v_model = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def train_w2v(self, texts):
        print("Training Word2Vec...")
        # Simple tokenization by split (preprocessing usually handled before)
        tokenized_sentences = [str(t).split() for t in texts]
        self.w2v_model = Word2Vec(sentences=tokenized_sentences, 
                                 vector_size=self.vector_size, 
                                 window=5, min_count=2, workers=4)
        print("Word2Vec trained.")

    def train_model(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, lr=0.001):
        if self.w2v_model is None:
            raise ValueError("Train Word2Vec first!")
            
        train_ds = TextDataset(X_train.values, y_train.values, self.w2v_model, self.max_len)
        val_ds = TextDataset(X_val.values, y_val.values, self.w2v_model, self.max_len)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        self.model = BiLSTM(self.vector_size, self.hidden_dim, 3).to(self.device)
        criterion = nn.CrossEntropyLoss() # Balanced weights could be added here
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                feats = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for batch in val_loader:
                    feats = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    outputs = self.model(feats)
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            acc = accuracy_score(val_true, val_preds)
            print(f"Epoch {epoch+1} Loss: {train_loss/len(train_loader):.4f} Val Acc: {acc:.4f}")
            
        return self.model

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained!")
            
        test_ds = TextDataset(X_test.values, y_test.values, self.w2v_model, self.max_len)
        test_loader = DataLoader(test_ds, batch_size=64)
        
        self.model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                feats = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(feats)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
        
        idx_to_label = {0: 'bad', 1: 'neutral', 2: 'good'}
        pred_labels = [idx_to_label[p] for p in all_preds]
        true_labels = [idx_to_label[t] for t in all_true]
        
        return classification_report(true_labels, pred_labels)
