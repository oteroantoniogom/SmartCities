from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import ollama
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# Para Secuencias (Word2Vec)
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Pooling: Usamos el último estado oculto o max-pooling
        # Aquí hacemos Max Pooling sobre la dimensión de la secuencia
        out, _ = torch.max(lstm_out, dim=1) 
        out = self.dropout(out)
        return self.fc(out)

# Para Vectores Semánticos (Transformers)
class SemanticMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(SemanticMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)


# Dataset flexible que se adapta según el tipo de embedding
class UniversalDataset(Dataset):
    def __init__(self, texts, labels, embedding_type='w2v', model_ref=None, max_len=100):
        self.texts = texts
        self.labels = labels
        self.embedding_type = embedding_type
        self.model_ref = model_ref # Puede ser w2v model o sentence-transformer model
        self.max_len = max_len
        self.label_map = {'bad': 0, 'neutral': 1, 'good': 2}
        
        # Pre-computar embeddings para Transformers (para no hacerlo en cada época)
        if self.embedding_type in ['transformer', 'ollama']:
            print(f"Pre-computando embeddings ({self.embedding_type})... esto puede tardar.")
            self.cached_vectors = self._encode_all()

    def _encode_all(self):
        # Lógica para HuggingFace
        if self.embedding_type == 'transformer':
            return self.model_ref.encode(list(self.texts), show_progress_bar=True)
        
        # Lógica para Ollama
        elif self.embedding_type == 'ollama':
            vecs = []
            for t in tqdm(self.texts, desc="Ollama Embedding"):
                # Llamada a la API local de Ollama
                resp = ollama.embeddings(model=self.model_ref, prompt=str(t))
                vecs.append(resp['embedding'])
            return np.array(vecs)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        label = torch.tensor(self.label_map[self.labels.iloc[idx]], dtype=torch.long)
        
        # Flujo W2V (Dinámico)
        if self.embedding_type == 'w2v':
            text = str(self.texts.iloc[idx]).split()
            if len(text) > self.max_len: text = text[:self.max_len]
            vectors = [self.model_ref.wv[w] for w in text if w in self.model_ref.wv]
            if len(vectors) == 0: vectors = [np.zeros(self.model_ref.vector_size)]
            # Padding
            while len(vectors) < self.max_len:
                vectors.append(np.zeros(self.model_ref.vector_size))
            return {'features': torch.tensor(np.array(vectors), dtype=torch.float32), 'label': label}
        
        # Flujo Transformer/Ollama (Estático/Cached)
        else:
            return {'features': torch.tensor(self.cached_vectors[idx], dtype=torch.float32), 'label': label}

class AdvancedDLManager:
    def __init__(self, strategy='w2v', model_name=None):
        """
        strategy: 'w2v', 'transformer', 'ollama'
        model_name: 
            - Si strategy='transformer': 'all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5'
            - Si strategy='ollama': 'embedding-gemma', 'nomic-embed-text'
        """
        self.strategy = strategy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = None
        self.classifier = None
        self.vector_dim = 100 # Default w2v
        
        if strategy == 'transformer':
            print(f"Cargando Sentence Transformer: {model_name}")
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
            self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
            
        elif strategy == 'ollama':
            print(f"Conectando a Ollama: {model_name}")
            self.embedding_model = model_name # Solo guardamos el nombre string
            # Hacemos una prueba dummy para sacar dimensión
            dummy = ollama.embeddings(model=model_name, prompt="hello")
            self.vector_dim = len(dummy['embedding'])
            
        print(f"Estrategia: {strategy} | Dimensión Vectores: {self.vector_dim}")

    def train_w2v(self, texts, vector_size=100):
        if self.strategy != 'w2v': return
        tokenized = [str(t).split() for t in texts]
        self.embedding_model = Word2Vec(sentences=tokenized, vector_size=vector_size, window=5, min_count=2, workers=4)
        self.vector_dim = vector_size

    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, lr=1e-3):
        # 1. Crear Datasets
        train_ds = UniversalDataset(X_train, y_train, self.strategy, self.embedding_model)
        val_ds = UniversalDataset(X_val, y_val, self.strategy, self.embedding_model)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        # 2. Seleccionar Arquitectura
        if self.strategy == 'w2v':
            # Secuencia -> BiLSTM
            self.classifier = BiLSTM(self.vector_dim, hidden_dim=128, output_dim=3).to(self.device)
        else:
            # Vector Único -> MLP
            self.classifier = SemanticMLP(self.vector_dim, hidden_dim=256, output_dim=3).to(self.device)
            
        # 3. Loop de Entrenamiento
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        
        print(f"\nEntrenando {self.classifier.__class__.__name__} en {self.device}...")
        
        history = []
        for epoch in range(epochs):
            self.classifier.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
                X_b = batch['features'].to(self.device)
                y_b = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                out = self.classifier(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validación
            acc = self._evaluate_loader(val_loader)
            print(f"Ep {epoch+1} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {acc:.4f}")
            history.append(acc)
            
        return history

    def evaluate(self, X_test, y_test):
        test_ds = UniversalDataset(X_test, y_test, self.strategy, self.embedding_model)
        test_loader = DataLoader(test_ds, batch_size=64)
        
        self.classifier.eval()
        preds, true_lbls = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                X_b = batch['features'].to(self.device)
                out = self.classifier(X_b)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                true_lbls.extend(batch['label'].numpy())
                
        map_l = {0:'bad', 1:'neutral', 2:'good'}
        return classification_report([map_l[i] for i in true_lbls], [map_l[i] for i in preds])

    def _evaluate_loader(self, loader):
        self.classifier.eval()
        p, t = [], []
        with torch.no_grad():
            for batch in loader:
                X_b = batch['features'].to(self.device)
                p.extend(torch.argmax(self.classifier(X_b), dim=1).cpu().numpy())
                t.extend(batch['label'].numpy())
        return accuracy_score(t, p)