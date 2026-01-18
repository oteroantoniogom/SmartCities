# Guía de Exportación para Modelos Deep Learning (`raa_phase3_deep_learning.ipynb`)

Si en el futuro decide utilizar modelos de Deep Learning (ej. BiLSTM, BERT) en producción, utilice estos snippets para guardarlos y cargarlos correctamente.

## 1. TensorFlow / Keras (ej. BiLSTM)

### Guardar Modelo y Tokenizer
```python
# Asumiendo que 'model' es su modelo Keras entrenado y 'tokenizer' es su TextVectorization/Tokenizer

import pickle
from pathlib import Path

# 1. Guardar arquitectura y pesos
models_dir = Path("../models")
models_dir.mkdir(parents=True, exist_ok=True)

model.save(models_dir / "sentiment_bilstm.keras")

# 2. Guardar Tokenizer (necesario para preprocesar input igual al training)
# Si usó Tokenizer de keras.preprocessing:
with open(models_dir / "tokenizer.pickle", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("Modelo y tokenizer guardados.")
```

### Cargar en `api.py`
```python
from tensorflow.keras.models import load_model
import pickle

# Cargar modelo
dl_model = load_model("models/sentiment_bilstm.keras")

# Cargar tokenizer
with open("models/tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

# Uso
input_text = ["Queja sobre el servicio"]
seq = tokenizer.texts_to_sequences(input_text)
padded = pad_sequences(seq, maxlen=...)
prediction = dl_model.predict(padded)
```

## 2. PyTorch (ej. Transformer/BERT)

### Guardar
```python
import torch

# Guardar state dict (recomendado)
torch.save(model.state_dict(), models_dir / "bert_sentiment.pth")

# Guardar tokenizer (si es HuggingFace)
tokenizer.save_pretrained(models_dir / "bert_tokenizer")
```

### Cargar
```python
# Inicializar arquitectura igual al training
model = BertClassifier(...) 
model.load_state_dict(torch.load("models/bert_sentiment.pth"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("models/bert_tokenizer")
```
