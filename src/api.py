from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models import SentimentClassifier
from rag_engine import RAGEngine
from llm_client import OllamaClient

app = FastAPI(title="Smart Urban Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev, or specific ["http://localhost:4321"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Models
# Initialize globally to load into memory once
classifier = None
rag_engine = None
llm_client = None

def load_models():
    global classifier, rag_engine, llm_client
    
    # 1. Load Classifier
    print("Loading Classifier...")
    model_path = Path("models/sentiment_lgbm.joblib")
    if model_path.exists():
        classifier = SentimentClassifier(model_type='lgbm')
        classifier.load(str(model_path))
    else:
        print("Warning: Classifier model not found. /classify will fail.")

    # 2. Load RAG
    print("Loading RAG Engine...")
    data_path = Path("data/processed_corpus.csv")
    if data_path.exists():
        # Using full corpus for better retrieval
        rag_engine = RAGEngine(data_path=str(data_path))
    else:
        print("Warning: Data for RAG not found.")
        
    # 3. Load LLM Client
    llm_client = OllamaClient()

@app.on_event("startup")
async def startup_event():
    load_models()

# Schemas
class TextPayload(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str

class RAGPayload(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str
    context: list

# Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=PredictResponse)
def classify(payload: TextPayload):
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    prediction = classifier.predict([payload.text])[0]
    return {"sentiment": prediction}

@app.post("/rag", response_model=RAGResponse)
def rag_chat(payload: RAGPayload):
    if not rag_engine or not llm_client:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # 1. Retrieve
    results = rag_engine.retrieve(payload.query, k=3)
    
    # 2. Generate
    prompt = rag_engine.format_prompt(payload.query, results)
    answer = llm_client.generate(prompt)
    
    return {
        "answer": answer,
        "context": [r['text'] for r in results]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
