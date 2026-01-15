import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from rag_engine import RAGEngine
from llm_client import OllamaClient

def main():
    print("Initializing RAG Engine...")
    # Adjust path if running from root
    data_path = Path("data/processed_corpus_balanced.csv")
    if not data_path.exists():
        print("Data not found. Run from project root.")
        return

    rag = RAGEngine(data_path=str(data_path))
    
    # Check for Ollama
    print("Initializing LLM Client...")
    llm = OllamaClient(model="deepseek-r1:8b-0528-qwen3-q4_K_M") # Default from SPEC, configurable
    
    llm.chat_loop(rag)

if __name__ == "__main__":
    main()
