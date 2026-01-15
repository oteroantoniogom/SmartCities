# Smart Urban System - NLP Module Specification

## 1. Overview

This document serves as the living technical specification for the NLP module of the Smart Urban System. It tracks architectural decisions, data schemas, model choices, and justification for technical approaches.

## 2. Technical Stack

## 2. Technical Stack

- **Language**: Python (Code & Docs in English)
- **Development**: Jupyter Notebook (`raa_dev.ipynb`) -> Astro Web App
- **LLM Engine**:
  - **Inference**: Ollama running `deepseek-r1:8b-0528-qwen3-q4_K_M`
  - **Finetuning**: `FunctionGemma` via `Unsloth` (for RAG/Function Calling)
- **Embeddings**:
  - Gensim `Word2Vec` (Context-independent baseline)
  - Transformer-based (e.g., `sentence-transformers`) (Context-aware)
- **Classification**:
  - Scikit-learn (SVM/RandomForest)
  - Deep Learning (PyTorch/TensorFlow)

## 3. Data Pipeline

### Sources

1. **Stores Complaints (Spanish)**: [Kaggle Link](https://www.kaggle.com/datasets/patriciamora/quejas-ms-frecuentes-en-compras-por-internet-2022) (`stores_complaints.xlsx`)
2. **Financial Complaints (English)**: [Kaggle Link](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp/data) (`financial_complaints.csv`)
3. **University Complaints (English)**: [Kaggle Link](https://www.kaggle.com/datasets/omarsobhy14/university-students-complaints-and-reports) (`university_complaints.csv`)
4. **Amazon Multilingual Reviews**: [Kaggle Link](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi) (`amazon_multilingual_reviews.csv`)
5. **News Sentiments (English)**: [Kaggle Link](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) `news_sentiments.csv` (General sentiment analysis, **ISO-8859-1 encoding**, no header).

### Schema Normalization Strategy

The heterogeneous sources will be mapped to a unified `SmartUrbanComplaint` schema:

- `id`: Unique identifier
- `text`: Original content
- `source`: Origin dataset
- `language`: 'es' | 'en' (Explicit for Stores/Financial/University/News, Column-based for Amazon)
- `category`: Mapped unified topic (e.g., Infrastructure, Service, Security)
- `sentiment_score`:
  - Amazon: 1-2 (Bad), 3 (Neutral), 4-5 (Good)
  - Complaints (Stores, Financial, University): Default 'Bad'
  - News: As per dataset

## 4. Models & Algorithms

### Embeddings Comparison

- **Word2Vec**: Custom trained on the unified corpus to capture specific "urban" vocabulary.
- **Transformer**: Pre-trained multilingual model (likely `paraphrase-multilingual-MiniLM-L12-v2`) for semantic search.

### Classification & Sentiment

- **Stage 1 (Baseline)**: TF-IDF + Logistic Regression/SVM.
- **Stage 2 (Advanced)**: Fine-tuned Transformer head or simple Feed-Forward NN on top of embeddings.

### RAG & Conversational Agent

- **Retriever**: Vector DB (ChromaDB or FAISS) using Transformer embeddings.
- **Generator**:
  - **DeepSeek-R1** (General Chat)
  - **FunctionGemma** (Fine-tuned for specific "Smart City" actions/routing).
- **Tool Use**: The agent will be able to "query" the complaint database (e.g., "Show me recent complaints about potholes").

## 5. Web Application (Phase 6)

- **Framework**: Astro (as per `frontend-web.md`)
- **Styling**: Tailwind CSS
- **Integration**: Python backend acting as API for the Astro frontend.

## 5. Decision Log

- [Date]: Initialized project structure.
