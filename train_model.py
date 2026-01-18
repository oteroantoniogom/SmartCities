import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.models import SentimentClassifier

def main():
    # 1. Cargar Dataset Completo
    data_path = Path("data/processed_corpus.csv")
    if not data_path.exists():
        # Fallback for running from root or src
        data_path = Path("../data/processed_corpus.csv")
    
    print(f"Cargando datos desde: {data_path.resolve()}")
    df_full = pd.read_csv(data_path)
    
    # Asegurar que tenemos las columnas necesarias
    # El reporte indica 'lemmas_text' como la feature ganadora
    df_full = df_full.dropna(subset=['lemmas_text', 'sentiment_score'])
    
    X = df_full['lemmas_text']
    y = df_full['sentiment_score']
    
    print(f"Total datos disponibles: {len(df_full)}")

    # 2. Split Riguroso (20% Test intocable)
    # Stratify asegura que la distribución de clases se mantenga
    X_train_raw, X_test_real, y_train_raw, y_test_real = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split realizado: Train={len(X_train_raw)}, Test={len(X_test_real)}")

    # 3. Balancear SOLO el Train
    print("Balanceando set de entrenamiento (Undersampling)...")
    train_df = pd.DataFrame({'text': X_train_raw, 'sentiment': y_train_raw})
    
    min_count = train_df['sentiment'].value_counts().min()
    balanced_train_df = train_df.groupby('sentiment').apply(
        lambda x: x.sample(min_count, random_state=42)
    ).reset_index(drop=True)
    
    X_train_bal = balanced_train_df['text']
    y_train_bal = balanced_train_df['sentiment']
    print(f"Train Balanceado: {len(X_train_bal)} muestras ({min_count} por clase)")

    # 4. Entrenar Modelo Ganador (LightGBM + CountVectorizer)
    print("\nEntrenando LightGBM con CountVectorizer...")
    # vectorizer_type='count' es clave según el reporte
    final_model = SentimentClassifier(model_type='lgbm', vectorizer_type='count')
    final_model.train(X_train_bal, y_train_bal)

    # 5. Evaluar en Test Real (Desbalanceado)
    print("\nEvaluación en Test Set Real (Original - NO Balanceado):")
    y_pred = final_model.predict(X_test_real)
    print(classification_report(y_test_real, y_pred))

    # 6. Guardar Modelo
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "sentiment_lgbm.joblib"
    
    final_model.save(save_path)
    print(f"\nModelo guardado exitosamente en: {save_path.resolve()}")

if __name__ == "__main__":
    main()
