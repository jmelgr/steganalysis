import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt

def evaluate_logistic_benchmark():
    features_dir = "data/features"
    X_test_path = os.path.join(features_dir, "X_test.npy")
    y_test_path = os.path.join(features_dir, "y_test.npy")
    X_train_path = os.path.join(features_dir, "X_train.npy")
    y_train_path = os.path.join(features_dir, "y_train.npy")

    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        print("Error: No se encontraron los archivos de prueba en data/features/.")
        return

    print("Cargando conjuntos de datos de prueba y entrenamiento para el Benchmark...")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    
    model_path = "models/logistic_benchmark_model.pkl"
    os.makedirs("models", exist_ok=True)

    print("Entrenando Regresión Logística (Benchmark Lineal)...")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )
    
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Modelo benchmark guardado en {model_path}")

    print("Ejecutando inferencia del benchmark sobre el conjunto de prueba...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)

    print("\n" + "="*50)
    print("      RESULTADOS DEL BENCHMARK (REGRESIÓN LOGÍSTICA)")
    print("="*50)
    print(f" Exactitud (Accuracy):           {acc * 100:.2f}%")
    print(f" Precisión (Precision):          {precision * 100:.2f}%")
    print(f" Exhaustividad (Recall/TPR):     {recall * 100:.2f}%")
    print(f" F1-Score:                       {f1 * 100:.2f}%")
    print(f" Área Bajo la Curva ROC (AUC):   {auc:.4f}")
    print(f" Tasa de Falsos Positivos (FPR): {fpr * 100:.2f}%")
    print("="*50)
    print(f" Matriz de Confusión -> TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print("="*50)

    os.makedirs("reports/figures", exist_ok=True)
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_arr, tpr_arr, color='red', lw=2, label=f'Benchmark Logístico (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
    plt.title('Evaluación Benchmark: Curva ROC (Regresión Logística)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    roc_output_path = "reports/figures/roc_curve_benchmark.png"
    plt.savefig(roc_output_path, dpi=300)
    plt.close()
    print(f" Gráfica ROC del benchmark guardada en: {roc_output_path}")

if __name__ == "__main__":
    evaluate_logistic_benchmark()
