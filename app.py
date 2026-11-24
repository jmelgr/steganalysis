import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
from PIL import Image
from scipy.stats import kurtosis, skew

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = str(BASE_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from transform_residuals import compute_spatial_residual

MODEL_PATH = BASE_DIR / "models" / "svm_optimized_model.pkl"
model = joblib.load(MODEL_PATH)


def _resolve_image_path(image) -> str:
    if isinstance(image, (str, Path)):
        return str(image)
    if isinstance(image, np.ndarray):
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        Image.fromarray(image).convert("L").save(tmp_path)
        return tmp_path
    raise TypeError(f"Tipo de entrada de imagen no soportado: {type(image)}")


def extract_features_from_residual_array(res_arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(res_arr, dtype=np.float32).flatten()

    mean = np.mean(arr)
    variance = np.var(arr)
    sk = skew(arr)
    kt = kurtosis(arr)

    hist, _ = np.histogram(arr, bins=64, range=(0, 255), density=True)

    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    iqr = q75 - q25

    return np.hstack(([mean, variance, sk, kt, q25, q75, iqr], hist)).astype(np.float64)


def build_feature_vector(image_path: str) -> np.ndarray:
    res_arr = compute_spatial_residual(image_path)
    features = extract_features_from_residual_array(res_arr)

    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != 71:
        raise ValueError(
            f"Se esperaban 71 características, pero se obtuvieron {arr.shape[1]}."
        )

    return arr


def predict_stego(image):
    if image is None:
        return "Sube una imagen para analizar", {}

    temp_created = None
    try:
        if isinstance(image, np.ndarray):
            temp_created = _resolve_image_path(image)
            img_path = temp_created
        else:
            img_path = _resolve_image_path(image)

        features = build_feature_vector(img_path)
        prediction = int(model.predict(features)[0])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            cover_prob = float(probs[0])
            stego_prob = float(probs[1])
        else:
            stego_prob = 1.0 if prediction == 1 else 0.0
            cover_prob = 1.0 - stego_prob

        diagnosis = (
            "STEGO — Posible alteración estenográfica detectada"
            if prediction == 1
            else "COVER — Imagen sin indicios de alteración LSB"
        )

        confidence_map = {
            "COVER": cover_prob,
            "STEGO": stego_prob,
        }
        return diagnosis, confidence_map

    except Exception as exc:
        return f"Error durante el procesamiento: {exc}", {}

    finally:
        if temp_created and os.path.exists(temp_created):
            try:
                os.remove(temp_created)
            except OSError:
                pass


demo = gr.Interface(
    fn=predict_stego,
    inputs=gr.Image(
        type="filepath",
        label="Imagen de entrada (preferible escala de grises / sin compresión agresiva)",
    ),
    outputs=[
        gr.Textbox(label="Diagnóstico del Clasificador"),
        gr.Label(num_top_classes=2, label="Distribución de probabilidad"),
    ],
    title="Detección de Esteganografía LSB con SVM (Núcleo RBF)",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
