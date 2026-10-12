import os
import glob
import numpy as np
from PIL import Image
from scipy.stats import skew, kurtosis

def extract_features_from_image(image_path):
    with Image.open(image_path) as img:
        arr = np.array(img, dtype=np.float32).flatten()
        
        mean = np.mean(arr)
        variance = np.var(arr)
        sk = skew(arr)
        kt = kurtosis(arr)
        
        hist, _ = np.histogram(arr, bins=16, range=(0, 255), density=True)
        
        features = np.hstack(([mean, variance, sk, kt], hist))
        return features

def build_dataset(split="train", processed_dir="data/processed"):
    features_list = []
    labels_list = []
    
    categories = {"cover": 0, "stego": 1}
    
    for cat, label in categories.items():
        dir_path = os.path.join(processed_dir, split, f"{cat}_residual")
        files = glob.glob(os.path.join(dir_path, "*.png"))
        
        print(f"Extrayendo características de {split}/{cat} ({len(files)} imágenes)...")
        for f in files:
            feats = extract_features_from_image(f)
            features_list.append(feats)
            labels_list.append(label)
            
    return np.array(features_list), np.array(labels_list)

def main():
    os.makedirs("data/features", exist_ok=True)
    
    X_train, y_train = build_dataset("train")
    np.save("data/features/X_train.npy", X_train)
    np.save("data/features/y_train.npy", y_train)
    
    X_test, y_test = build_dataset("test")
    np.save("data/features/X_test.npy", X_test)
    np.save("data/features/y_test.npy", y_test)
    
    print("¡Extracción e ingesta de características completada con éxito!")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

if __name__ == "__main__":
    main()
