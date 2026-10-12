import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def analyze_spatial_correlation(data_dir, output_path="reports/figures/spatial_correlation.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pgm_files = glob.glob(os.path.join(data_dir, "**/*.pgm"), recursive=True)
    if not pgm_files:
        print("No se encontraron archivos .pgm en la ruta especificada.")
        return

    sample_path = pgm_files[0]
    with Image.open(sample_path) as img:
        img_arr = np.array(img, dtype=float)
        
        h, w = img_arr.shape
        x1 = img_arr[:, :-1].flatten()
        x2 = img_arr[:, 1:].flatten()
        
        indices = np.random.choice(len(x1), size=5000, replace=False)
        x1_sample = x1[indices]
        x2_sample = x2[indices]
        
        corr_coef = np.corrcoef(x1_sample, x2_sample)[0, 1]
        print(f"Coeficiente de correlación horizontal medio: {corr_coef:.4f}")
        
        plt.figure(figsize=(6, 6))
        plt.scatter(x1_sample, x2_sample, s=5, color='black', alpha=0.3)
        plt.title(f"Correlación de Píxeles Vecinos (r = {corr_coef:.3f})")
        plt.xlabel("Valor del Píxel I(x, y)")
        plt.ylabel("Valor del Píxel Adyacente I(x+1, y)")
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Gráfica de correlación guardada en: {output_path}")

if __name__ == "__main__":
    analyze_spatial_correlation("data/raw/BOSSbase_1.01")
