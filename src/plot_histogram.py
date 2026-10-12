import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

def plot_sample_histogram(data_dir, output_path="reports/figures/histogram_sample.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pgm_files = glob.glob(os.path.join(data_dir, "**/*.pgm"), recursive=True)
    if not pgm_files:
        print("No se encontraron archivos .pgm en la ruta especificada.")
        return

    sample_path = pgm_files[0]
    with Image.open(sample_path) as img:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Imagen Cover Original")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.hist(list(img.getdata()), bins=256, range=(0, 256), color='black', alpha=0.7)
        plt.title("Histograma de Intensidad (0-255)")
        plt.xlabel("Nivel de Gris")
        plt.ylabel("Frecuencia de Píxeles")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Figura generada exitosamente en: {output_path}")

if __name__ == "__main__":
    plot_sample_histogram("data/raw/BOSSbase_1.01")
