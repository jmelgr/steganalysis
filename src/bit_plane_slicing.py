import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_bit_planes(data_dir, output_path="reports/figures/bit_planes_sample.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pgm_files = glob.glob(os.path.join(data_dir, "**/*.pgm"), recursive=True)
    if not pgm_files:
        print("No se encontraron archivos .pgm en la ruta especificada.")
        return

    sample_path = pgm_files[0]
    with Image.open(sample_path) as img:
        img_arr = np.array(img)
        
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes = axes.flatten()
        
        for i in range(8):
            bit_plane = ((img_arr >> i) & 1) * 255
            
            axes[i].imshow(bit_plane, cmap='gray')
            if i == 0:
                axes[i].set_title(f"Bit {i} (LSB - Ruido de Sensor)")
            elif i == 7:
                axes[i].set_title(f"Bit {i} (MSB - Estructura Principal)")
            else:
                axes[i].set_title(f"Bit {i}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Figura de planos de bits guardada exitosamente en: {output_path}")

if __name__ == "__main__":
    plot_bit_planes("data/raw/BOSSbase_1.01")
