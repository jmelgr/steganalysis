import os
import glob
import numpy as np
from PIL import Image

def embed_lsb(image_path, output_path, payload_rate=0.4, seed=42):
    with Image.open(image_path) as img:
        img_arr = np.array(img, dtype=np.uint8)
        h, w = img_arr.shape
        total_pixels = h * w
        
        payload_bits_len = int(total_pixels * payload_rate)
        
        np.random.seed(seed + hash(image_path) % 10000)
        payload = np.random.randint(0, 2, size=payload_bits_len, dtype=np.uint8)
        
        flat_img = img_arr.flatten()
        
        flat_img[:payload_bits_len] = (flat_img[:payload_bits_len] & 254) | payload
        
        stego_arr = flat_img.reshape((h, w))
        stego_img = Image.fromarray(stego_arr)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        stego_img.save(output_path)

def generate_stego_dataset(processed_dir="data/processed", payload_rate=0.4):
    for split in ["train", "test"]:
        cover_dir = os.path.join(processed_dir, split, "cover")
        stego_dir = os.path.join(processed_dir, split, "stego")
        
        cover_files = glob.glob(os.path.join(cover_dir, "*.pgm"))
        if not cover_files:
            print(f"No se encontraron imágenes en {cover_dir}. Ejecute primero la partición.")
            continue
            
        print(f"Procesando conjunto de {split}: generando {len(cover_files)} imágenes stego a {payload_rate} bpp...")
        
        for i, f in enumerate(cover_files):
            filename = os.path.basename(f)
            output_path = os.path.join(stego_dir, filename)
            embed_lsb(f, output_path, payload_rate=payload_rate)
            
    print("Construcción del conjunto de datos stego finalizada exitosamente.")

if __name__ == "__main__":
    generate_stego_dataset()
