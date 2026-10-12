import os
import glob
import numpy as np
from PIL import Image

def compute_spatial_residual(image_path):
    with Image.open(image_path) as img:
        img_arr = np.array(img, dtype=np.float32)
        residual = img_arr[:, 1:] - img_arr[:, :-1]
        
        res_img_arr = np.clip(residual + 128, 0, 255).astype(np.uint8)
        return res_img_arr

def process_residuals(processed_dir="data/processed"):
    for split in ["train", "test"]:
        for cat in ["cover", "stego"]:
            input_dir = os.path.join(processed_dir, split, cat)
            output_dir = os.path.join(processed_dir, split, f"{cat}_residual")
            os.makedirs(output_dir, exist_ok=True)
            
            files = glob.glob(os.path.join(input_dir, "*.pgm"))
            if not files:
                print(f"No se encontraron imágenes en {input_dir}.")
                continue
                
            print(f"Calculando residuales para {split}/{cat} ({len(files)} imágenes)...")
            for f in files:
                res_arr = compute_spatial_residual(f)
                filename = os.path.basename(f).replace(".pgm", ".png")
                output_path = os.path.join(output_dir, filename)
                
                Image.fromarray(res_arr).save(output_path)
                
    print("Transformación de residuales espaciales completada con éxito.")

if __name__ == "__main__":
    process_residuals()
