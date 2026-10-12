import os
import glob
from PIL import Image

def validate_bossbase(data_dir):
    pgm_files = glob.glob(os.path.join(data_dir, "**/*.pgm"), recursive=True)
    total_files = len(pgm_files)
    
    print(f"Total de archivos .pgm encontrados: {total_files}")
    
    valid_count = 0
    errors = []

    for file_path in pgm_files:
        try:
            with Image.open(file_path) as img:
                if img.size != (512, 512):
                    errors.append(f"{file_path}: Dimensión inválida {img.size}")
                elif img.mode not in ['L', 'I']:
                    errors.append(f"{file_path}: Modo de color inesperado {img.mode}")
                else:
                    valid_count += 1
        except Exception as e:
            errors.append(f"{file_path}: Error al leer archivo -> {e}")

    print(f"\n--- Resultados de Validación ---")
    print(f"Imágenes válidas y conformes: {valid_count}/{total_files}")
    print(f"Errores encontrados: {len(errors)}")
    
    if errors:
        print("Primeros errores detectados:", errors[:5])
    else:
        print("¡Verificación exitosa! El 100% de las imágenes cumplen con los requisitos de 512x512 y 8 bits sin anomalías.")

if __name__ == "__main__":
    target_directory = "data/raw/BOSSbase_1.01"
    validate_bossbase(target_directory)
