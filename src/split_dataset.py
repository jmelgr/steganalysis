import os
import glob
import shutil
import numpy as np

def split_corpus(data_dir="data/raw/BOSSbase_1.01", output_dir="data/processed", test_size=0.2, seed=42):
    pgm_files = glob.glob(os.path.join(data_dir, "**/*.pgm"), recursive=True)
    if not pgm_files:
        print("No se encontraron archivos .pgm en la ruta especificada.")
        return

    np.random.seed(seed)
    indices = np.random.permutation(len(pgm_files))

    split_index = int(len(pgm_files) * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_files = [pgm_files[i] for i in train_indices]
    test_files = [pgm_files[i] for i in test_indices]

    train_dir = os.path.join(output_dir, "train/cover")
    test_dir = os.path.join(output_dir, "test/cover")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"Dividiendo corpus: {len(train_files)} para entrenamiento, {len(test_files)} para pruebas.")

    for f in train_files:
        shutil.copy(f, os.path.join(train_dir, os.path.basename(f)))
    for f in test_files:
        shutil.copy(f, os.path.join(test_dir, os.path.basename(f)))

    print("Partición de datos completada exitosamente.")

if __name__ == "__main__":
    split_corpus()
