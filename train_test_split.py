import os
import shutil
import random

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
input_folder  = '/home/edxnG02/processedImages_balanced'
output_folder = '/home/edxnG02/split_style'
train_ratio   = 0.80
val_ratio     = 0.10
test_ratio    = 0.10   # el resto
random.seed(42)

# ─────────────────────────────────────────────
# CREAR SPLIT TRAIN / VAL / TEST
# ─────────────────────────────────────────────
print(f"Creando split {int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}...\n")

total_train = total_val = total_test = 0

for category in sorted(os.listdir(input_folder)):
    cat_path = os.path.join(input_folder, category)
    if not os.path.isdir(cat_path):
        continue

    imagenes = [f for f in os.listdir(cat_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imagenes)

    n        = len(imagenes)
    n_train  = int(n * train_ratio)
    n_val    = int(n * val_ratio)
    # el test agafa el que queda per evitar perdre imatges per arrodoniment
    n_test   = n - n_train - n_val

    train_imgs = imagenes[:n_train]
    val_imgs   = imagenes[n_train:n_train + n_val]
    test_imgs  = imagenes[n_train + n_val:]

    # Crear carpetes destí
    for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        split_path = os.path.join(output_folder, split, category)
        os.makedirs(split_path, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(cat_path, img), os.path.join(split_path, img))

    total_train += len(train_imgs)
    total_val   += len(val_imgs)
    total_test  += len(test_imgs)

    print(f"  {category:<35}  train: {len(train_imgs):>4}  |  val: {len(val_imgs):>4}  |  test: {len(test_imgs):>4}")

# ─────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"TOTAL train  : {total_train}")
print(f"TOTAL val    : {total_val}")
print(f"TOTAL test   : {total_test}")
print(f"TOTAL global : {total_train + total_val + total_test}")
print(f"Classes      : {len(os.listdir(os.path.join(output_folder, 'train')))}")
print(f"Guardat a    : {output_folder}")
print(f"{'='*60}")