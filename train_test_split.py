import os
import shutil
import random
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
input_folder  = '/home/edxnG02/processedImages'
output_folder = '/home/edxnG02/split'
min_images    = 100
train_ratio   = 0.8
random.seed(42)

categories = [
    'Abstract_Expressionism',
    'Action_painting',
    'Analytical_Cubism',
    'Art_Nouveau_Modern',
    'Baroque'
]

# ─────────────────────────────────────────────
# AGRUPAR IMÁGENES POR AUTOR
# ─────────────────────────────────────────────
print("Agrupando imágenes por autor...")
autor_imagenes = defaultdict(list)

for category in categories:
    cat_path = os.path.join(input_folder, category)
    if not os.path.isdir(cat_path):
        print(f"Carpeta no encontrada: {cat_path}")
        continue
    for filename in os.listdir(cat_path):
        if filename.endswith(('.jpg', '.jpeg', '.JPG')):
            autor = filename.split('_')[0]
            autor_imagenes[autor].append(os.path.join(cat_path, filename))

# ─────────────────────────────────────────────
# FILTRAR AUTORES CON MÁS DE 100 IMÁGENES
# ─────────────────────────────────────────────
print(f"\nFiltrando autores con más de {min_images} imágenes...")
autores_validos = {autor: imgs for autor, imgs in autor_imagenes.items() 
                   if len(imgs) >= min_images}

print(f"Autores válidos: {len(autores_validos)}")
for autor, imgs in sorted(autores_validos.items(), key=lambda x: -len(x[1])):
    print(f"  {autor}: {len(imgs)} imágenes")

# ─────────────────────────────────────────────
# CREAR SPLIT TRAIN / TEST
# ─────────────────────────────────────────────
print(f"\nCreando split {int(train_ratio*100)}/{int((1-train_ratio)*100)}...")

total_train = 0
total_test  = 0

for autor, imagenes in autores_validos.items():
    random.shuffle(imagenes)
    n_train = int(len(imagenes) * train_ratio)
    
    train_imgs = imagenes[:n_train]
    test_imgs  = imagenes[n_train:]

    train_path = os.path.join(output_folder, 'train', autor)
    test_path  = os.path.join(output_folder, 'test', autor)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path,  exist_ok=True)

    for img in train_imgs:
        shutil.copy(img, os.path.join(train_path, os.path.basename(img)))
    for img in test_imgs:
        shutil.copy(img, os.path.join(test_path, os.path.basename(img)))

    total_train += len(train_imgs)
    total_test  += len(test_imgs)
    print(f"  {autor}: {len(train_imgs)} train | {len(test_imgs)} test")

# ─────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"TOTAL train : {total_train}")
print(f"TOTAL test  : {total_test}")
print(f"Autores     : {len(autores_validos)}")
print(f"Guardat a   : {output_folder}")
print(f"{'='*50}")
