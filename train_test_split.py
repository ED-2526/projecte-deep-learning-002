import os
import shutil
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
input_folder  = '/home/edxnG02/processedImages_total'
output_folder = '/home/edxnG02/split'
min_images    = 400   
max_images    = 600   
train_ratio   = 0.8
random.seed(42)

categories = [
    'Abstract_Expressionism',
    'Action_painting',
    'Analytical_Cubism',
    'Art_Nouveau_Modern',
    'Baroque',
    'Color_Field_Painting',
    'Contemporary_Realism',
    'Cubism',
    'Early_Renaissance',
    'Expressionism',
    'Fauvism',
    'High_Renaissance',
    'Impressionism',
    'Mannerism_Late_Renaissance',
    'Minimalism',
    'Naive_Art_Primitivism',
    'New_Realism',
    'Northern_Renaissance',
    'Pointillism',
    'Pop_Art',
    'Post_Impressionism',
    'Realism',
    'Rococo',
    'Romanticism',
    'Symbolism',
    'Synthetic_Cubism',
    'Ukiyo_e'
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
# FILTRAR: mínimo 400, capear a 600
# ─────────────────────────────────────────────
print(f"\nFiltrando autores con mínimo {min_images} imágenes (máximo {max_images})...")

autores_validos = {}
for autor, imgs in autor_imagenes.items():
    if len(imgs) >= min_images:
        random.shuffle(imgs)
        autores_validos[autor] = imgs[:max_images]  # ← capping aquí, antes del split

print(f"Autores válidos: {len(autores_validos)}")
for autor, imgs in sorted(autores_validos.items(), key=lambda x: -len(x[1])):
    print(f"  {autor}: {len(imgs)} imágenes (usadas para train/test)")

# ─────────────────────────────────────────────
# CREAR SPLIT TRAIN / TEST
# ─────────────────────────────────────────────
print(f"\nCreando split {int(train_ratio*100)}/{int((1-train_ratio)*100)}...")

total_train = 0
total_test  = 0

for autor, imagenes in autores_validos.items():
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

# ─────────────────────────────────────────────
# HISTOGRAMA AUTORES FINALES
# ─────────────────────────────────────────────
print("\nGenerando histograma de autores finales...")

autors_ordenats = sorted(autores_validos.items(), key=lambda x: -len(x[1]))
noms   = [a[0] for a in autors_ordenats]
counts = [len(a[1]) for a in autors_ordenats]

fig, ax = plt.subplots(figsize=(16, 7))
bars = ax.bar(range(len(noms)), counts, color='steelblue', edgecolor='white')
ax.set_xticks(range(len(noms)))
ax.set_xticklabels(noms, rotation=45, ha='right', fontsize=9)
ax.set_title(f'Autors seleccionats per al model (min {min_images} – max {max_images} imatges)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Autor')
ax.set_ylabel("Nombre d'imatges usades")
ax.set_ylim(0, max_images + 50)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=min_images, color='red',    linestyle='--', alpha=0.6, label=f'Mínim ({min_images})')
ax.axhline(y=max_images, color='orange', linestyle='--', alpha=0.6, label=f'Màxim ({max_images})')
ax.legend()

for i, v in enumerate(counts):
    ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
hist_path = os.path.join(output_folder, 'histograma_autors_finals.png')
plt.savefig(hist_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Histograma guardat a: {hist_path}")