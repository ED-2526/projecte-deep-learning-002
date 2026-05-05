# importar librerias
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
input_base    = '/home/datasets/wikiart'
output_folder = '/home/edxnG02/processedImages'
classes_csv   = os.path.join(input_base, 'classes.csv')
target_size   = (128, 128)


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
# CARGAR CSV
# ─────────────────────────────────────────────
print("Cargando classes.csv...")
df_classes = pd.read_csv(classes_csv)
print(df_classes.head())
print(f"Columnas disponibles: {df_classes.columns.tolist()}")

# ─────────────────────────────────────────────
# VERIFICACIÓ DE NULS
# ─────────────────────────────────────────────
print("\nNULS PER COLUMNA:")
print(df_classes.isnull().sum())

print(f"\nPERCENTATGE DE NULS:")
print((df_classes.isnull().sum() / len(df_classes) * 100).round(2).astype(str) + " %")

filas_con_nulos = df_classes[df_classes.isnull().any(axis=1)]
print(f"\nFiles amb algun nul: {len(filas_con_nulos)} de {len(df_classes)}")

if len(filas_con_nulos) > 0:
    print(filas_con_nulos.head(10))
    df_classes = df_classes.dropna()
    print(f"\nFiles amb nuls eliminades. Files restants: {len(df_classes)}")
else:
    print("No hi ha nuls al CSV.")

# ─────────────────────────────────────────────
# VERIFICACIÓ DE DUPLICATS AL CSV
# ─────────────────────────────────────────────
print("\nCOMPROVACIÓ DE DUPLICATS AL CSV:")
duplicats = df_classes.duplicated(subset='filename')
print(f"Filenames duplicats: {duplicats.sum()}")

if duplicats.sum() > 0:
    print(df_classes[duplicats][['filename']].head(10))
    df_classes = df_classes.drop_duplicates(subset='filename')
    print(f"Duplicats eliminats. Files restants: {len(df_classes)}")
else:
    print("No hi ha duplicats al CSV.")

# ── Construir set de filenames vàlids ─────────
valid_filenames = set(df_classes['filename'].astype(str).str.strip().values)
print(f"\nTotal filenames vàlids: {len(valid_filenames)}\n")



# ─────────────────────────────────────────────
# PROCESSAR CADA CATEGORIA I DETECTAR DUPLICATS
# ─────────────────────────────────────────────
total_procesadas = 0
total_corruptas  = 0
total_no_en_csv  = 0
total_errores    = 0
total_duplicados = 0
resumen = []

hashes_vistos = {}  # hash MD5 → primer filepath que el va generar

for category in categories:
    input_path  = os.path.join(input_base, category)
    output_path = os.path.join(output_folder, category)

    if not os.path.isdir(input_path):
        print(f"Carpeta no trobada, s'omet: {input_path}")
        continue

    os.makedirs(output_path, exist_ok=True)

    cat_procesadas = 0
    cat_corruptas  = 0
    cat_no_en_csv  = 0
    cat_errores    = 0
    cat_duplicados = 0

    archivos = glob.glob(os.path.join(input_path, '*.jpg')) + \
               glob.glob(os.path.join(input_path, '*.JPG')) + \
               glob.glob(os.path.join(input_path, '*.jpeg'))

    print(f"Processant: {category}  ({len(archivos)} arxius trobats)")

    for filepath in archivos:
        basename      = os.path.basename(filepath)
        relative_path = f"{category}/{basename}"

        # ── VALIDACIÓ 1: Està al CSV? ─────────────────
        if basename not in valid_filenames and relative_path not in valid_filenames:
            cat_no_en_csv += 1
            continue

        # ── VALIDACIÓ 2: Duplicat per hash MD5? ───────
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in hashes_vistos:
            cat_duplicados += 1
            print(f"   Duplicat: {basename} == {hashes_vistos[file_hash]}")
            continue
        hashes_vistos[file_hash] = relative_path

        # ── VALIDACIÓ 3: Integritat de la imatge ──────
        try:
            img = cv2.imread(filepath)

            if img is None:
                cat_corruptas += 1
                continue

            h, w = img.shape[:2]
            if h == 0 or w == 0:
                cat_corruptas += 1
                continue

            # ── PROCESSAMENT ──────────────────────────
            img_resized = cv2.resize(img, target_size)
            output_file = os.path.join(output_path, basename)
            cv2.imwrite(output_file, img_resized)
            cat_procesadas += 1

        except Exception as e:
            print(f"   Error en {basename}: {e}")
            cat_errores += 1

    total_procesadas += cat_procesadas
    total_corruptas  += cat_corruptas
    total_no_en_csv  += cat_no_en_csv
    total_errores    += cat_errores
    total_duplicados += cat_duplicados

    resumen.append({
        'Categoria'  : category,
        'Processades': cat_procesadas,
        'Corruptes'  : cat_corruptas,
        'No al CSV'  : cat_no_en_csv,
        'Duplicades' : cat_duplicados,
        'Errors'     : cat_errores
    })

    print(f"   Processades: {cat_procesadas}  |    Corruptes: {cat_corruptas}"
          f"  |  No al CSV: {cat_no_en_csv}  |   Duplicades: {cat_duplicados}"
          f"  |  Errors: {cat_errores}\n")

# ─────────────────────────────────────────────
# RESUM FINAL
# ─────────────────────────────────────────────
print("=" * 70)
print("RESUM FINAL")
print("=" * 70)
df_resumen = pd.DataFrame(resumen)
print(df_resumen.to_string(index=False))
print("-" * 70)
print(f"TOTAL processades : {total_procesadas}")
print(f"TOTAL corruptes   : {total_corruptas}")
print(f"TOTAL no al CSV   : {total_no_en_csv}")
print(f"TOTAL duplicades  : {total_duplicados}")
print(f"TOTAL errors      : {total_errores}")
print("=" * 70)

# ─────────────────────────────────────────────
# HISTOGRAMA D'AUTORS
# ─────────────────────────────────────────────
mask = df_classes['filename'].astype(str).apply(
    lambda x: any(x.startswith(cat) for cat in categories)
)
df_filtrat = df_classes[mask].copy()
imatges_per_autor = df_filtrat['artist'].value_counts()

print(f"Total autors únics: {len(imatges_per_autor)}")

top_n       = 100
top_autors  = imatges_per_autor.head(top_n)

os.makedirs(output_folder, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 7))
ax.bar(range(len(top_autors)), top_autors.values, color='steelblue', edgecolor='white')
ax.set_xticks(range(len(top_autors)))
ax.set_xticklabels(top_autors.index, rotation=45, ha='right', fontsize=8)
ax.set_title(f'Top {top_n} autors amb més imatges', fontsize=14, fontweight='bold')
ax.set_xlabel('Autor')
ax.set_ylabel('Nombre d\'imatges')
ax.grid(axis='y', alpha=0.3)

for i, v in enumerate(top_autors.values):
    ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'histograma_autors.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"\n Histograma guardat a: {output_folder}/histograma_autors.png")
