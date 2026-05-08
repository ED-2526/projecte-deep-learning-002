import os
import shutil
import random
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
input_folder  = '/home/edxnG02/processedImages_style'
output_folder = '/home/edxnG02/processedImages_balanced'

MIN_IMAGES  = 2000   # clases con menos → eliminadas
TARGET      = 3500   # objetivo oversampling
MAX_IMAGES  = 4000   # cap para clases muy grandes

random.seed(42)
os.makedirs(output_folder, exist_ok=True)

# ─────────────────────────────────────────────
# PROCESAR CADA ESTILO
# ─────────────────────────────────────────────
resumen = []

for category in sorted(os.listdir(input_folder)):
    cat_input = os.path.join(input_folder, category)
    if not os.path.isdir(cat_input):
        continue

    imagenes = [f for f in os.listdir(cat_input)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n = len(imagenes)

    # ── ELIMINAR clases con menos de MIN_IMAGES ──
    if n < MIN_IMAGES:
        print(f"  [SKIP]  {category:<35} ({n} imgs) — eliminada")
        resumen.append({'Estilo': category, 'Original': n, 'Final': 0, 'Acció': 'Eliminada'})
        continue

    cat_output = os.path.join(output_folder, category)
    os.makedirs(cat_output, exist_ok=True)

    # ── CAP: clases con más de MAX_IMAGES ────────
    if n > MAX_IMAGES:
        seleccionades = random.sample(imagenes, MAX_IMAGES)
        accio = f'Cap a {MAX_IMAGES}'

    # ── OVERSAMPLING: clases entre MIN y TARGET ──
    elif n < TARGET:
        seleccionades = imagenes.copy()
        extra_needed  = TARGET - n
        seleccionades += random.choices(imagenes, k=extra_needed)
        accio = f'Oversample {n}→{TARGET}'

    # ── OK: clases ja dins del rang ─────────────
    else:
        seleccionades = imagenes
        accio = 'OK'

    # Copiar (amb nom únic per les duplicades)
    for i, fname in enumerate(seleccionades):
        src  = os.path.join(cat_input, fname)
        # Afegim prefix numèric per evitar col·lisions en oversampling
        dst  = os.path.join(cat_output, f"{i:05d}_{fname}")
        shutil.copy(src, dst)

    final = len(seleccionades)
    print(f"  [OK]    {category:<35} ({n:>5} → {final:>5} imgs)  [{accio}]")
    resumen.append({'Estilo': category, 'Original': n, 'Final': final, 'Acció': accio})

# ─────────────────────────────────────────────
# RESUMEN
# ─────────────────────────────────────────────
classes_kept = [r for r in resumen if r['Final'] > 0]
print(f"\n{'='*60}")
print(f"Clases mantingudes : {len(classes_kept)}")
print(f"Clases eliminades  : {len(resumen) - len(classes_kept)}")
print(f"Imatges totals     : {sum(r['Final'] for r in resumen)}")
print(f"{'='*60}")

# ─────────────────────────────────────────────
# HISTOGRAMA — imatges reals (blau) + duplicades (taronja)
# ─────────────────────────────────────────────
kept = [r for r in resumen if r['Final'] > 0]
labels    = [r['Estilo']    for r in kept]
originals = [r['Original']  for r in kept]
finals    = [r['Final']     for r in kept]
# Imatges afegides per oversampling (0 si no s'ha fet oversampling)
added     = [max(0, f - o) for o, f in zip(originals, finals)]

x = list(range(len(labels)))
fig, ax = plt.subplots(figsize=(18, 7))

# Barra base: imatges reals (fins al valor original)
ax.bar(x, originals, color='steelblue', label='Imatges reals', zorder=3)

# Barra apilada: imatges duplicades per oversampling
ax.bar(x, added, bottom=originals, color='coral', label='Duplicades (oversampling)', zorder=3)

# Línia de referència TARGET i MAX
ax.axhline(y=TARGET,     color='green',  linestyle='--', linewidth=1.5, label=f'Target ({TARGET})')
ax.axhline(y=MAX_IMAGES, color='red',    linestyle='--', linewidth=1.5, label=f'Cap ({MAX_IMAGES})')

# Etiquetes amb valor final a dalt de cada barra
for i, (orig, fin) in enumerate(zip(originals, finals)):
    ax.text(i, fin + 30, str(fin), ha='center', va='bottom', fontsize=8, fontweight='bold')
    # Si hi ha oversampling, mostrem també el valor original
    if fin > orig:
        ax.text(i, orig - 80, str(orig), ha='center', va='top', fontsize=7,
                color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_title('Dataset balancejat — Imatges reals vs Duplicades per oversampling',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Nombre d\'imatges')
ax.set_ylim(0, MAX_IMAGES + 500)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'histograma_balanced.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"\nHistograma guardat a: {output_folder}/histograma_balanced.png")