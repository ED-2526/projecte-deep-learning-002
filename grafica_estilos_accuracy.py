import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# DATOS: autor, num_estilos, accuracy
# ─────────────────────────────────────────────
datos = [
    ("raphael-kirchner",      1,  86.5),
    ("albrecht-durer",        1,  63.3),
    ("ivan-aivazovsky",       1,  74.1),
    ("ivan-shishkin",         1,  57.7),
    ("gustave-dore",          1,  53.3),
    ("rembrandt",             1,  54.2),
    ("camille-corot",         2,  50.0),
    ("edgar-degas",           2,  54.2),
    ("eugene-boudin",         2,  55.9),
    ("childe-hassam",         2,  35.5),
    ("claude-monet",          2,  35.8),
    ("john-singer-sargent",   2,  29.2),
    ("pierre-auguste-renoir", 2,  31.7),
    ("ilya-repin",            2,  24.1),
    ("isaac-levitan",         2,  16.7),
    ("vincent-van-gogh",      2,  15.8),
    ("alfred-sisley",         3,  20.4),
    ("camille-pissarro",      3,  30.8),
    ("nicholas-roerich",      3,  43.3),
    ("marc-chagall",          3,  23.3),
    ("james-tissot",          3,   8.6),
    ("boris-kustodiev",       4,   4.2),
    ("odilon-redon",          4,  25.6),
    ("paul-cezanne",          4,   9.5),
    ("martiros-saryan",       7,  25.0),
    ("pyotr-konchalovsky",    7,  50.0),
    ("henri-matisse",         9,  10.1),
    ("salvador-dali",         9,   4.1),
    ("pablo-picasso",        11,  30.0),
]

autores  = [d[0] for d in datos]
estilos  = [d[1] for d in datos]
accuracy = [d[2] for d in datos]

# ─────────────────────────────────────────────
# COLORES SEGÚN NÚMERO DE ESTILOS
# ─────────────────────────────────────────────
def get_color(n):
    if n == 1:   return '#639922'
    elif n <= 3: return '#EF9F27'
    else:        return '#E24B4A'

colors = [get_color(n) for n in estilos]

# ─────────────────────────────────────────────
# GRÁFICA 1 — scatter: estilos vs accuracy
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Relación entre número de estilos por autor y accuracy', 
             fontsize=14, fontweight='bold', y=1.02)

ax1 = axes[0]
scatter = ax1.scatter(estilos, accuracy, c=colors, s=120, zorder=3, edgecolors='white', linewidth=0.5)

for i, autor in enumerate(autores):
    ax1.annotate(autor, (estilos[i], accuracy[i]),
                 textcoords="offset points", xytext=(6, 0),
                 fontsize=7, color='#444441')

ax1.set_xlabel('Número de estilos artísticos del autor', fontsize=11)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('Scatter: estilos vs accuracy por autor', fontsize=11)
ax1.grid(axis='both', alpha=0.3)
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 100)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#639922', label='1 estilo'),
    Patch(facecolor='#EF9F27', label='2-3 estilos'),
    Patch(facecolor='#E24B4A', label='4+ estilos'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

z = np.polyfit(estilos, accuracy, 1)
p = np.poly1d(z)
x_line = np.linspace(1, 11, 100)
ax1.plot(x_line, p(x_line), '--', color='#888780', linewidth=1.5, label='tendencia')

# ─────────────────────────────────────────────
# GRÁFICA 2 — barras: accuracy ordenada por num estilos
# ─────────────────────────────────────────────
ax2 = axes[1]
datos_ordenados = sorted(datos, key=lambda x: (x[1], -x[2]))
autores_ord  = [d[0].replace('-', '\n') for d in datos_ordenados]
estilos_ord  = [d[1] for d in datos_ordenados]
accuracy_ord = [d[2] for d in datos_ordenados]
colors_ord   = [get_color(n) for n in estilos_ord]

bars = ax2.barh(range(len(autores_ord)), accuracy_ord, color=colors_ord, edgecolor='white', linewidth=0.3)

for i, (acc, n) in enumerate(zip(accuracy_ord, estilos_ord)):
    ax2.text(acc + 0.5, i, f'{acc:.1f}%', va='center', fontsize=7, color='#444441')
    ax2.text(-1, i, str(n), va='center', ha='right', fontsize=7, color='#888780')

ax2.set_yticks(range(len(autores_ord)))
ax2.set_yticklabels(autores_ord, fontsize=7)
ax2.set_xlabel('Accuracy (%)', fontsize=11)
ax2.set_title('Accuracy por autor (ordenado por nº estilos)', fontsize=11)
ax2.set_xlim(-5, 105)
ax2.grid(axis='x', alpha=0.3)
ax2.text(-4.5, len(autores_ord) - 0.5, 'n\nestilos', fontsize=7, color='#888780', ha='right')

plt.tight_layout()
plt.savefig('/home/edxnG02/Mireia_Ros/projecte-deep-learning-002/grafica_estilos_accuracy.png',
            dpi=150, bbox_inches='tight')
print("Gráfica guardada en el repo.")
