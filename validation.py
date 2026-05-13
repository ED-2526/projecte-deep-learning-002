
import itertools
from main import model_pipeline

# ─────────────────────────────────────────────
# GRID D'HIPERPARÀMETRES
# ─────────────────────────────────────────────

learning_rates = [5e-4]
batch_sizes    = [8, 16]
dropouts       = [0.3, 0.4]

# ─────────────────────────────────────────────
# GENERAR COMBINACIONS
# ─────────────────────────────────────────────

combinations = list(itertools.product(
    learning_rates,
    batch_sizes,
    dropouts
))

print(f"Total combinacions: {len(combinations)}")

# ─────────────────────────────────────────────
# EXECUTAR EXPERIMENTS
# ─────────────────────────────────────────────

for lr, batch_size, dropout in combinations:

    print("=" * 60)
    print(f"LR={lr} | BATCH={batch_size} | DROPOUT={dropout}")
    print("=" * 60)

    config = dict(
        epochs=30,
        classes=13,
        batch_size=batch_size,
        learning_rate=lr,
        dropout_rate=dropout,
        dataset="WikiArt",
        architecture="ResNet18",
        task="style_classification"
    )

    model_pipeline(config)