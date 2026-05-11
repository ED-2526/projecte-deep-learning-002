import os
import random
import wandb
import numpy as np
import torch
import torch.nn as nn

from train import *
from test import *
from utils.utils import *
from tqdm.auto import tqdm

# ─────────────────────────────────────────────
# COMPORTAMIENTO DETERMINISTA
# ─────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# ─────────────────────────────────────────────
# DISPOSITIVO
# ─────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def model_pipeline(cfg: dict) -> None:
    with wandb.init(project="wikiart-style-classification", config=cfg):  # ← nombre proyecto cambiado
        config = wandb.config
        model, train_loader, val_loader, test_loader, criterion, optimizer = make(config, device=device)
        train(model, train_loader, val_loader, criterion, optimizer, config, device=device)
        # Carreguem el millor model guardat durant l'entrenament
        model.load_state_dict(torch.load("best_model.pth"))
        test(model, test_loader, device=device)
    return model

# ─────────────────────────────────────────────
# CONFIGURACIÓN DEL EXPERIMENTO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=30,
        classes=13,
        kernels=[],           # ya no se usa pero lo dejamos para no romper nada
        batch_size=32,        # reducido de 64 a 32 porque 224x224 ocupa más VRAM
        learning_rate=1e-3,   # lr inicial para la cabeza en fase 1
        dataset="WikiArt",
        architecture="ResNet18",
        task="style_classification"
    )

    model = model_pipeline(config)
