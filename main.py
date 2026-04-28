import os
import random
import wandb
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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
# DISPOSITIVO — GPU si disponible, sino CPU
# ─────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def model_pipeline(cfg: dict) -> None:
    with wandb.init(project="wikiart-autor-classification", config=cfg):
        config = wandb.config

        # Cargar datos, modelo, loss y optimizador
        model, train_loader, test_loader, criterion, optimizer = make(config, device=device)

        # Entrenar
        train(model, train_loader, criterion, optimizer, config, device=device)

        # Evaluar
        test(model, test_loader, device=device)

    return model

# ─────────────────────────────────────────────
# CONFIGURACIÓN DEL EXPERIMENTO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=10,
        classes=25,
        kernels=[16, 32, 64],
        batch_size=64,
        learning_rate=1e-3,
        dataset="WikiArt",
        architecture="CNN")

    model = model_pipeline(config)

