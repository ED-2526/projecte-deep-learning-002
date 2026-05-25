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

torch.cuda.manual_seed_all(
    hash("so runs are repeatable") % 2**32 - 1
)

# ─────────────────────────────────────────────
# DISPOSITIVO
# ─────────────────────────────────────────────

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print(f"Usando dispositivo: {device}")

# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def model_pipeline(cfg: dict):

    with wandb.init(
        project="wikiart-style-classification",
        config=cfg
    ):

        config = wandb.config

        model, train_loader, val_loader, test_loader, \
        criterion, optimizer = make(
            config,
            device=device
        )

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            config,
            device=device
        )

        # Cargar mejor modelo guardado
        model.load_state_dict(
            torch.load("best_model.pth")
        )

        # Test final
        test(model,test_loader,epoch=config.epochs,device=device)

    return model

# ─────────────────────────────────────────────
# CONFIGURACIÓN DEL EXPERIMENTO
# ─────────────────────────────────────────────

if __name__ == "__main__":

    wandb.login()

    config = dict(

        # Entrenamiento
        epochs=30,

        # Dataset
        classes=13,
        dataset="WikiArt",

        # Modelo
        architecture="ResNet50",

        # Hyperparameters
        batch_size=8,
        learning_rate=1e-4,
        dropout_rate=0.3,

        # Compatibilidad
        kernels=[],

        # Task
        task="style_classification"
    )

    model = model_pipeline(config)

