import os
import wandb
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.models import *

# ─────────────────────────────────────────────
# RUTAS DEL DATASET
# ─────────────────────────────────────────────
TRAIN_DIR = '/home/edxnG02/split/train'
TEST_DIR  = '/home/edxnG02/split/test'

# ─────────────────────────────────────────────
# TRANSFORMACIONES
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────────
def get_data(train=True):
    if train:
        dataset = ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    else:
        dataset = ImageFolder(root=TEST_DIR, transform=test_transforms)
    return dataset

# ─────────────────────────────────────────────
# CREAR DATALOADER
# ─────────────────────────────────────────────
def make_loader(dataset, batch_size):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=2)
    return loader

# ─────────────────────────────────────────────
# MONTAR TODO
# ─────────────────────────────────────────────
def make(config, device="cuda"):
    train_data = get_data(train=True)
    test_data  = get_data(train=False)

    train_loader = make_loader(train_data, batch_size=config.batch_size)
    test_loader  = make_loader(test_data,  batch_size=config.batch_size)

    num_classes = len(train_data.classes)
    print(f"Clases detectadas: {num_classes} → {train_data.classes}")

    model = ConvNet(config.kernels, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer