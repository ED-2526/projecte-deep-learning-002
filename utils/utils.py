import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import WikiArtCNN

# ─────────────────────────────────────────────
# RUTAS DEL SPLIT  (ajusta si es necesario)
# ─────────────────────────────────────────────
TRAIN_DIR = '/home/edxnG02/split_style/train'
VAL_DIR   = '/home/edxnG02/split_style/val'
TEST_DIR  = '/home/edxnG02/split_style/test'

# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL: make()
# ─────────────────────────────────────────────
def make(config, device="cuda"):
    model                       = make_model(config, device)
    train_ldr, val_ldr, test_ldr = make_loaders(config)
    criterion                   = nn.CrossEntropyLoss()
    optimizer                   = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return model, train_ldr, val_ldr, test_ldr, criterion, optimizer


# ─────────────────────────────────────────────
# DATA LOADERS  —  ImageFolder lee las subcarpetas
# como clases automáticamente:
#   split_style/train/Abstract_Expressionism/ → clase 0
#   split_style/train/Action_painting/        → clase 1
#   ...
# ─────────────────────────────────────────────
def make_loaders(config):
    train_transform = transforms.Compose([
        transforms.Resize((148, 148)),          # una mica més gran per poder fer crop
        transforms.RandomCrop((128, 128)),       # crop aleatori
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=20),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.4, hue=0.1
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # tapa zones aleatòries
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_ds   = ImageFolder(root=VAL_DIR,   transform=eval_transform)
    test_ds  = ImageFolder(root=TEST_DIR,  transform=eval_transform)

    assert len(train_ds.classes) == config.classes, \
        f"Clases encontradas: {len(train_ds.classes)}, esperadas: {config.classes}"

    print(f"Clases ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

    train_ldr = DataLoader(train_ds, batch_size=config.batch_size,
                           shuffle=True,  num_workers=4, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=config.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_ldr  = DataLoader(test_ds,  batch_size=config.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    return train_ldr, val_ldr, test_ldr


# ─────────────────────────────────────────────
# MODELO CNN
# ─────────────────────────────────────────────
def make_model(config, device="cuda"):
    model = WikiArtCNN(n_classes=config.classes, kernels=config.kernels).to(device)
    return model