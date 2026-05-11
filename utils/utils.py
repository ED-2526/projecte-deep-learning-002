import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import WikiArtResNet  # ← cambiado de WikiArtCNN a WikiArtResNet

TRAIN_DIR = '/home/edxnG02/split_style/train'
VAL_DIR   = '/home/edxnG02/split_style/val'
TEST_DIR  = '/home/edxnG02/split_style/test'

def make(config, device="cuda"):
    model                        = make_model(config, device)
    train_ldr, val_ldr, test_ldr = make_loaders(config)
    
    # label_smoothing=0.1 → en lugar de aprender "100% Impresionismo"
    # aprende "90% Impresionismo, 10% resto" — más robusto
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW con weight_decay → penaliza pesos grandes, reduce overfitting
    # Solo optimiza parámetros que NO están congelados (la cabeza)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-4
    )
    return model, train_ldr, val_ldr, test_ldr, criterion, optimizer


def make_loaders(config):
    # 224x224 porque ResNet-18 fue diseñada para este tamaño
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),       # un poco más grande para poder recortar
        transforms.RandomCrop((224, 224)),   # recorte aleatorio → más variedad
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),       # sin augmentation en val/test
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


def make_model(config, device="cuda"):
    model = WikiArtResNet(n_classes=config.classes).to(device)
    model.freeze_backbone()  # Fase 1: solo entrenamos la cabeza
    return model