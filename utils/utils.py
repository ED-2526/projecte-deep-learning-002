
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import WikiArtResNet

TRAIN_DIR = '/home/edxnG02/split_style/train'
VAL_DIR   = '/home/edxnG02/split_style/val'
TEST_DIR  = '/home/edxnG02/split_style/test'


def make(config, device="cuda"):

    model = make_model(config, device)

    train_ldr, val_ldr, test_ldr = make_loaders(config)

    # Label smoothing → més robust
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    return model, train_ldr, val_ldr, test_ldr, criterion, optimizer


def make_loaders(config):

    # ─────────────────────────────────────────────
    # TRAIN TRANSFORMS
    # ─────────────────────────────────────────────

    train_transform = transforms.Compose([

        # Imatge més gran per ResNet50
        transforms.Resize((300, 300)),

        # Crop final a 256
        transforms.RandomCrop((256, 256)),

        transforms.RandomHorizontalFlip(p=0.5),

        # Rotació més suau
        transforms.RandomRotation(degrees=5),

        # Color jitter més conservador
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        # RandomErasing eliminat
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    # ─────────────────────────────────────────────
    # VALIDATION / TEST TRANSFORMS
    # ─────────────────────────────────────────────

    eval_transform = transforms.Compose([

        transforms.Resize((256, 256)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ─────────────────────────────────────────────
    # DATASETS
    # ─────────────────────────────────────────────

    train_ds = ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    val_ds = ImageFolder(
        root=VAL_DIR,
        transform=eval_transform
    )

    test_ds = ImageFolder(
        root=TEST_DIR,
        transform=eval_transform
    )

    # ─────────────────────────────────────────────
    # CHECK CLASSES
    # ─────────────────────────────────────────────

    assert len(train_ds.classes) == config.classes, \
        f"Clases encontradas: {len(train_ds.classes)}, esperadas: {config.classes}"

    print(f"Clases ({len(train_ds.classes)}): {train_ds.classes}")

    print(
        f"Train: {len(train_ds)}  |  "
        f"Val: {len(val_ds)}  |  "
        f"Test: {len(test_ds)}"
    )

    # ─────────────────────────────────────────────
    # DATALOADERS
    # ─────────────────────────────────────────────

    train_ldr = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_ldr = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_ldr = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_ldr, val_ldr, test_ldr


def make_model(config, device="cuda"):

    model = WikiArtResNet(
        n_classes=config.classes,
        dropout_rate=config.dropout_rate
    ).to(device)

    # Fase inicial → backbone congelat
    model.freeze_backbone()

    return model
