import torch
import wandb
from tqdm.auto import tqdm


def train(model, train_loader, val_loader, criterion, optimizer, config, device="cuda"):
    wandb.watch(model, criterion, log="all", log_freq=10)

    best_val_acc  = 0.0
    best_val_loss = float('inf')
    example_ct    = 0

    for epoch in tqdm(range(config.epochs)):

        # ─────────────────────────────────────────────
        # TRAIN
        # ─────────────────────────────────────────────
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for images, labels in train_loader:
            loss, correct = train_batch(images, labels, model, optimizer, criterion, device)

            train_loss    += loss.item() * images.size(0)
            train_correct += correct
            train_total   += images.size(0)
            example_ct    += images.size(0)

        avg_train_loss = train_loss / train_total
        train_acc      = 100 * train_correct / train_total

        # ─────────────────────────────────────────────
        # VALIDACIÓ
        # ─────────────────────────────────────────────
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # ─────────────────────────────────────────────
        # LOG A WANDB
        # ─────────────────────────────────────────────
        wandb.log({
            "epoch"      : epoch + 1,
            "train_loss" : avg_train_loss,
            "train_acc"  : train_acc,
            "val_loss"   : avg_val_loss,
            "val_acc"    : val_acc,
        }, step=example_ct)

        print(f"Epoch {epoch+1:>3}/{config.epochs} | "
              f"Train loss: {avg_train_loss:.4f}  acc: {train_acc:.2f}% | "
              f"Val loss: {avg_val_loss:.4f}  acc: {val_acc:.2f}%")

        # ─────────────────────────────────────────────
        # GUARDAR MILLOR MODEL
        # ─────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Nou millor model guardat (val_acc: {best_val_acc:.2f}%)")
            wandb.log({"best_val_acc": best_val_acc})

    print(f"\nEntrenament acabat. Millor val_acc: {best_val_acc:.2f}%")


def train_batch(images, labels, model, optimizer, criterion, device="cuda"):
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss    = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    correct = (outputs.argmax(dim=1) == labels).sum().item()
    return loss, correct


def evaluate(model, loader, criterion, device="cuda"):
    """Loop de validació — sense gradient, sense augmentation."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs  = model(images)
            loss     = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)

    avg_loss = total_loss / total
    acc      = 100 * correct / total
    return avg_loss, acc