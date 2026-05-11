import torch
import wandb
from tqdm.auto import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, config, device="cuda"):
    wandb.watch(model, criterion, log="all", log_freq=10)

    best_val_acc  = 0.0
    best_val_loss = float('inf')
    example_ct    = 0

    # Cosine scheduler → reduce el lr suavemente en forma de coseno
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    for epoch in tqdm(range(config.epochs)):

        # ── Fase 2: en época 5 descongelamos todo el backbone ──
        if epoch == 5:
            model.unfreeze_backbone()
            # El backbone aprende muy lento (1e-5) para no destruir
            # lo que ya sabe de ImageNet
            # La cabeza aprende más rápido (1e-4)
            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in model.model.named_parameters()
                            if 'fc' not in n], 'lr': 1e-5},
                {'params': model.model.fc.parameters(), 'lr': 1e-4}
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs - 5
            )
            print("  → Backbone descongelat, fine-tuning complet activat")

        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            loss, correct = train_batch(
                images, labels, model, optimizer, criterion, device
            )
            train_loss    += loss.item() * images.size(0)
            train_correct += correct
            train_total   += images.size(0)
            example_ct    += images.size(0)

        avg_train_loss = train_loss / train_total
        train_acc      = 100 * train_correct / train_total

        # ── Validació ──
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # El scheduler avanza una época
        scheduler.step()

        # ── Log wandb ──
        wandb.log({
            "epoch"      : epoch + 1,
            "train_loss" : avg_train_loss,
            "train_acc"  : train_acc,
            "val_loss"   : avg_val_loss,
            "val_acc"    : val_acc,
            "lr"         : scheduler.get_last_lr()[0]
        }, step=example_ct)

        print(f"Epoch {epoch+1:>3}/{config.epochs} | "
              f"Train loss: {avg_train_loss:.4f}  acc: {train_acc:.2f}% | "
              f"Val loss: {avg_val_loss:.4f}  acc: {val_acc:.2f}%")

        # ── Guardar millor model ──
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
    # Gradient clipping → si el gradiente es muy grande lo corta a 1.0
    # evita que un batch malo rompa los pesos
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    correct = (outputs.argmax(dim=1) == labels).sum().item()
    return loss, correct


def evaluate(model, loader, criterion, device="cuda"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs    = model(images)
            loss       = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, 100 * correct / total