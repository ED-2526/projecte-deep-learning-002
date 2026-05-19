import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


def test(model, test_loader, device="cuda", save: bool = True):
    """
    Evalúa el modelo sobre test:
    - Accuracy global
    - Matriz de confusión
    - Accuracy por clase (para estudiar desbalanceo)
    - Classification report completo
    - ROC Curves + AUC
    - Logging en wandb
    """

    model.eval()

    # ─────────────────────────────────────────────
    # ACUMULADORES GLOBALES
    # ─────────────────────────────────────────────

    correct, total = 0, 0

    all_preds = []
    all_labels = []

    # NUEVO → probabilidades para ROC
    all_probs = []

    # Nombres reales de clases desde ImageFolder
    class_names = test_loader.dataset.classes

    # ─────────────────────────────────────────────
    # TEST LOOP
    # ─────────────────────────────────────────────

    with torch.no_grad():

        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)

            # NUEVO → probabilidades softmax
            probs = torch.softmax(outputs, dim=1)

            # Predicción final = índice con mayor score
            _, predicted = torch.max(outputs.data, 1)

            # Accuracy global
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            # Guardar para análisis posterior
            all_preds.extend(predicted.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

            # NUEVO → guardar probabilidades
            all_probs.extend(probs.cpu().numpy())

    # ─────────────────────────────────────────────
    # ACCURACY GLOBAL
    # ─────────────────────────────────────────────

    test_accuracy = correct / total

    print("=" * 70)

    print(
        f"Accuracy global del modelo sobre "
        f"{total} imágenes de test: "
        f"{test_accuracy:.4%}"
    )

    print("=" * 70)

    wandb.log({"test_accuracy": test_accuracy})

    # ─────────────────────────────────────────────
    # MATRIZ DE CONFUSIÓN
    # ─────────────────────────────────────────────

    cm = confusion_matrix(all_labels, all_preds)

    print("\nMATRIZ DE CONFUSIÓN:")

    print(cm)

    # Guardar imagen de la matriz
    plt.figure(figsize=(32, 18))

    plt.imshow(
        cm,
        interpolation='nearest',
        cmap=plt.cm.Blues
    )

    plt.title("Matriz de Confusión")

    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(
        tick_marks,
        class_names,
        rotation=90,
        fontsize=12
    )

    plt.yticks(
        tick_marks,
        class_names,
        fontsize=12
    )

    plt.xlabel("Predicción")

    plt.ylabel("Etiqueta Real")

    plt.tight_layout(pad=3.0)

    plt.savefig(
        "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    wandb.log({
        "confusion_matrix": wandb.Image("confusion_matrix.png")
    })

    # ─────────────────────────────────────────────
    # CLASSIFICATION REPORT (PRECISION / RECALL / F1)
    # ─────────────────────────────────────────────

    print("\nCLASSIFICATION REPORT:")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )

    print(report)

    # ─────────────────────────────────────────────
    # ACCURACY POR AUTOR (CLAVE PARA DESBALANCEO)
    # ─────────────────────────────────────────────

    print("\nACCURACY POR AUTOR:")

    class_accuracies = {}

    for i, class_name in enumerate(class_names):

        # Total real de esta clase
        class_total = np.sum(np.array(all_labels) == i)

        # Correctas de esta clase
        class_correct = np.sum(
            (np.array(all_labels) == i) &
            (np.array(all_preds) == i)
        )

        class_acc = (
            class_correct / class_total
            if class_total > 0 else 0
        )

        class_accuracies[class_name] = class_acc

        print(
            f"{class_name:<25} | "
            f"Accuracy: {class_acc:.4%} | "
            f"Samples: {class_total}"
        )

        # Log individual a wandb
        wandb.log({
            f"accuracy_{class_name}": class_acc
        })

    # ─────────────────────────────────────────────
    # NUEVO → ROC CURVES + AUC
    # ─────────────────────────────────────────────

    print("\nGenerando ROC Curves...")

    y_true = label_binarize(
        all_labels,
        classes=range(len(class_names))
    )

    y_score = np.array(all_probs)

    plt.figure(figsize=(14, 12))

    for i, class_name in enumerate(class_names):

        fpr, tpr, _ = roc_curve(
            y_true[:, i],
            y_score[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC = {roc_auc:.2f})"
        )

    # Línea diagonal aleatoria
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curves")

    plt.legend(
        loc="lower right",
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        "roc_curves.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    wandb.log({
        "roc_curves": wandb.Image("roc_curves.png")
    })

    print("ROC curves guardadas correctamente.")

    # ─────────────────────────────────────────────
    # DETECTAR AUTORES MÁS CONFUNDIDOS
    # ─────────────────────────────────────────────

    print("\nAUTORES MÁS CONFUNDIDOS:")

    cm_no_diag = cm.copy()

    np.fill_diagonal(cm_no_diag, 0)

    # Top 10 confusiones más grandes
    flat_indices = np.argsort(
        cm_no_diag,
        axis=None
    )[::-1]

    shown = 0

    for idx in flat_indices:

        real_class, pred_class = np.unravel_index(
            idx,
            cm_no_diag.shape
        )

        if cm_no_diag[real_class, pred_class] == 0:
            break

        print(
            f"Real: {class_names[real_class]:<20} → "
            f"Predicho: {class_names[pred_class]:<20} | "
            f"Errores: {cm_no_diag[real_class, pred_class]}"
        )

        shown += 1

        if shown >= 10:
            break

    # ─────────────────────────────────────────────
    # EXPORTAR MODELO
    # ─────────────────────────────────────────────

    if save:

        print("\nGuardant model com a TorchScript...")

        model.eval()

        # CANVIAT → ara uses 256x256
        dummy_input = torch.randn(1, 3, 256, 256).to(device)

        scripted = torch.jit.trace(model, dummy_input)

        scripted.save("model_scripted.pt")

        wandb.save("model_scripted.pt")

        wandb.save("confusion_matrix.png")

        # NUEVO
        wandb.save("roc_curves.png")

        print("Model guardat correctament.")

"""
    if save:
        print("\nExportando modelo a ONNX...")

        dummy_input = torch.randn(1, 3, 118, 118).to(device)

        torch.onnx.export(
            model,
            dummy_input,
            "model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        wandb.save("model.onnx")
        wandb.save("confusion_matrix.png")

    print("\nEvaluación completada correctamente.")
"""
