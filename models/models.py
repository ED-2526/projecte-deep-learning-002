import torch
import torch.nn as nn


class WikiArtCNN(nn.Module):
    """
    CNN per classificació d'estils artístics WikiArt.
    Entrada : (B, 3, 128, 128)
    Sortida : (B, n_classes)
    """

    def __init__(self, n_classes=13, kernels=[32, 64, 128, 256]):
        super().__init__()

        # CAPA 1 — vores i canvis de color bàsics
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(kernels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))   # 128x128 → 64x64

        # CAPA 2 — textures i formes
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(kernels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))   # 64x64 → 32x32

        # CAPA 3 — patrons de pinzellada i estil
        self.layer3 = nn.Sequential(
            nn.Conv2d(kernels[1], kernels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(kernels[2]),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2))   # 32x32 → 16x16

        # CAPA 4 — combinació de patrons complexos d'estil
        self.layer4 = nn.Sequential(
            nn.Conv2d(kernels[2], kernels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(kernels[3]),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2))   # 16x16 → 8x8

        # CAP CLASSIFICADOR
        # Amb 128x128 d'entrada i 4 MaxPool(2) → 8x8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(kernels[3] * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# TEST RÀPID
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model  = WikiArtCNN(n_classes=13, kernels=[32, 64, 128, 256])
    dummy  = torch.randn(8, 3, 128, 128)
    output = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {output.shape}")   # → (8, 13)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paràmetres entrenables: {total_params:,}")