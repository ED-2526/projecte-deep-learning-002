# models.py (modificat)
import torch
import torch.nn as nn
import torchvision.models as models

class WikiArtResNet(nn.Module):
    def __init__(self, n_classes=13, dropout_rate=0.4):  # ← Afegim dropout_rate
        super().__init__()
        backbone = models.resnet50(weights='IMAGENET1K_V1')
        
        # Usem el dropout_rate passat com a paràmetre
        backbone.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True