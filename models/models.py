import torch
import torch.nn as nn
import torchvision.models as models

class WikiArtResNet(nn.Module):
    def __init__(self, n_classes=13):
        super().__init__()
        # Cargamos ResNet-18 con pesos preentrenados de ImageNet
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # La última capa de ResNet-18 original es Linear(512, 1000)
        # La sustituimos por nuestra cabeza para 13 clases
        backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        # Congela todo menos la cabeza (fc)
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        # Descongela todo el modelo para fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True