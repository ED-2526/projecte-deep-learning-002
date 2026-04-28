import torch.nn as nn

# Red neuronal convolucional para clasificación de autores WikiArt

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=25):
        super(ConvNet, self).__init__()
        
        # 3 canales (RGB) en lugar de 1 (blanco y negro)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Capa extra para aprender patrones más complejos
        self.layer3 = nn.Sequential(
            nn.Conv2d(kernels[1], kernels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Capas fully connected
        self.fc1 = nn.Linear(16 * 16 * kernels[2], 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out