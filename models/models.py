import torch.nn as nn

# Red neuronal convolucional para clasificación de autores WikiArt

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=25):
        super(ConvNet, self).__init__()
        
        # CAPA 1 — bordes y cambios de color básicos
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(kernels[0]), # estabilitza el entrenamiento para que ningun filtro domine sobre otros
            nn.ReLU(), # elimina valors negatius
            nn.MaxPool2d(kernel_size=2, stride=2)) # reduce la imagen a la mitad quedándose solo con las activaciones más fuertes
        
        # CAPA 2 — texturas y formas
        # Combina patrones simples para detectar cosas más complejas — texturas, formas, estilos de pincelada.
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(kernels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # CAPA 3 — patrones de pincelada y estilo
        # apaga aleatoriamente el 30% de los filtros durante el entrenamiento.
        self.layer3 = nn.Sequential(
            nn.Conv2d(kernels[1], kernels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(kernels[2]),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # CAPA 4 — características de alto nivel
        self.layer4 = nn.Sequential(
            nn.Conv2d(kernels[2], kernels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(kernels[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # CAPAS FULLY CONNECTED
        self.fc1 = nn.Linear(14 * 14 * kernels[3], 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, classes)
        """los 256 mapas de 14x14 se convierten en un vector de 50.176 números 
        y se van comprimiendo progresivamente con dropout hasta llegar a 29 neuronas 
        — una por autor. La que tenga más puntuación es la predicción."""
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out