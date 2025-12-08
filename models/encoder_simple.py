import torch
import torch.nn as nn

class SimpleSpatialEncoder(nn.Module):
    """
    A lightweight CNN encoder (LeNet-style) to prevent overfitting.
    Input: (B, 1, 160, 160) Grayscale
    Output: (B, 512, 5, 5) Spatial Map
    """
    def __init__(self, output_dim=512):
        super().__init__()
        
        # 4-Layer CNN
        # 160x160 -> 80x80 -> 40x40 -> 20x20 -> 10x10 (Strided Conv or MaxPool)
        # We want final 5x5. So we need 5 downsamples.
        
        self.features = nn.Sequential(
            # L1: 160 -> 80
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # L2: 80 -> 40
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # L3: 40 -> 20
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # L4: 20 -> 10
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # L5: 10 -> 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Adapter to match 512 channels
        self.adapter = nn.Conv2d(256, output_dim, kernel_size=1)
        
    def forward(self, x):
        # x: (B, 1, 160, 160)
        x = self.features(x) # (B, 256, 5, 5)
        x = self.adapter(x)  # (B, 512, 5, 5)
        return x
