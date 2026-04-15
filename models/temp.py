

import torch
import torch.nn as nn
import torch.nn.functional as F



class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B,16,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,16,14,14]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),# [B,32,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,32,7,7]
        )
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)