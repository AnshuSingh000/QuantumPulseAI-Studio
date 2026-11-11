import torch
from torch import nn

class PulsePredictionNetwork(nn.Module):
    def __init__(self, n_steps=20, max_amplitude=4.0):
        super().__init__()
        self.max_amplitude = max_amplitude
        self.network = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, n_steps), nn.Tanh(),
        )
    def forward(self, angles):
        return self.max_amplitude * self.network(angles)
