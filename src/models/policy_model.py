from typing import Any

import torch
import torch.nn as nn


class PolicyModel(nn.Module):
    # Deutscher Kommentar:
    # Einfaches Platzhaltermodell für die Reasoning-Action-Strecke.

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Any:
        # Deutscher Kommentar:
        # Führt den Vorwärtsschritt des Modells aus.
        return self.network(x)
