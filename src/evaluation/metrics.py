from typing import Dict

import torch


def action_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Deutscher Kommentar:
    # Berechnet den mittleren quadratischen Fehler.
    return torch.mean((pred - target) ** 2)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    # Deutscher Kommentar:
    # Gibt die Auswertungsmetriken als Wörterbuch zurück.
    mse = action_mse(pred, target)
    return {
        "action_mse": float(mse.item())
    }
