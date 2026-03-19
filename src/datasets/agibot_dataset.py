import os
from typing import Any, Dict, List

from torch.utils.data import Dataset


class AgibotDataset(Dataset):
    # Deutscher Kommentar:
    # Diese Klasse ist eine minimale Platzhalter-Implementierung
    # für den Datensatz des Wettbewerbs.

    def __init__(self, root_dir: str, split: str = "train") -> None:
        # Deutscher Kommentar:
        # Initialisiert das Datensatzobjekt mit Pfad und Datenteil.
        self.root_dir = root_dir
        self.split = split
        self.samples: List[Dict[str, Any]] = self._load_index()

    def _load_index(self) -> List[Dict[str, Any]]:
        # Deutscher Kommentar:
        # Lädt die Datenindexstruktur. Aktuell wird nur
        # eine leere Struktur zurückgegeben.
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.exists(split_dir):
            return []
        return []

    def __len__(self) -> int:
        # Deutscher Kommentar:
        # Gibt die Anzahl der Stichproben zurück.
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Deutscher Kommentar:
        # Gibt eine einzelne Beispielstruktur zurück.
        return {
            "observation": None,
            "action": None,
            "language": None,
            "metadata": {"index": idx, "split": self.split},
        }
