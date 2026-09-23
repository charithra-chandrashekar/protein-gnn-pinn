"""PyTorch Dataset/DataLoader wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from src.data.graph_builder import ProteinGraph, load_graph


class ProteinGraphDataset(Dataset):
    """Dataset over preprocessed protein graphs."""

    def __init__(
        self,
        split_file: str | Path,
        processed_dir: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        with open(split_file) as f:
            self.protein_ids = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> ProteinGraph:
        protein_id = self.protein_ids[idx]
        graph = load_graph(str(self.processed_dir / f"{protein_id}.pt"))
        if self.transform is not None:
            graph = self.transform(graph)
        return graph
