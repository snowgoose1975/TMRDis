from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .io import load_embedding, load_embedding_sources, load_vector, read_manifest
from .normalization import apply_normalization
from .feature_preprocessing import apply_preprocessing, load_preprocessing


def load_manifest_embedding(row: dict[str, str], preprocessing=None):
    sources = row.get("embedding_sources", "")
    if sources:
        embedding = load_embedding_sources(sources)
    else:
        path = row.get("embedding", "")
        if not path:
            raise ValueError("missing embedding source")
        embedding = load_embedding(path, row.get("embedding_key") or None)
    declared = load_preprocessing(row.get("feature_preprocessing", ""))
    if preprocessing is not None and declared is not None and preprocessing != declared:
        raise ValueError("Manifest recipe differs from checkpoint")
    return apply_preprocessing(embedding, preprocessing if preprocessing is not None else declared)


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        normalization: dict[str, Any] | None = None,
        cache_samples: bool = False,
    ):
        self.rows = read_manifest(manifest)
        self.normalization = normalization
        self.cache_samples = cache_samples
        self._samples: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache_samples and index in self._samples:
            return self._samples[index]
        row = self.rows[index]
        embedding = load_manifest_embedding(row)
        embedding = apply_normalization(embedding, self.normalization)
        sample: dict[str, Any] = {
            "id": row["id"],
            "embedding": embedding,
        }

        labels_path = row.get("labels", "")
        if not labels_path:
            raise ValueError(f"{row['id']}: training manifest needs a labels column")
        labels = load_vector(labels_path)
        if labels.shape[0] != embedding.shape[0]:
            raise ValueError(
                f"{row['id']}: labels length {labels.shape[0]} != "
                f"embedding length {embedding.shape[0]}"
            )
        valid = labels != 2
        labels = (labels == 1).astype("float32")

        mask_path = row.get("mask", "")
        if mask_path:
            mask = load_vector(mask_path).astype(bool)
            if mask.shape[0] != embedding.shape[0]:
                raise ValueError(f"{row['id']}: mask length does not match embedding")
            valid &= mask

        sample["labels"] = labels
        sample["valid"] = valid
        if self.cache_samples:
            self._samples[index] = sample
        return sample


def pad_batch(batch: list[dict[str, Any]]):
    if not batch:
        raise ValueError("empty batch")
    dimension = batch[0]["embedding"].shape[1]
    max_length = max(item["embedding"].shape[0] for item in batch)
    size = len(batch)

    embeddings = torch.zeros((size, max_length, dimension), dtype=torch.float32)
    labels = torch.zeros((size, max_length), dtype=torch.float32)
    valid = torch.zeros((size, max_length), dtype=torch.bool)
    lengths = torch.zeros(size, dtype=torch.long)
    ids: list[str] = []

    for row_index, item in enumerate(batch):
        current = item["embedding"]
        if current.shape[1] != dimension:
            raise ValueError(
                f"{item['id']}: dimension {current.shape[1]} != batch dimension {dimension}"
            )
        length = current.shape[0]
        embeddings[row_index, :length] = torch.from_numpy(current)
        labels[row_index, :length] = torch.from_numpy(item["labels"])
        valid[row_index, :length] = torch.from_numpy(item["valid"])
        lengths[row_index] = length
        ids.append(item["id"])
    return embeddings, labels, valid, ids, lengths
