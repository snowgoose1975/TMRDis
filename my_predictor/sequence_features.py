"""Canonical 42-channel amino-acid sequence features used by paper-v1."""
from __future__ import annotations

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
WINDOW_RADIUS = 10
FEATURE_RECIPE = "paper_v1_sequence42_centered_window21"


def sequence_features(sequence: str) -> np.ndarray:
    """Return 21-way residue identity plus centered local frequencies.

    Standard amino acids use the order in :data:`AMINO_ACIDS`; every other
    uppercase symbol maps to category 20. Each frequency window is clipped at
    sequence boundaries and includes at most ten residues on either side.
    """
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must be non-empty text")
    if sequence != sequence.upper() or any(character.isspace() for character in sequence):
        raise ValueError("sequence must be uppercase text without whitespace")
    length = len(sequence)
    one_hot = np.zeros((length, 21), dtype=np.float32)
    aa_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    indices = np.fromiter((aa_index.get(letter, 20) for letter in sequence), dtype=np.int64, count=length)
    one_hot[np.arange(length), indices] = 1.0
    cumulative = np.vstack((np.zeros((1, 21), dtype=np.float64), np.cumsum(one_hot, axis=0, dtype=np.float64)))
    positions = np.arange(length)
    starts = np.maximum(positions - WINDOW_RADIUS, 0)
    ends = np.minimum(positions + WINDOW_RADIUS + 1, length)
    local = (cumulative[ends] - cumulative[starts]) / (ends - starts)[:, None]
    result = np.concatenate((one_hot, local.astype(np.float32)), axis=1)
    if result.shape != (length, 42) or not np.isfinite(result).all():
        raise RuntimeError("sequence feature construction produced invalid values")
    return result
