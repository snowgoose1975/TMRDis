"""Recorded input reparameterization; original norms are appended last."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np


@lru_cache(maxsize=64)
def load_preprocessing(path):
    return json.loads(Path(path).read_text()) if path else None


def apply_preprocessing(embedding, metadata):
    if not metadata:
        return embedding
    values = np.asarray(embedding, dtype=np.float32)
    if metadata.get('method') == 'select_columns':
        indices = metadata.get('columns', [])
        if values.ndim != 2 or values.shape[1] != metadata.get('input_dim'):
            raise ValueError('Column selection input dimension mismatch')
        if (not indices or any(type(i) is not int for i in indices)
                or len(set(indices)) != len(indices)
                or min(indices) < 0 or max(indices) >= values.shape[1]):
            raise ValueError('Invalid selected columns')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite selection input')
        return values[:, indices]
    widths = metadata['segments']
    if values.ndim != 2 or sum(widths) != values.shape[1] or any(w <= 0 for w in widths):
        raise ValueError('Preprocessing segment widths do not match raw embedding')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite raw embedding')
    methods = metadata.get("segment_methods", [metadata["method"]] * len(widths))
    if len(methods) != len(widths) or any(m not in ("raw", "unit", "zscore") for m in methods):
        raise ValueError("Invalid per-segment preprocessing methods")
    chunks, norms, start = [], [], 0
    for width, method in zip(widths, methods):
        chunk = values[:, start:start + width]
        # Compute before any transform, in float64 to avoid squaring overflow.
        norm = np.linalg.norm(chunk.astype(np.float64), axis=1).astype(np.float32)
        norms.append(np.log1p(norm)[:, None])
        if method == 'unit':
            chunk = np.divide(chunk, norm[:, None], out=np.zeros_like(chunk), where=norm[:, None] > 0)
        elif method == 'zscore':
            mean = np.asarray(metadata['mean'][start:start + width], dtype=np.float32)
            scale = np.asarray(metadata['scale'][start:start + width], dtype=np.float32)
            if mean.shape != (width,) or scale.shape != (width,) or np.any(scale <= 0):
                raise ValueError('Malformed train-only feature statistics')
            chunk = (chunk - mean) / scale
        elif method != 'raw':
            raise ValueError('Unknown feature preprocessing method')
        chunks.append(chunk)
        start += width
    if metadata.get('append_norm'):
        chunks.extend(norms)
    result = np.concatenate(chunks, axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite preprocessed embedding')
    return result
