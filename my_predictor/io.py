from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FastaRecord:
    id: str
    description: str
    sequence: str


def read_fasta(path: str | Path) -> list[FastaRecord]:
    path = Path(path)
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(_finish_record(header, sequence_parts, path, line_number))
                header = line[1:].strip()
                sequence_parts = []
                if not header:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
            else:
                if header is None:
                    raise ValueError(f"{path}:{line_number}: sequence before first header")
                sequence_parts.append(line.replace(" ", "").upper())

    if header is not None:
        records.append(_finish_record(header, sequence_parts, path, "EOF"))
    return records


def _finish_record(
    header: str,
    sequence_parts: list[str],
    path: Path,
    line_number: int | str,
) -> FastaRecord:
    sequence = "".join(sequence_parts)
    if not sequence:
        raise ValueError(f"{path}:{line_number}: empty sequence for {header!r}")
    return FastaRecord(
        id=header.split()[0],
        description=header,
        sequence=sequence,
    )


def load_embedding(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    loaded: Any = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        array = loaded
    else:
        try:
            names = list(loaded.files)
            preferred = [key] if key else []
            preferred += ["embedding", "embeddings", "residue_embeddings", "x", "arr_0"]
            selected = next((name for name in preferred if name and name in names), None)
            if selected is None:
                two_dimensional = [
                    name for name in names
                    if getattr(loaded[name], "ndim", None) == 2
                ]
                if len(two_dimensional) == 1:
                    selected = two_dimensional[0]
            if selected is None:
                raise ValueError(
                    f"{path}: cannot find a unique 2-D embedding array; keys={names}"
                )
            array = loaded[selected]
        finally:
            loaded.close()

    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{path}: expected [length, dimension], got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{path}: embedding is not numeric: {array.dtype}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: embedding contains NaN or infinite values")
    return array



def _parse_embedding_sources(
    sources: str | list[Any] | tuple[Any, ...],
) -> list[dict[str, str]]:
    if isinstance(sources, str):
        try:
            decoded: Any = json.loads(sources)
        except json.JSONDecodeError as exc:
            raise ValueError("embedding_sources must be valid JSON") from exc
    else:
        decoded = sources
    if not isinstance(decoded, (list, tuple)) or not decoded:
        raise ValueError("embedding_sources must be a non-empty JSON list")

    parsed: list[dict[str, str]] = []
    for index, item in enumerate(decoded):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            raise ValueError(f"embedding_sources[{index}] must be a string or object")
        source_path = item.get("path") or item.get("embedding")
        if not source_path:
            raise ValueError(f"embedding_sources[{index}] is missing path")
        spec = {"path": str(source_path)}
        for key in ("key", "embedding_key", "family", "name"):
            value = item.get(key)
            if value is not None and str(value):
                spec[key] = str(value)
        parsed.append(spec)
    return parsed


def load_embedding_sources(
    sources: str | list[Any] | tuple[Any, ...],
) -> np.ndarray:
    """Load and concatenate same-length residue embeddings in listed order."""
    parsed = _parse_embedding_sources(sources)
    arrays = [
        load_embedding(spec["path"], spec.get("key") or spec.get("embedding_key"))
        for spec in parsed
    ]
    length = arrays[0].shape[0]
    for index, array in enumerate(arrays[1:], start=1):
        if array.shape[0] != length:
            raise ValueError(
                "embedding_sources have different residue lengths: "
                f"source 0 has {length}, source {index} has {array.shape[0]}"
            )
    return np.concatenate(arrays, axis=1)

def load_vector(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        array = np.asarray(value)
    else:
        loaded: Any = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            array = loaded
        else:
            try:
                names = list(loaded.files)
                preferred = [key] if key else []
                preferred += ["labels", "label", "mask", "y", "arr_0"]
                selected = next((name for name in preferred if name and name in names), None)
                if selected is None and len(names) == 1:
                    selected = names[0]
                if selected is None:
                    raise ValueError(f"{path}: cannot select a vector array; keys={names}")
                array = loaded[selected]
            finally:
                loaded.close()

    array = np.asarray(array).reshape(-1)
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{path}: vector is not numeric: {array.dtype}")
    return array


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing TSV header")
        if "id" not in reader.fieldnames:
            raise ValueError(f"{path}: missing columns ['id']")
        if "embedding" not in reader.fieldnames and "embedding_sources" not in reader.fieldnames:
            raise ValueError(f"{path}: needs embedding or embedding_sources column")

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if not row.get("id"):
                raise ValueError(f"{path}:{row_number}: id is required")
            has_single = bool(row.get("embedding"))
            has_bundle = bool(row.get("embedding_sources"))
            if has_single == has_bundle:
                raise ValueError(
                    f"{path}:{row_number}: provide exactly one of embedding and embedding_sources"
                )
            resolved = {key: (value or "") for key, value in row.items()}
            if has_single:
                value = Path(resolved["embedding"])
                if not value.is_absolute():
                    resolved["embedding"] = str((path.parent / value).resolve())
            else:
                parsed = _parse_embedding_sources(resolved["embedding_sources"])
                for source in parsed:
                    source_path = Path(source["path"])
                    if not source_path.is_absolute():
                        source["path"] = str((path.parent / source_path).resolve())
                resolved["embedding_sources"] = json.dumps(
                    parsed,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            for column in ("labels", "mask", "feature_preprocessing"):
                value = resolved.get(column, "")
                if value and not Path(value).is_absolute():
                    resolved[column] = str((path.parent / value).resolve())
            rows.append(resolved)
    if not rows:
        raise ValueError(f"{path}: no samples")
    return rows


def read_embedding_index(path: str | Path) -> dict[str, Path]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing TSV header")
        id_column = next(
            (name for name in ("id", "sequence_id", "native_id") if name in reader.fieldnames),
            None,
        )
        path_column = next(
            (name for name in ("embedding", "path", "file") if name in reader.fieldnames),
            None,
        )
        if id_column is None or path_column is None:
            raise ValueError(f"{path}: expected id and embedding/path columns")
        result: dict[str, Path] = {}
        for row in reader:
            identifier = row.get(id_column, "")
            value = row.get(path_column, "")
            if identifier and value:
                candidate = Path(value)
                result[identifier] = candidate if candidate.is_absolute() else (path.parent / candidate).resolve()
    return result
