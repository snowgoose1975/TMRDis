"""Paper-v1 training for the fixed study:bigru model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ManifestDataset, pad_batch
from .head_study_models import build as build_study_head
from .io import load_embedding, load_vector, read_manifest
from .metrics import average_precision
from .sequence_features import FEATURE_RECIPE

PAPER_SEEDS = (20260926, 20260927, 20260928, 20260929, 20260930)
ALLOWED_FEATURE_RECIPES = {FEATURE_RECIPE, "prepare_mlp_search_input.sequence_features"}
FAMILY_INPUTS = {
    "prott5": {"layer": "24", "plm_width": 1024, "input_dim": 1066},
    "esmc": {"layer": "36", "plm_width": 1152, "input_dim": 1194},
}
HEAD_SPEC = {"name": "bigru", "head": "bigru", "width": 256}
DEFAULT_LR = {"prott5": 0.0003, "esmc": 0.001}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _sources(row: dict[str, str]) -> list[dict[str, str]]:
    raw = row.get("embedding_sources", "")
    if not raw:
        raise ValueError(f"{row['id']}: paper manifests require declared embedding_sources")
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row['id']}: malformed embedding_sources JSON") from exc
    if not isinstance(items, list) or len(items) != 2:
        raise ValueError(f"{row['id']}: expected one PLM and one 42-column sequence source")
    result = []
    for item in items:
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            raise ValueError(f"{row['id']}: malformed embedding source")
        path = item.get("path") or item.get("embedding")
        if not path:
            raise ValueError(f"{row['id']}: embedding source lacks path")
        result.append({
            "path": str(path),
            "family": str(item.get("family", "")).lower(),
            "key": str(item.get("key") or item.get("embedding_key") or ""),
            "name": str(item.get("name", "")).lower(),
        })
    return result


_SHAPE_CACHE: dict[tuple[str, str, str], tuple[int, int]] = {}


def _source_shape(path: Path, key: str, *, expected_key: str | None = None) -> tuple[int, int]:
    path = path.resolve()
    stat = path.stat()
    signature = f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"
    cache_key = (str(path), key, signature)
    if cache_key in _SHAPE_CACHE:
        return _SHAPE_CACHE[cache_key]
    if expected_key is not None and key != expected_key:
        raise ValueError(f"PLM source key must be exactly {expected_key!r}, got {key!r}")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        array = loaded
    else:
        try:
            selected_key = expected_key if expected_key is not None else (key or None)
            if selected_key and selected_key not in loaded.files:
                if expected_key is not None:
                    raise KeyError(f"{path} has no required NPZ member {expected_key!r}; keys={list(loaded.files)}")
                raise KeyError(f"{path} has no requested embedding key {selected_key!r}; keys={list(loaded.files)}")
            if selected_key is None:
                candidates = [name for name in loaded.files if np.asarray(loaded[name]).ndim == 2]
                if len(candidates) != 1:
                    raise ValueError(f"{path}: cannot select a unique 2-D feature array; keys={list(loaded.files)}")
                selected_key = candidates[0]
            array = loaded[selected_key]
        finally:
            loaded.close()
    array = np.asarray(array)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{path}:{key} must be a numeric [length, dimension] matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite values in {path}:{key}")
    shape = (int(array.shape[0]), int(array.shape[1]))
    _SHAPE_CACHE[cache_key] = shape
    return shape


def _identity_and_width(row: dict[str, str], family: str) -> tuple[tuple[tuple[str, str, str], ...], int, int]:
    expected = FAMILY_INPUTS[family]
    sources = _sources(row)
    expected_key = f"layer_{expected['layer']}"
    if sources[0]["key"] != expected_key:
        raise ValueError(f"{row['id']}: PLM source key must be exactly {expected_key!r}, got {sources[0]['key']!r}")
    family_text = (sources[0]["family"] + " " + sources[0]["name"]).lower().replace("-", "").replace("_", "")
    if family not in family_text:
        raise ValueError(f"{row['id']}: PLM source family does not identify {family!r}")
    sequence_source = sources[1]
    if sequence_source["family"] != "derived_sequence" or sequence_source["key"] != "embedding":
        raise ValueError(f"{row['id']}: sequence42 source must be family=derived_sequence, key=embedding")
    feature_recipe = row.get("feature_recipe", "").strip()
    if feature_recipe and feature_recipe not in ALLOWED_FEATURE_RECIPES:
        raise ValueError(f"{row['id']}: unsupported sequence feature recipe {feature_recipe!r}")
    shapes = [
        _source_shape(Path(sources[0]["path"]), sources[0]["key"], expected_key=expected_key),
        _source_shape(Path(sources[1]["path"]), sources[1]["key"]),
    ]
    if shapes[0][0] != shapes[1][0]:
        raise ValueError(f"{row['id']}: PLM and sequence feature lengths differ")
    if shapes[0][1] != expected["plm_width"]:
        raise ValueError(f"{row['id']}: {family} PLM width {shapes[0][1]} != {expected['plm_width']}")
    if shapes[1][1] != 42:
        raise ValueError(f"{row['id']}: expected a single 42-column sequence source, got {shapes[1][1]}")
    labels = load_vector(row["labels"])
    if not np.isin(labels, (0, 1, 2)).all():
        observed = np.unique(labels).tolist()
        raise ValueError(f"{row['id']}: labels must contain only 0, 1, or 2; got {observed[:10]}")
    if labels.shape[0] != shapes[0][0]:
        raise ValueError(f"{row['id']}: labels length {labels.shape[0]} != embedding length {shapes[0][0]}")
    mask_path = row.get("mask", "")
    if mask_path:
        mask = load_vector(mask_path)
        if mask.shape[0] != labels.shape[0]:
            raise ValueError(f"{row['id']}: mask length {mask.shape[0]} != labels length {labels.shape[0]}")
    identity = tuple((source["family"], source["key"], source["name"]) for source in sources)
    return identity, sum(width for _length, width in shapes), shapes[0][0]

def _validate_manifests(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]], family: str) -> tuple[int, tuple[tuple[str, str, str], ...]]:
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    identity = None
    feature_recipe_identity = None
    for rows, split, seen in ((train_rows, "train", train_ids), (val_rows, "val", val_ids)):
        for row in rows:
            protein_id = row["id"]
            if protein_id in seen:
                raise ValueError(f"duplicate ID {protein_id!r} in {split} manifest")
            seen.add(protein_id)
            if row.get("split", "").strip().lower() != split:
                raise ValueError(f"{protein_id}: manifest requires split={split!r}")
            if not row.get("labels"):
                raise ValueError(f"{protein_id}: labels are required")
            if row.get("feature_preprocessing", "").strip():
                raise ValueError(f"{protein_id}: paper training requires raw features")
            current, width, _length = _identity_and_width(row, family)
            current_recipe = row.get("feature_recipe", "").strip()
            if feature_recipe_identity is None:
                feature_recipe_identity = current_recipe
            elif feature_recipe_identity != current_recipe:
                raise ValueError(f"{protein_id}: sequence feature recipe differs across split rows")
            if width != FAMILY_INPUTS[family]["input_dim"]:
                raise ValueError(f"{protein_id}: {family} input width {width} != {FAMILY_INPUTS[family]['input_dim']}")
            if identity is None:
                identity = current
            elif identity != current:
                raise ValueError(f"{protein_id}: embedding source identity/order differs")
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"train/validation protein ID leakage: {sorted(overlap)[:5]}")
    if identity is None:
        raise ValueError("empty train or validation manifest")
    return FAMILY_INPUTS[family]["input_dim"], identity

def _asset_paths(rows: list[dict[str, str]]) -> set[Path]:
    paths: set[Path] = set()
    for row in rows:
        for field in ("labels", "mask", "feature_preprocessing"):
            if row.get(field):
                paths.add(Path(row[field]).resolve())
        for source in _sources(row):
            paths.add(Path(source["path"]).resolve())
    return paths


def _stat_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"


def _hash(path: Path, cache: dict[str, str], signatures: dict[str, str]) -> str:
    path = path.resolve()
    key = str(path)
    signature = _stat_signature(path)
    cached_key = f"{key}|{signature}"
    signatures[key] = signature
    if cached_key not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if _stat_signature(path) != signature:
            raise RuntimeError(f"input changed while hashing: {path}")
        cache[cached_key] = digest.hexdigest()
    return cache[cached_key]


def _verify_signatures(signatures: dict[str, str]) -> None:
    for name, expected in signatures.items():
        if _stat_signature(Path(name)) != expected:
            raise RuntimeError(f"training input changed during this run: {name}")


def _input_hashes(train_path: Path, val_path: Path, train_rows: list[dict[str, str]], val_rows: list[dict[str, str]], cache: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    hashes = cache if cache is not None else {}
    signatures: dict[str, str] = {}
    manifest_paths = sorted({train_path.resolve(), val_path.resolve()})
    assets = sorted(_asset_paths(train_rows + val_rows))
    paths = manifest_paths + assets
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    result = {
        "manifests": {str(path): _hash(path, hashes, signatures) for path in manifest_paths},
        "assets": {str(path): _hash(path, hashes, signatures) for path in assets},
    }
    _verify_signatures(signatures)
    return result, signatures

def _epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    known = 0
    score_chunks, label_chunks, mask_chunks = [], [], []
    for embeddings, labels, valid, _ids, lengths in loader:
        embeddings, labels, valid = embeddings.to(device), labels.to(device), valid.to(device)
        with torch.set_grad_enabled(training):
            logits = model(embeddings, lengths)
            selected_logits, selected_labels = logits[valid], labels[valid]
            if selected_labels.numel() == 0:
                continue
            loss = torch.nn.functional.binary_cross_entropy_with_logits(selected_logits, selected_labels, reduction="mean")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        n = int(selected_labels.numel())
        known += n
        loss_sum += float(loss.detach().cpu()) * n
        score_chunks.append(logits.detach().cpu().numpy().reshape(-1))
        label_chunks.append(labels.detach().cpu().numpy().reshape(-1))
        mask_chunks.append(valid.detach().cpu().numpy().reshape(-1))
    if known == 0:
        raise ValueError("split has no known residue labels")
    scores, gold, masks = map(np.concatenate, (score_chunks, label_chunks, mask_chunks))
    return {"loss": loss_sum / known, "average_precision": float(average_precision(scores, gold, masks)), "known_residues": known}


def _finite_or_none(value: float | int) -> float | int | None:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return int(value)


def _train_one(*, train_manifest: str | Path, val_manifest: str | Path, family: str, seed: int, output_dir: str | Path, epochs: int = 10, batch_size: int = 4, lr: float, lr_override: bool = False, device: str = "cpu", threads: int = 1, cache_embeddings: bool = False, hash_cache: dict[str, str] | None = None, command: list[str] | None = None, paper_matrix_scope: bool = False) -> dict[str, Any]:
    if family not in FAMILY_INPUTS:
        raise ValueError(f"unsupported family {family!r}")
    if seed not in PAPER_SEEDS:
        raise ValueError(f"seed must be one of {PAPER_SEEDS}")
    if epochs < 1 or batch_size < 1 or threads < 1:
        raise ValueError("epochs, batch-size and threads must be positive")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be positive and finite")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    train_path, val_path = Path(train_manifest).resolve(), Path(val_manifest).resolve()
    manifest_signatures_before_read = {str(path): _stat_signature(path) for path in (train_path, val_path)}
    train_rows, val_rows = read_manifest(train_path), read_manifest(val_path)
    hashes, input_signatures = _input_hashes(train_path, val_path, train_rows, val_rows, hash_cache)
    for path, signature in manifest_signatures_before_read.items():
        if input_signatures[str(Path(path).resolve())] != signature:
            raise RuntimeError(f"manifest changed while being read and hashed: {path}")
    input_dim, source_identity = _validate_manifests(train_rows, val_rows, family)
    _verify_signatures(input_signatures)

    torch.set_num_threads(threads)
    _set_seed(seed)
    train_data = ManifestDataset(train_path, normalization={"method": "none"}, cache_samples=cache_embeddings)
    val_data = ManifestDataset(val_path, normalization={"method": "none"}, cache_samples=cache_embeddings)
    for dataset in (train_data, val_data):
        for index in range(len(dataset)):
            sample = dataset[index]
            if sample["embedding"].ndim != 2 or sample["embedding"].shape[1] != input_dim:
                raise ValueError(f"{sample['id']}: feature width mismatch")
            if sample["labels"].shape != (sample["embedding"].shape[0],) or sample["valid"].shape != sample["labels"].shape:
                raise ValueError(f"{sample['id']}: residue labels/mask do not align with embedding")
        if not any(bool(dataset[i]["valid"].any()) for i in range(len(dataset))):
            raise ValueError("split has no known residue labels")

    _verify_signatures(input_signatures)
    target = _device(device)
    model = build_study_head(input_dim, HEAD_SPEC).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_batch)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pad_batch)

    config = {
        "protocol": "paper-v1-study-bigru",
        "architecture": "study:bigru",
        "family": family,
        "seed": seed,
        "train_manifest": str(train_path),
        "val_manifest": str(val_path),
        "input_dim": input_dim,
        "embedding_sources": train_rows[0].get("embedding_sources"),
        "source_identity": source_identity,
        "feature_recipe": train_rows[0].get("feature_recipe") or None,
        "feature_preprocessing": None,
        "embedding_normalization": "none",
        "hidden_dim": 256,
        "dropout": 0.1,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "learning_rate_overridden": lr_override,
        "paper_member_config": not lr_override and lr == DEFAULT_LR[family] and epochs == 10 and batch_size == 4,
        "paper_matrix_scope": paper_matrix_scope,
        "optimizer": {"name": "AdamW", "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        "loss": "BCEWithLogits_mean_over_known_residues",
        "checkpoint_selection": "validation_average_precision_on_logits_strictly_best",
        "device_requested": device,
        "device_used": str(target),
        "threads": threads,
        "cache_embeddings": bool(cache_embeddings),
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "input_hashes": hashes,
        "command": command or [],
    }
    output.mkdir(parents=True)
    _write_json(output / "config.json", config)
    history: list[dict[str, Any]] = []
    best_ap, best_epoch, fallback_epoch = -float("inf"), None, None
    checkpoint_path = output / "best.pt"
    for epoch in range(1, epochs + 1):
        _verify_signatures(input_signatures)
        train_metrics = _epoch(model, train_loader, target, optimizer)
        with torch.inference_mode():
            val_metrics = _epoch(model, val_loader, target, None)
        ap = float(val_metrics["average_precision"])
        history.append({
            "epoch": epoch,
            "train": {k: _finite_or_none(v) for k, v in train_metrics.items()},
            "validation": {k: _finite_or_none(v) for k, v in val_metrics.items()},
        })
        save = False
        if math.isfinite(ap):
            if best_epoch is None or ap > best_ap:
                best_ap, best_epoch, save = ap, epoch, True
        elif not checkpoint_path.exists():
            # Keep first-epoch weights only when AP is undefined; never replace a
            # finite best checkpoint with a lower or undefined validation result.
            fallback_epoch, save = epoch, True
        if save:
            payload = {
                "model": "study:bigru",
                "architecture": "study:bigru",
                "input_dim": input_dim,
                "hidden_dim": 256,
                "dropout": 0.1,
                "embedding_normalization": {"method": "none"},
                "feature_preprocessing": None,
                "selected_epoch": best_epoch,
                "validation_average_precision": best_ap if best_epoch is not None else None,
                "checkpoint_selection": "validation_average_precision_on_logits",
                "state_dict": model.state_dict(),
                "config": config,
            }
            pending = checkpoint_path.with_name(checkpoint_path.name + ".pending")
            torch.save(payload, pending)
            pending.replace(checkpoint_path)
        _write_json(output / "history.json", history)
        print(
            f"family={family} seed={seed} epoch={epoch}/{epochs} "
            f"loss={_finite_or_none(train_metrics.get('loss'))} "
            f"valAP={_finite_or_none(ap)}",
            flush=True,
        )
        _verify_signatures(input_signatures)

    summary = {
        "protocol": "paper-v1-study-bigru",
        "spec": {"name": "bigru", "head": "bigru", "width": 256, "dropout": 0.1},
        "family": family,
        "seed": seed,
        "learning_rate": lr,
        "learning_rate_overridden": lr_override,
        "rate": lr,
        "selected_epoch": best_epoch,
        "metrics": {"average_precision": best_ap if best_epoch is not None else None},
        "best_epoch": best_epoch,
        "fallback_epoch": fallback_epoch if best_epoch is None else None,
        "best_validation_average_precision": best_ap if best_epoch is not None else None,
        "checkpoint": "best.pt",
        "checkpoint_selection": config["checkpoint_selection"],
        "epoch_log": "history.json",
        "config": "config.json",
        "input_hashes": hashes,
        "epochs_completed": len(history),
        "paper_member_config": not lr_override and lr == DEFAULT_LR[family] and epochs == 10 and batch_size == 4,
        "paper_matrix_scope": paper_matrix_scope,
    }
    _write_json(output / "summary.json", summary)
    return summary


def _single_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train paper-v1 study:bigru")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--family", choices=tuple(FAMILY_INPUTS), required=True)
    p.add_argument("--seed", type=int, choices=PAPER_SEEDS, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cache-embeddings", action="store_true")
    return p


def train_main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _single_parser().parse_args(argv)
    return _train_one(
        train_manifest=args.train_manifest, val_manifest=args.val_manifest,
        family=args.family, seed=args.seed, output_dir=args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=DEFAULT_LR[args.family] if args.lr is None else args.lr,
        lr_override=args.lr is not None,
        device=args.device, threads=args.threads, cache_embeddings=args.cache_embeddings,
        command=["paper-train", *(argv if argv is not None else sys.argv[1:])],
    )


def train_matrix_main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description="Run the paper family-by-seed training matrix")
    p.add_argument("--prott5-train-manifest", required=True)
    p.add_argument("--prott5-val-manifest", required=True)
    p.add_argument("--esmc-train-manifest", required=True)
    p.add_argument("--esmc-val-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--family", action="append", choices=tuple(FAMILY_INPUTS))
    p.add_argument("--seed", action="append", type=int, choices=PAPER_SEEDS)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cache-embeddings", action="store_true")
    a = p.parse_args(argv)
    if a.epochs < 1 or a.batch_size < 1 or a.threads < 1:
        raise ValueError("epochs, batch-size and threads must be positive")
    if a.lr is not None and (not math.isfinite(a.lr) or a.lr <= 0):
        raise ValueError("lr override must be positive and finite")
    families, seeds = a.family or list(FAMILY_INPUTS), a.seed or list(PAPER_SEEDS)
    if len(set(families)) != len(families) or len(set(seeds)) != len(seeds):
        raise ValueError("family and seed selections must not contain duplicates")
    root = Path(a.output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing matrix output: {root}")
    root.mkdir(parents=True)
    manifests = {
        "prott5": (a.prott5_train_manifest, a.prott5_val_manifest),
        "esmc": (a.esmc_train_manifest, a.esmc_val_manifest),
    }
    state: dict[str, Any] = {
        "protocol": "paper-v1-study-bigru",
        "phase": "running",
        "paper_matrix_scope": set(families) == set(FAMILY_INPUTS) and set(seeds) == set(PAPER_SEEDS) and a.epochs == 10 and a.batch_size == 4 and a.lr is None,
        "families": families,
        "seeds": seeds,
        "learning_rates": {family: (a.lr if a.lr is not None else DEFAULT_LR[family]) for family in families},
        "learning_rate_overridden": a.lr is not None,
        "expected_runs": len(families) * len(seeds),
        "completed": [],
        "failed": [],
    }
    _write_json(root / "run-index.json", state)
    hash_cache: dict[str, str] = {}
    for family in families:
        train_manifest, val_manifest = manifests[family]
        for seed in seeds:
            run_name = f"{family}-seed-{seed}"
            run_lr = DEFAULT_LR[family] if a.lr is None else a.lr
            command = [
                "paper-train", "--train-manifest", str(Path(train_manifest).resolve()),
                "--val-manifest", str(Path(val_manifest).resolve()), "--family", family,
                "--seed", str(seed), "--output-dir", str(root / run_name),
                "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
                "--lr", str(run_lr), "--device", a.device, "--threads", str(a.threads),
            ]
            if a.cache_embeddings:
                command.append("--cache-embeddings")
            state["active_run"] = run_name
            state["active_command"] = command
            _write_json(root / "run-index.json", state)
            try:
                _train_one(
                    train_manifest=train_manifest, val_manifest=val_manifest, family=family,
                    seed=seed, output_dir=root / run_name, epochs=a.epochs,
                    batch_size=a.batch_size, lr=run_lr, lr_override=a.lr is not None,
                    device=a.device, threads=a.threads,
                    cache_embeddings=a.cache_embeddings, hash_cache=hash_cache, command=command,
                    paper_matrix_scope=state["paper_matrix_scope"],
                )
            except Exception as exc:
                state["failed"].append({"run": run_name, "command": command, "error": repr(exc)})
                state["phase"] = "failed"
                state.pop("active_run", None)
                state.pop("active_command", None)
                _write_json(root / "run-index.json", state)
                raise
            state["completed"].append({"run": run_name, "summary": f"{run_name}/summary.json", "command": command})
            state.pop("active_run", None)
            state.pop("active_command", None)
            _write_json(root / "run-index.json", state)
    state["phase"] = "complete"
    state.pop("active_run", None)
    state.pop("active_command", None)
    if state["paper_matrix_scope"]:
        members = []
        for seed in PAPER_SEEDS:
            for family in ("prott5", "esmc"):
                run_name = f"{family}-seed-{seed}"
                run_dir = root / run_name
                summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
                ap = summary["metrics"]["average_precision"]
                epoch = summary["selected_epoch"]
                if epoch is None or ap is None or not math.isfinite(float(ap)):
                    state["phase"] = "complete_without_members"
                    state["member_table"] = None
                    state["member_table_reason"] = (
                        f"{run_name} has no finite best validation AP; members.tsv was not generated"
                    )
                    _write_json(root / "run-index.json", state)
                    return state
                checkpoint = run_dir / "best.pt"
                members.append({
                    "member": f"{family}-s{seed}",
                    "branch": family,
                    "seed": seed,
                    "learning_rate": summary["learning_rate"],
                    "selected_epoch": epoch,
                    "validation_AP": ap,
                    "checkpoint_sha256": _hash(checkpoint, {}, {}),
                    "checkpoint": f"{run_name}/best.pt",
                })
        import csv
        table = root / "members.tsv"
        fields = ["member", "branch", "seed", "learning_rate", "selected_epoch", "validation_AP", "checkpoint_sha256", "checkpoint"]
        with table.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(members)
        state["member_table"] = "members.tsv"
    _write_json(root / "run-index.json", state)
    return state


if __name__ == "__main__":
    print(json.dumps(train_main(), indent=2, sort_keys=True, allow_nan=False))
