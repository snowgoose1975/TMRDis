"""Portable paper-v1 input preparation and ten-member inference."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from my_predictor.head_study_models import build
from my_predictor.io import read_fasta, read_manifest
from my_predictor.sequence_features import FEATURE_RECIPE, sequence_features

FAMILIES = ("prott5", "esmc")
SEEDS = (20260926, 20260927, 20260928, 20260929, 20260930)
PLM_WIDTHS = {"prott5": 1024, "esmc": 1152}
MODEL_WIDTHS = {family: width + 42 for family, width in PLM_WIDTHS.items()}
from .paper import config_directory
PAPER_MEMBERS = config_directory() / "paper-v1-members.tsv"
MEMBER_FIELDS = ("member", "branch", "seed", "learning_rate", "selected_epoch",
                 "validation_AP", "checkpoint_sha256", "checkpoint")


def _load_exact_embedding(path: str | Path, key: str, expected_family: str) -> np.ndarray:
    """Load an array with an exact NPZ key, without fallback behavior."""
    path = Path(path)
    expected_key = ("layer_24" if expected_family == "prott5" else
                    "layer_36" if expected_family == "esmc" else "embedding")
    if key != expected_key:
        raise ValueError(f"{expected_family} inputs must explicitly select {expected_key}, got {key!r}")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        array = loaded
    else:
        try:
            if key not in loaded.files:
                raise KeyError(f"{path} has no requested embedding key {key!r}; keys={list(loaded.files)}")
            array = loaded[key]
        finally:
            loaded.close()
    array = np.asarray(array)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{path}:{key} must be a numeric [length, dimension] array")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite values in {path}:{key}")
    return array


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _cached_digest(path: str | Path, cache: dict[str, str]) -> str:
    resolved = str(Path(path).resolve())
    if resolved not in cache:
        cache[resolved] = _digest(Path(resolved))
    return cache[resolved]


def _native_metadata_literal(path: str | Path) -> str | None:
    """Return an NPZ producer metadata scalar as literal text, without parsing it."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        return None
    with np.load(path, allow_pickle=False) as loaded:
        if "__metadata_json__" not in loaded.files:
            return None
        value = np.asarray(loaded["__metadata_json__"])
        if value.size == 1:
            item = value.reshape(-1)[0].item()
            if isinstance(item, bytes):
                return item.decode("utf-8", errors="replace")
            return item if isinstance(item, str) else repr(item)
        return repr(value.tolist())


def _read_member_table(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(MEMBER_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"member table missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if len(rows) != 10:
        raise ValueError(f"paper-v1 member table must contain exactly 10 rows, got {len(rows)}")
    seen = set()
    for row in rows:
        family = row["branch"]
        try:
            seed = int(row["seed"])
            epoch = int(row["selected_epoch"])
            ap = float(row["validation_AP"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid member metadata row: {row}") from exc
        if family not in FAMILIES or seed not in SEEDS:
            raise ValueError(f"unexpected paper-v1 branch/seed: {family}/{seed}")
        pair = family, seed
        if pair in seen:
            raise ValueError(f"duplicate member branch/seed: {pair}")
        seen.add(pair)
        if row["member"] != f"{family}-s{seed}":
            raise ValueError(f"unexpected historical member name for {pair}: {row['member']!r}")
        row["score_column"] = f"ours_{family}_s{seed}"
        if epoch < 1 or not math.isfinite(ap):
            raise ValueError(f"invalid selected epoch/AP for {pair}")
        digest = row["checkpoint_sha256"].lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA256 for {row['member']}")
        if not row["checkpoint"]:
            raise ValueError(f"missing checkpoint path for {row['member']}")
        row["seed"] = str(seed)
        row["selected_epoch"] = str(epoch)
        row["validation_AP"] = repr(ap)
        row["checkpoint_sha256"] = digest
    expected = {(family, seed) for family in FAMILIES for seed in SEEDS}
    if seen != expected:
        raise ValueError("member table must contain each of the 10 paper-v1 branch/seed pairs")
    return sorted(rows, key=lambda r: (FAMILIES.index(r["branch"]), int(r["seed"])))


def _checkpoint_identity_status(members: list[dict[str, Any]]) -> str:
    """Classify weights by their exact branch/seed SHA map against bundled paper-v1."""
    official = {
        (row["branch"], int(row["seed"])): row["checkpoint_sha256"]
        for row in _read_member_table(PAPER_MEMBERS)
    }
    supplied = {
        (member["branch"], int(member["seed"])): member["checkpoint_sha256"]
        for member in members
    }
    return "original" if supplied == official else "retrained_or_custom"


def _checkpoint_path(models_dir: str | Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (Path(models_dir) / candidate)


def verify_models(models_dir: str | Path, members_tsv: str | Path = PAPER_MEMBERS) -> list[dict[str, Any]]:
    """Verify all checkpoint digests/contracts before deserializing any weights."""
    rows = _read_member_table(members_tsv)
    root = Path(models_dir)
    verified = []
    for row in rows:
        checkpoint_path = _checkpoint_path(root, row["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        actual = _digest(checkpoint_path)
        if actual != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint SHA256 mismatch for {row['member']}")
        summary_path = checkpoint_path.with_name("summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        spec = summary.get("spec", {"head": "bigru", "width": 256})
        if not isinstance(spec, dict) or spec.get("head") != "bigru":
            raise ValueError(f"{row['member']} summary must declare study head bigru")
        if int(spec.get("width", -1)) != 256:
            raise ValueError(f"{row['member']} must use study width 256")
        if "dropout" in spec and not math.isclose(float(spec["dropout"]), 0.1, abs_tol=1e-12):
            raise ValueError(f"{row['member']} must use dropout 0.1")
        verified.append({**row, "checkpoint_path": checkpoint_path, "summary": summary, "spec": spec})
    return verified


def _load_members(models_dir: str | Path, members_tsv: str | Path, device: torch.device):
    verified = verify_models(models_dir, members_tsv)
    members = {family: [] for family in FAMILIES}
    # Hash every artifact before any torch deserialization.
    for entry in verified:
        checkpoint = torch.load(entry["checkpoint_path"], map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != "study:bigru":
            raise ValueError(f"{entry['member']} checkpoint architecture is not study:bigru")
        if int(checkpoint.get("input_dim", -1)) != MODEL_WIDTHS[entry["branch"]]:
            raise ValueError(f"{entry['member']} checkpoint has wrong paper-v1 input width")
        normalization = checkpoint.get("embedding_normalization")
        if not isinstance(normalization, dict) or normalization.get("method") != "none":
            raise ValueError(f"{entry['member']} must use raw, unstandardized features")
        if checkpoint.get("feature_preprocessing") not in (None, "", {}):
            raise ValueError(f"{entry['member']} must not use feature preprocessing")
        if int(checkpoint.get("selected_epoch", -1)) != int(entry["selected_epoch"]):
            raise ValueError(f"{entry['member']} selected epoch differs from member table")
        model = build(MODEL_WIDTHS[entry["branch"]], entry["spec"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        entry["model"] = model
        members[entry["branch"]].append(entry)
        del checkpoint
    return members


def _load_paper_manifest_embedding(row: dict[str, str], family: str) -> np.ndarray:
    """Load exactly one required PLM layer plus the released 42-channel input."""
    expected_layer = "layer_24" if family == "prott5" else "layer_36"
    if family not in FAMILIES:
        raise ValueError(f"unknown embedding family: {family}")
    if row.get("feature_preprocessing", ""):
        raise ValueError("paper-v1 inference requires untransformed manifest features")
    recipe = row.get("feature_recipe", "")
    if recipe and recipe not in (FEATURE_RECIPE, "prepare_mlp_search_input.sequence_features"):
        raise ValueError(f"unsupported paper-v1 feature recipe: {recipe!r}")
    sources = json.loads(row.get("embedding_sources") or "[]")
    if len(sources) != 2:
        raise ValueError("paper-v1 manifest must have exactly PLM and sequence42 sources")
    plm_source, sequence_source = sources
    if plm_source.get("family") != family or plm_source.get("key") != expected_layer:
        raise ValueError(f"{family} manifest must explicitly select {expected_layer}")
    if sequence_source.get("family") != "derived_sequence" or sequence_source.get("key") != "embedding":
        raise ValueError("paper-v1 manifest must append exactly one sequence42 embedding")
    plm = _load_exact_embedding(plm_source["path"], expected_layer, family)
    if plm.shape[1] != PLM_WIDTHS[family]:
        raise ValueError(f"{family} layer input width must be {PLM_WIDTHS[family]}")
    sequence = _load_exact_embedding(sequence_source["path"], "embedding", "sequence42")
    if sequence.shape[1] != 42:
        raise ValueError("sequence42 input width must be 42")
    if sequence.shape[0] != plm.shape[0]:
        raise ValueError("PLM and sequence42 source lengths differ")
    return np.concatenate((plm, sequence), axis=1)


def _manifest_index(path: str | Path):
    rows = read_manifest(path)
    indexed = {}
    for row in rows:
        identity = str(row["id"])
        if identity in indexed:
            raise ValueError(f"duplicate ID {identity!r} in {path}")
        length = int(row.get("length") or 0)
        if length < 1:
            raise ValueError(f"invalid length for {identity!r} in {path}")
        indexed[identity] = (length, row)
    return rows, indexed


def _predict_one(member, embedding: np.ndarray, device: torch.device) -> np.ndarray:
    expected = MODEL_WIDTHS[member["branch"]]
    if embedding.ndim != 2 or embedding.shape[1] != expected:
        raise ValueError(f"{member['member']} got {embedding.shape}; expected (*, {expected})")
    if not np.isfinite(embedding).all():
        raise ValueError(f"nonfinite embedding for {member['member']}")
    x = torch.from_numpy(np.ascontiguousarray(embedding, dtype=np.float32)).unsqueeze(0).to(device)
    lengths = torch.tensor([len(embedding)], dtype=torch.long, device=device)
    with torch.inference_mode():
        scores = torch.sigmoid(member["model"](x, lengths)).reshape(-1).float().cpu().numpy()
    if len(scores) != len(embedding) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"invalid residue probabilities from {member['member']}")
    return scores


def predict_ensemble(prott5_manifest: str | Path, esmc_manifest: str | Path,
                     models_dir: str | Path, members_tsv: str | Path = PAPER_MEMBERS,
                     output: str | Path | None = None, device: str = "cpu",
                     threads: int = 2, limit_proteins: int | None = None) -> Path:
    """Run frozen paper-v1 ten members on paired, label-free manifests."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if threads < 1 or (limit_proteins is not None and limit_proteins < 1):
        raise ValueError("threads and optional limit must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if output is None or not str(output).endswith(".tsv.gz"):
        raise ValueError("output must be a new .tsv.gz path")
    torch.set_num_threads(threads)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.parent / "receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {receipt_path}")
    manifest_receipts = {
        "prott5": {"path": str(Path(prott5_manifest).resolve()), "sha256": _digest(Path(prott5_manifest))},
        "esmc": {"path": str(Path(esmc_manifest).resolve()), "sha256": _digest(Path(esmc_manifest))},
    }
    pt_rows, pt_index = _manifest_index(prott5_manifest)
    _, esmc_index = _manifest_index(esmc_manifest)
    if set(pt_index) != set(esmc_index):
        raise ValueError("ProtT5 and ESMC manifest IDs differ")
    for identity, (length, _) in pt_index.items():
        if esmc_index[identity][0] != length:
            raise ValueError(f"manifest length mismatch for {identity!r}")
    rows = pt_rows if limit_proteins is None else pt_rows[:limit_proteins]
    if not rows:
        raise ValueError("no proteins to predict")
    torch_device = torch.device(device)
    members = _load_members(models_dir, members_tsv, torch_device)
    fields = ["id", "position", "score", "ours_ensemble", "ours_prott5_mean", "ours_esmc_mean"]
    fields += [m["score_column"] for family in FAMILIES for m in members[family]]
    residue_count = 0
    with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for count, pt_row in enumerate(rows, start=1):
            identity, length = str(pt_row["id"]), int(pt_row["length"])
            family_scores = {}
            for family in FAMILIES:
                manifest_row = pt_row if family == "prott5" else esmc_index[identity][1]
                embedding = _load_paper_manifest_embedding(manifest_row, family)
                if embedding.shape[0] != length:
                    raise ValueError(f"embedding length mismatch for {identity!r}/{family}")
                family_scores[family] = np.stack(
                    [_predict_one(member, embedding, torch_device) for member in members[family]], axis=0
                ).astype(np.float64)
            pt_mean = family_scores["prott5"].mean(axis=0)
            esmc_mean = family_scores["esmc"].mean(axis=0)
            # Preserve the handoff expression: average paired branch probabilities, then members.
            ensemble = ((family_scores["prott5"] + family_scores["esmc"]) / 2).mean(axis=0)
            residue_count += length
            for position in range(length):
                record = {"id": identity, "position": position + 1, "score": float(ensemble[position]),
                          "ours_ensemble": float(ensemble[position]), "ours_prott5_mean": float(pt_mean[position]),
                          "ours_esmc_mean": float(esmc_mean[position])}
                for family in FAMILIES:
                    for member_index, member in enumerate(members[family]):
                        record[member["score_column"]] = float(family_scores[family][member_index, position])
                writer.writerow(record)
            print(f"{count}/{len(rows)} {identity} residues={length}", flush=True)
    member_rows = [member for family in FAMILIES for member in members[family]]
    identity_status = _checkpoint_identity_status(member_rows)
    model_id = (
        "paper-v1-prott5-esmc-bigru"
        if identity_status == "original"
        else "paper-v1-retrained-ten-member-bigru"
    )
    receipt = {
        "model_id": model_id,
        "checkpoint_identity_status": identity_status,
        "output": {"path": str(target.resolve()), "sha256": _digest(target)},
        "member_table": {"path": str(Path(members_tsv).resolve()), "sha256": _digest(Path(members_tsv))},
        "members": [
            {"member": member["member"], "score_column": member["score_column"],
             "branch": member["branch"], "seed": int(member["seed"]),
             "checkpoint_sha256": member["checkpoint_sha256"]}
            for member in member_rows
        ],
        "manifests": manifest_receipts,
        "runtime": {"device": device, "threads": threads, "python": sys.version,
                    "numpy": np.__version__, "torch": torch.__version__},
        "records": len(rows),
        "residues": residue_count,
        "averaging_protocol": (
            "For each matching seed, average ProtT5 and ESMC member probabilities; "
            "then take the arithmetic mean over five seeds (equal weight for all ten members)."
        ),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _read_embedding_index(path: str | Path, expected_family: str) -> dict[str, dict[str, str]]:
    path = Path(path).resolve()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or ())
        id_col = next((c for c in ("id", "accession", "sequence_id") if c in columns), None)
        path_col = next((c for c in ("path", "embedding", "file") if c in columns), None)
        if not id_col or not path_col or not {"key", "family"}.issubset(columns):
            raise ValueError(f"{path} needs id, path, key, and family columns")
        result = {}
        for line, row in enumerate(reader, start=2):
            identity = row[id_col].strip()
            family = row["family"].strip().lower()
            value = row[path_col].strip()
            key = row["key"].strip()
            if not identity or identity in result or not value or not key:
                raise ValueError(f"blank/duplicate identity or missing embedding fields at {path}:{line}")
            if family != expected_family:
                raise ValueError(f"{path}:{line} family {family!r} does not match {expected_family!r}")
            resolved = Path(value)
            if not resolved.is_absolute():
                resolved = (path.parent / resolved).resolve()
            source = {"path": str(resolved), "key": key, "family": family}
            for optional_hash in ("sha256", "sequence_sha256"):
                value_hash = row.get(optional_hash, "").strip().lower()
                if value_hash:
                    if len(value_hash) != 64 or any(c not in "0123456789abcdef" for c in value_hash):
                        raise ValueError(f"invalid {optional_hash} at {path}:{line}")
                    source[optional_hash] = value_hash
            result[identity] = source
    if not result:
        raise ValueError(f"no embedding rows in {path}")
    return result


def prepare_inputs(fasta: str | Path, prott5_index: str | Path, esmc_index: str | Path,
                   output_dir: str | Path) -> Path:
    """Prepare two label-free manifests from supplied PLM arrays and sequences."""
    records = read_fasta(fasta)
    sequences = {}
    for record in records:
        if record.id in sequences:
            raise ValueError(f"duplicate FASTA ID: {record.id}")
        sequences[record.id] = record.sequence
    pt = _read_embedding_index(prott5_index, "prott5")
    esmc = _read_embedding_index(esmc_index, "esmc")
    if set(sequences) != set(pt) or set(sequences) != set(esmc):
        raise ValueError("FASTA and both embedding indices must contain identical IDs")
    digest_cache: dict[str, str] = {}
    metadata_cache: dict[str, str | None] = {}
    provenance = {}
    for identity, sequence in sequences.items():
        length = len(sequence)
        sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        sequence_features(sequence)  # Validate now; derive again during streaming output.
        protein_sources = {}
        for family, index, width in (("prott5", pt, PLM_WIDTHS["prott5"]), ("esmc", esmc, PLM_WIDTHS["esmc"])):
            source = index[identity]
            if source.get("sequence_sha256") and source["sequence_sha256"] != sequence_hash:
                raise ValueError(f"{identity}/{family} sequence SHA256 does not match FASTA")
            file_hash = _cached_digest(source["path"], digest_cache)
            if source.get("sha256") and source["sha256"] != file_hash:
                raise ValueError(f"{identity}/{family} embedding file SHA256 mismatch")
            values = _load_exact_embedding(source["path"], source["key"], family)
            if values.shape != (length, width):
                raise ValueError(f"{identity}/{family} shape {values.shape}; expected {(length, width)}")
            resolved_path = str(Path(source["path"]).resolve())
            if resolved_path not in metadata_cache:
                metadata_cache[resolved_path] = _native_metadata_literal(resolved_path)
            protein_sources[family] = {
                "path": resolved_path, "key": source["key"], "width": int(values.shape[1]),
                "file_sha256": file_hash,
                "native_metadata_literal": metadata_cache[resolved_path],
            }
        provenance[identity] = {
            "sequence_sha256": sequence_hash,
            "length": length,
            "plm_sources": protein_sources,
        }
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to reuse existing output directory: {root}")
    root.mkdir(parents=True)
    feature_dir = root / "sequence42"
    feature_dir.mkdir()
    fields = ("id", "length", "sequence_key", "embedding", "embedding_key", "embedding_sources",
              "feature_preprocessing", "feature_recipe")
    sequence_feature_hashes = {}
    for family, index in (("prott5", pt), ("esmc", esmc)):
        with (root / f"{family}.tsv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for identity, sequence in sequences.items():
                feature_path = feature_dir / (hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".npz")
                if family == "prott5":
                    with feature_path.open("xb") as array_handle:
                        np.savez_compressed(array_handle, embedding=sequence_features(sequence))
                    sequence_feature_hashes[identity] = _digest(feature_path)
                sources = [
                    index[identity],
                    {"path": str(feature_path.resolve()), "key": "embedding", "family": "derived_sequence"},
                ]
                writer.writerow({"id": identity, "length": len(sequence), "sequence_key": identity,
                                 "embedding": "", "embedding_key": "", "embedding_sources": json.dumps(sources, separators=(",", ":")),
                                 "feature_preprocessing": "", "feature_recipe": FEATURE_RECIPE})
    preparation = {
        "model_id": "paper-v1-study-bigru-5x2",
        "inputs": {
            "fasta": {"path": str(Path(fasta).resolve()), "sha256": _digest(Path(fasta))},
            "prott5_index": {"path": str(Path(prott5_index).resolve()), "sha256": _digest(Path(prott5_index))},
            "esmc_index": {"path": str(Path(esmc_index).resolve()), "sha256": _digest(Path(esmc_index))},
        },
        "sequence_feature_recipe": FEATURE_RECIPE,
        "records": len(sequences),
        "proteins": {
            identity: {**provenance[identity], "sequence42_file_sha256": sequence_feature_hashes[identity]}
            for identity in sequences
        },
    }
    (root / "preparation.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _prepare_parser():
    parser = argparse.ArgumentParser(description="Prepare paper-v1 manifests from supplied sequence and PLM data.")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--prott5-index", required=True, help="TSV: id, path, key, family=prott5")
    parser.add_argument("--esmc-index", required=True, help="TSV: id, path, key, family=esmc")
    parser.add_argument("--output-dir", required=True, help="Must be a new directory")
    return parser


def prepare_main(argv=None):
    args = _prepare_parser().parse_args(argv)
    result = prepare_inputs(args.fasta, args.prott5_index, args.esmc_index, args.output_dir)
    print(f"INPUTS_READY {result}", flush=True)


def _inference_parser():
    parser = argparse.ArgumentParser(description="Run the frozen paper-v1 ten-member ensemble.")
    parser.add_argument("--prott5-manifest", required=True)
    parser.add_argument("--esmc-manifest", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--members", "--members-tsv", dest="members_tsv", default=str(PAPER_MEMBERS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-proteins", type=int, default=None)
    return parser


def predict_main(argv=None):
    args = _inference_parser().parse_args(argv)
    result = predict_ensemble(args.prott5_manifest, args.esmc_manifest, args.models_dir,
                              args.members_tsv, args.output, args.device, args.threads, args.limit_proteins)
    print(f"PREDICTIONS_READY {result}", flush=True)


def verify_models_main(argv=None):
    parser = argparse.ArgumentParser(description="Verify paper-v1 checkpoint hashes and summary metadata (no weight deserialization).")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--members", "--members-tsv", dest="members_tsv", default=str(PAPER_MEMBERS))
    args = parser.parse_args(argv)
    for row in verify_models(args.models_dir, args.members_tsv):
        print(f"HASH_VERIFIED {row['member']} {row['checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    predict_main()
