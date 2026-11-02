"""Portable replay and export commands for paper-v1 external baseline metrics.

This module uses only the Python standard library. Export is the only entrypoint
that accepts private source roots; replay reads only the normalized artifact.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paper import config_directory
from .paper_metrics import evaluate

DEFAULT_CONFIG = config_directory() / "paper-v1-evaluation.json"
FIELDS = [
    "model", "dataset", "proteins", "valid_residues", "positive_residues",
    "threshold_policy", "threshold", "average_precision", "auprc_trapezoid",
    "auroc", "mcc", "f1", "precision", "recall", "specificity", "accuracy",
    "tp", "tn", "fp", "fn", "brier_score", "expected_calibration_error",
    "precision_at_5", "precision_at_10", "precision_at_20", "precision_at_50",
    "precision_at_100", "precision_at_top_1pct", "precision_at_top_5pct",
    "precision_at_top_10pct", "source_file",
]
NUMERIC = [x for x in FIELDS if x not in {"model", "dataset", "threshold_policy", "source_file"}]
RANK_FIELDS = {
    "precision_at_5": "5", "precision_at_10": "10", "precision_at_20": "20",
    "precision_at_50": "50", "precision_at_100": "100",
    "precision_at_top_1pct": "top_1pct", "precision_at_top_5pct": "top_5pct",
    "precision_at_top_10pct": "top_10pct",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict[str, Any]:
    path = root / "MANIFEST.json"
    manifest = load_json(path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError(f"Invalid artifact manifest: {path}")
    listed = {}
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"Artifact contains a symbolic link: {candidate}")
    for item in manifest["files"]:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts or item["path"] == "MANIFEST.json":
            raise ValueError(f"Unsafe or recursive manifest path: {item['path']!r}")
        target = root / rel
        if not target.is_file():
            raise FileNotFoundError(f"Manifest artifact is missing: {target}")
        if target.stat().st_size != int(item["bytes"]) or sha256(target) != item["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {target}")
        if item["path"] in listed:
            raise ValueError(f"Duplicate artifact manifest path: {item['path']}")
        listed[item["path"]] = True
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "MANIFEST.json"
    }
    if actual != set(listed):
        raise ValueError(f"Artifact manifest file set differs: unlisted={sorted(actual-set(listed))[:5]}")
    return manifest


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required paper-v1 resource is missing: {path}") from exc


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported evaluation config schema: {config.get('schema_version')!r}")
    if len(config.get("datasets", [])) != 6 or len(config.get("predictors", [])) != 14:
        raise ValueError("paper-v1 config must define six datasets and fourteen distinct predictors")
    ids = [item["id"] for item in config["predictors"]]
    if len(ids) != len(set(ids)):
        raise ValueError("paper-v1 predictor IDs must be unique")
    return config


def native_plan_records(path: Path) -> list[dict[str, Any]]:
    plan = load_json(path)
    sequences = {item["token"]: item["sequence"] for item in plan.get("sequences", [])}
    records = []
    for item in plan.get("records", []):
        token = item["token"]
        if token not in sequences:
            raise ValueError(f"Native plan record references unknown token {token!r}")
        record = {
            "dataset": item["dataset"],
            "accession": str(item.get("record_id", item.get("accession", ""))),
            "record_key": item["record_key"],
            "sequence": sequences[token],
            "labels": str(item["labels"]),
            "valid": [bool(value) for value in item["valid"]],
            "exact_development_train": bool(item.get("exact_development_train", False)),
            "exact_development_val": bool(item.get("exact_development_val", False)),
            "exact_development_test": bool(item.get("exact_development_test", False)),
        }
        if not record["accession"] or not (len(record["sequence"]) == len(record["labels"]) == len(record["valid"])):
            raise ValueError(f"Native plan sequence/label/mask mismatch for {record['accession']!r}")
        records.append(record)
    return records


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        return native_plan_records(path)
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"dataset", "accession", "record_key", "sequence", "labels", "valid"}
            if not required <= set(record):
                raise ValueError(f"{path}:{line_number}: missing record fields {required - set(record)}")
            if not (len(record["sequence"]) == len(record["labels"]) == len(record["valid"])):
                raise ValueError(f"{path}:{line_number}: sequence/label/mask length mismatch")
            records.append(record)
    return records


def index_records(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    seen_keys = set()
    for record in records:
        key = (str(record["dataset"]), str(record["accession"]))
        record_key = (key[0], str(record["record_key"]))
        labels = str(record["labels"])
        invalid = sorted(set(labels) - set("01-#2"))
        if invalid:
            raise ValueError(f"Invalid native label character(s) {invalid!r} for {key!r}")
        if len(labels) != len(record["sequence"]) or len(record["valid"]) != len(labels):
            raise ValueError(f"Sequence/label/mask length mismatch for {key!r}")
        if key in indexed or record_key in seen_keys:
            raise ValueError(f"Duplicate plan record or record_key: {key!r}")
        indexed[key] = record
        seen_keys.add(record_key)
    if not indexed:
        raise ValueError("No frozen records were supplied")
    return indexed


def valid_label(record: dict[str, Any], offset: int) -> int | None:
    label = str(record["labels"][offset])
    if label not in "01-#2":
        raise ValueError(f"Invalid native label character {label!r}")
    if not record["valid"][offset] or label not in "01":
        return None
    return int(label)


def read_score_pairs(
    path: Path, indexed: dict[tuple[str, str], dict[str, Any]], *,
    datasets: set[str] | None = None, score_column: str = "score",
) -> tuple[dict[str, list[tuple[float, int]]], dict[str, dict[str, int]]]:
    selected = {key: rec for key, rec in indexed.items() if datasets is None or key[0] in datasets}
    seen = {key: bytearray(len(rec["sequence"])) for key, rec in selected.items()}
    pairs: dict[str, list[tuple[float, int]]] = {}
    full: dict[str, int] = {}
    valid: dict[str, int] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"dataset", "accession", "position", "residue", score_column}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{path}: missing columns {required - set(reader.fieldnames or [])}")
        for line, row in enumerate(reader, 2):
            key = (row["dataset"], row["accession"])
            if key not in selected:
                raise ValueError(f"{path}:{line}: unknown/unselected record {key!r}")
            rec = selected[key]
            try:
                pos = int(row["position"])
                score = float(row[score_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid position or score") from exc
            offset = pos - 1
            if not 0 <= offset < len(rec["sequence"]):
                raise ValueError(f"{path}:{line}: out-of-range coordinate {key!r}:{pos}")
            if seen[key][offset]:
                raise ValueError(f"{path}:{line}: duplicate coordinate {key!r}:{pos}")
            if row["residue"] != rec["sequence"][offset]:
                raise ValueError(f"{path}:{line}: residue mismatch at {key!r}:{pos}")
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line}: non-finite score at {key!r}:{pos}")
            seen[key][offset] = 1
            full[key[0]] = full.get(key[0], 0) + 1
            label = valid_label(rec, offset)
            if label is not None:
                pairs.setdefault(key[0], []).append((score, label))
                valid[key[0]] = valid.get(key[0], 0) + 1
    missing = [key for key, marks in seen.items() if not all(marks)]
    if missing:
        raise ValueError(f"{path}: incomplete full-sequence coordinate coverage; examples: {missing[:5]}")
    coverage = {}
    for dataset in sorted({key[0] for key in selected}):
        recs = [rec for key, rec in selected.items() if key[0] == dataset]
        coverage[dataset] = {
            "proteins": len(recs),
            "full_residues": sum(len(rec["sequence"]) for rec in recs),
            "scored_residues": full.get(dataset, 0),
            "valid_residues": valid.get(dataset, 0),
            "positive_residues": sum(label for _, label in pairs.get(dataset, [])),
        }
    return pairs, coverage


def metric_row(
    predictor: dict[str, Any], dataset: dict[str, Any], threshold: float,
    policy: str, pairs: list[tuple[float, int]], source: str, probabilities: bool,
) -> dict[str, str]:
    metrics, ranks = evaluate(pairs, threshold, probabilities=probabilities)
    counts = metrics["counts"]
    row = {
        "model": predictor["name"], "dataset": dataset["name"],
        "proteins": str(dataset["proteins"]),
        "valid_residues": str(metrics["valid_residues"]),
        "positive_residues": str(metrics["positive_residues"]),
        "threshold_policy": policy, "threshold": repr(float(threshold)),
        "average_precision": repr(float(metrics["average_precision"])),
        "auprc_trapezoid": repr(float(metrics["auprc_trapezoid"])),
        "auroc": repr(float(metrics["auroc"])), "mcc": repr(float(metrics["mcc"])),
        "f1": repr(float(metrics["f1"])), "precision": repr(float(metrics["precision"])),
        "recall": repr(float(metrics["recall"])), "specificity": repr(float(metrics["specificity"])),
        "accuracy": repr(float(metrics["accuracy"])), "tp": str(counts["tp"]),
        "tn": str(counts["tn"]), "fp": str(counts["fp"]), "fn": str(counts["fn"]),
        "brier_score": "" if metrics["brier_score"] is None else repr(float(metrics["brier_score"])),
        "expected_calibration_error": "" if metrics["expected_calibration_error"] is None else repr(float(metrics["expected_calibration_error"])),
        "source_file": source,
    }
    for field, rank_key in RANK_FIELDS.items():
        value = ranks[rank_key]["precision"]
        row[field] = "" if value is None else repr(float(value))
    return row


def compare_reference(rows: list[dict[str, str]], reference: Path, tolerance: float) -> dict[str, Any]:
    expected = {}
    with reference.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected reference columns in {reference}")
        for row in reader:
            key = (row["model"], row["dataset"], row["threshold_policy"])
            if key in expected:
                raise ValueError(f"Duplicate immutable reference row {key!r}")
            expected[key] = row
    actual = {(r["model"], r["dataset"], r["threshold_policy"]): r for r in rows}
    if set(actual) != set(expected):
        raise ValueError(f"Reference row keys differ: missing={sorted(set(expected)-set(actual))[:5]}, extra={sorted(set(actual)-set(expected))[:5]}")
    checked = 0
    for key, ref in expected.items():
        for field in NUMERIC:
            if ref[field] == "":
                continue
            checked += 1
            got = actual[key][field]
            if got == "" or abs(float(got) - float(ref[field])) > tolerance:
                raise ValueError(f"Immutable reference mismatch at {key}/{field}: expected={ref[field]!r}, actual={got!r}")
    return {"status": "PASS", "rows": len(rows), "numeric_fields_checked": checked, "tolerance": tolerance}


def resolve_reference(config_path: Path, config: dict[str, Any], artifact_root: Path) -> Path:
    rel = Path(config["reference"]["path"])
    for candidate in (artifact_root / rel, config_path.parent / rel):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Immutable reference table missing: {artifact_root / rel}")


def replay_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute all paper-v1 metrics from portable scores and labels.")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    config_path = args.config or args.artifact_root / "paper-v1-evaluation.json"
    if not config_path.is_file():
        config_path = DEFAULT_CONFIG
    config = load_config(config_path)
    verify_manifest(args.artifact_root)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    records_path = args.artifact_root / config["artifact"]["records"]
    indexed = index_records(read_records(records_path))
    proofs = load_json(args.artifact_root / config["artifact"]["thresholds"])
    specs = {x["id"]: x for x in config["datasets"]}
    if {key[0] for key in indexed} != set(specs):
        raise ValueError("Frozen plan dataset IDs differ from config")
    for dataset_id, spec in specs.items():
        records = [rec for key, rec in indexed.items() if key[0] == dataset_id]
        counts = {
            "proteins": len(records),
            "full_residues": sum(len(rec["sequence"]) for rec in records),
            "valid_residues": sum(1 for rec in records for i in range(len(rec["sequence"])) if valid_label(rec, i) is not None),
            "positive_residues": sum(valid_label(rec, i) or 0 for rec in records for i in range(len(rec["sequence"]))),
        }
        if counts != {field: spec[field] for field in counts}:
            raise ValueError(f"Plan cohort mismatch for {dataset_id}: {counts}")
    output_rows = []
    score_hashes = {}
    for predictor in config["predictors"]:
        pid = predictor["id"]
        score_path = args.artifact_root / config["artifact"]["scores_directory"] / f"{pid}.tsv.gz"
        if not score_path.is_file():
            raise FileNotFoundError(f"Required continuous score artifact is missing: {score_path}")
        score_hashes[pid] = sha256(score_path)
        pairs, coverage = read_score_pairs(score_path, indexed)
        proof = proofs.get("predictors", {}).get(pid)
        if not isinstance(proof, dict) or proof.get("source_threshold") is None:
            raise ValueError(f"Missing frozen validation threshold proof for {pid}")
        threshold = float(proof["source_threshold"])
        for dataset in config["datasets"]:
            did = dataset["id"]
            expected = {
                "proteins": dataset["proteins"], "full_residues": dataset["full_residues"],
                "scored_residues": dataset["full_residues"],
                "valid_residues": dataset["valid_residues"],
                "positive_residues": dataset["positive_residues"],
            }
            if coverage[did] != expected:
                raise ValueError(f"{pid}/{did} coverage mismatch: {coverage[did]} != {expected}")
            output_rows.append(metric_row(
                predictor, dataset, threshold, predictor["validation_policy"],
                pairs[did], f"scores/{pid}.tsv.gz", bool(predictor["probabilities"]),
            ))
            if predictor.get("official_threshold") is not None:
                output_rows.append(metric_row(
                    predictor, dataset, float(predictor["official_threshold"]),
                    predictor["official_policy"], pairs[did],
                    f"scores/{pid}.tsv.gz", bool(predictor["probabilities"]),
                ))
    if len(output_rows) != 90:
        raise ValueError(f"Expected 90 operating rows; generated {len(output_rows)}")
    reference = resolve_reference(config_path, config, args.artifact_root)
    comparison = compare_reference(output_rows, reference, float(config["reference"]["tolerance"]))
    args.output_dir.mkdir(parents=True)
    table = args.output_dir / "paper-v1-replayed.tsv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(output_rows)
    result = {
        "status": "complete", "protocol": config["protocol"], "datasets": config["datasets"],
        "predictors": [x["id"] for x in config["predictors"]], "operating_rows": len(output_rows),
        "input_sha256": {
            "records": sha256(records_path), "thresholds": sha256(args.artifact_root / config["artifact"]["thresholds"]),
            "scores": score_hashes, "reference": sha256(reference),
        },
        "reference_comparison": comparison, "rows": output_rows,
    }
    (args.output_dir / "paper-v1-replayed.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PAPER_REPLAY_PASS {len(output_rows)} rows; {comparison['numeric_fields_checked']} reference numeric fields")
    return 0


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate one portable continuous-score TSV against frozen labels.")
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path, help="Normalized records JSONL(.gz) or native frozen-plan JSON")
    parser.add_argument("--threshold", required=True, type=float, help="Frozen external threshold selected independently")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--probabilities", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    indexed = index_records(read_records(args.labels))
    selected = set(args.dataset) if args.dataset else None
    pairs, coverage = read_score_pairs(args.scores, indexed, datasets=selected)
    result = {
        "status": "complete", "threshold": args.threshold,
        "threshold_policy": "frozen external threshold; score >= threshold",
        "scores_sha256": sha256(args.scores), "labels_sha256": sha256(args.labels), "datasets": {},
    }
    for did, values in sorted(pairs.items()):
        metrics, ranks = evaluate(values, args.threshold, probabilities=args.probabilities)
        result["datasets"][did] = {"coverage": coverage[did], "metrics": metrics, "precision_at_k": ranks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PAPER_EVALUATE_PASS {len(result['datasets'])} datasets")
    return 0


def require_file(path: Path, purpose: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {purpose}: {path}; no merged-metric fallback or inference is performed")
    return path


def write_gzip_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                for rec in records:
                    stream.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=True) + "\n")


def export_scores(path: Path, predictor: dict[str, Any], records: dict[tuple[str, str], dict[str, Any]], sources: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = {key: bytearray(len(rec["sequence"])) for key, rec in records.items()}
    needed = {"dataset", "accession", "position", "residue", predictor["score_column"]}
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as output:
                writer = csv.writer(output, delimiter="\t", lineterminator="\n")
                writer.writerow(["dataset", "accession", "position", "residue", "score"])
                for did in sorted({key[0] for key in records}):
                    source = sources[did]
                    opener = gzip.open if source.suffix == ".gz" else open
                    with opener(source, "rt", encoding="utf-8", newline="") as stream:
                        reader = csv.DictReader(stream, delimiter="\t")
                        if reader.fieldnames is None or not needed <= set(reader.fieldnames):
                            raise ValueError(f"{source}: missing columns {needed - set(reader.fieldnames or [])}")
                        for line, row in enumerate(reader, 2):
                            if row["dataset"] != did:
                                raise ValueError(f"{source}:{line}: unexpected dataset {row['dataset']!r}")
                            key = (did, row["accession"])
                            if key not in records:
                                raise ValueError(f"{source}:{line}: unknown plan record {key!r}")
                            rec = records[key]
                            pos = int(row["position"])
                            offset = pos - 1
                            if not 0 <= offset < len(rec["sequence"]):
                                raise ValueError(f"{source}:{line}: out-of-range coordinate {key!r}:{pos}")
                            if seen[key][offset]:
                                raise ValueError(f"{source}:{line}: duplicate coordinate {key!r}:{pos}")
                            if row["residue"] != rec["sequence"][offset]:
                                raise ValueError(f"{source}:{line}: residue mismatch at {key!r}:{pos}")
                            score = float(row[predictor["score_column"]])
                            if not math.isfinite(score):
                                raise ValueError(f"{source}:{line}: invalid score at {key!r}:{pos}")
                            seen[key][offset] = 1
                            writer.writerow([did, row["accession"], pos, row["residue"], repr(score)])
    missing = [key for key, marks in seen.items() if not all(marks)]
    if missing:
        path.unlink(missing_ok=True)
        raise ValueError(f"{predictor['id']} lacks full-sequence coverage; examples: {missing[:5]}")


def receipt_proof(baseline_root: Path, predictor: dict[str, Any]) -> dict[str, Any]:
    path = require_file(baseline_root / predictor["source_run"] / predictor["receipt"], f"{predictor['id']} completion receipt")
    receipt = load_json(path)
    if receipt.get("status") != "complete" or receipt.get("validation_threshold") is None:
        raise ValueError(f"{predictor['id']} receipt is incomplete or has no validation threshold: {path}")
    evidence_hash = receipt.get("threshold_evidence_sha256")
    if not isinstance(evidence_hash, str) or len(evidence_hash) != 64 or any(char not in "0123456789abcdef" for char in evidence_hash.lower()):
        raise ValueError(f"{predictor['id']} receipt lacks a valid threshold evidence SHA256: {path}")
    return {
        "source_threshold": float(receipt["validation_threshold"]),
        "proof_kind": "completed evaluation receipt with frozen validation-MCC threshold",
        "receipt_sha256": sha256(path), "proof_sha256": evidence_hash,
        "proof_record": f"{predictor['source_run']}/{predictor['receipt']}",
        "evaluation_input_plan_sha256": receipt.get("input_plan_sha256"),
    }


def handoff_proof(handoff_root: Path, predictor_id: str) -> dict[str, Any]:
    path = require_file(handoff_root / "analysis" / "validation-thresholds.tsv", "handoff validation threshold table")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["predictor", "threshold", "validation_MCC", "valid_residues"]:
            raise ValueError(f"Unexpected threshold evidence columns in {path}")
        matches = [row for row in reader if row["predictor"] == predictor_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one {predictor_id!r} threshold record, found {len(matches)}")
    return {
        "source_threshold": float(matches[0]["threshold"]),
        "proof_kind": "frozen V2 validation-MCC threshold from handoff validation-thresholds.tsv",
        "proof_sha256": sha256(path), "proof_record": "analysis/validation-thresholds.tsv",
    }


def manifest(root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    result = {"schema_version": 1, "file_count": len(entries), "files": entries}
    (root / "MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def export_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export source roots to a checksummed portable paper-v1 artifact.")
    parser.add_argument("--own-root", required=True, type=Path, help="Contains inputs/frozen-plan.json and outputs/")
    parser.add_argument("--baseline-root", required=True, type=Path, help="Contains evaluation/<run>/ receipts and predictions")
    parser.add_argument("--punch2-root", required=True, type=Path, help="Contains outputs/<dataset>/prediction.tsv.gz")
    parser.add_argument("--handoff-root", required=True, type=Path, help="Contains analysis/validation-thresholds.tsv")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.artifact_root.exists():
        raise FileExistsError(f"Refusing to overwrite artifact directory: {args.artifact_root}")
    plan_path = require_file(args.own_root / "inputs" / "frozen-plan.json", "frozen six-set plan")
    plan = load_json(plan_path)
    if plan.get("datasets") != [item["id"] for item in config["datasets"]]:
        raise ValueError("Frozen source plan dataset IDs/order differ from paper-v1 config")
    sequences = {item["token"]: item["sequence"] for item in plan.get("sequences", [])}
    records_list = []
    records = {}
    for item in plan.get("records", []):
        token = item["token"]
        if token not in sequences:
            raise ValueError(f"Frozen plan references unknown sequence token {token!r}")
        rec = {
            "dataset": item["dataset"],
            "accession": str(item.get("record_id", item.get("accession", ""))),
            "record_key": item["record_key"], "sequence": sequences[token],
            "labels": str(item["labels"]), "valid": [bool(x) for x in item["valid"]],
            "exact_development_train": bool(item.get("exact_development_train", False)),
            "exact_development_val": bool(item.get("exact_development_val", False)),
            "exact_development_test": bool(item.get("exact_development_test", False)),
        }
        if not rec["accession"] or not (len(rec["sequence"]) == len(rec["labels"]) == len(rec["valid"])):
            raise ValueError(f"Frozen sequence/label/mask mismatch for {rec['accession']!r}")
        key = (rec["dataset"], rec["accession"])
        if key in records:
            raise ValueError(f"Duplicate frozen plan record {key!r}")
        records[key] = rec
        records_list.append(rec)
    for dataset in config["datasets"]:
        recs = [rec for key, rec in records.items() if key[0] == dataset["id"]]
        observed = {
            "proteins": len(recs), "full_residues": sum(len(r["sequence"]) for r in recs),
            "valid_residues": sum(1 for r in recs for i in range(len(r["sequence"])) if valid_label(r, i) is not None),
            "positive_residues": sum(valid_label(r, i) or 0 for r in recs for i in range(len(r["sequence"]))),
        }
        expected = {key: dataset[key] for key in observed}
        if observed != expected:
            raise ValueError(f"Frozen plan cohort mismatch for {dataset['id']}: {observed} != {expected}")

    plan_sha = sha256(plan_path)
    sources: dict[str, dict[str, Path]] = {}
    proofs: dict[str, dict[str, Any]] = {}
    source_roots = {"own": args.own_root, "baseline": args.baseline_root, "punch2": args.punch2_root}
    for predictor in config["predictors"]:
        pid = predictor["id"]
        root = source_roots[predictor["source_root"]]
        sources[pid] = {
            dataset["id"]: require_file(
                root / predictor["source_run"] / dataset["id"] / "prediction.tsv.gz",
                f"{pid} continuous predictions for {dataset['id']}",
            )
            for dataset in config["datasets"]
        }
        threshold_source = predictor.get("threshold_source")
        if threshold_source == "own_handoff_validation_thresholds":
            proofs[pid] = handoff_proof(args.handoff_root, "ours_ensemble")
            if abs(proofs[pid]["source_threshold"] - float(plan["threshold"])) > 1e-12:
                raise ValueError("Own threshold in frozen external plan disagrees with validation evidence")
        elif threshold_source == "handoff_validation_thresholds":
            proofs[pid] = handoff_proof(args.handoff_root, pid)
        else:
            proof = receipt_proof(args.baseline_root, predictor)
            if proof.get("evaluation_input_plan_sha256") != plan_sha:
                raise ValueError(f"{pid} completion receipt references a different frozen input plan")
            receipt = load_json(args.baseline_root / predictor["source_run"] / predictor["receipt"])
            if bool(receipt.get("probabilities")) != bool(predictor["probabilities"]):
                raise ValueError(f"{pid} probability declaration disagrees with completion receipt")
            proofs[pid] = proof
    args.artifact_root.mkdir(parents=True)
    write_gzip_jsonl(args.artifact_root / config["artifact"]["records"], records_list)
    for predictor in config["predictors"]:
        pid = predictor["id"]
        export_scores(
            args.artifact_root / config["artifact"]["scores_directory"] / f"{pid}.tsv.gz",
            predictor, records, sources[pid],
        )
    (args.artifact_root / config["artifact"]["thresholds"]).write_text(json.dumps({
        "schema_version": 1, "source_plan_sha256": plan_sha,
        "protocol": config["protocol"]["threshold_selection"], "predictors": proofs,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copyfile(args.config, args.artifact_root / "paper-v1-evaluation.json")
    reference = args.reference or args.config.parent / config["reference"]["path"]
    require_file(reference, "immutable expected reference comparison")
    shutil.copyfile(reference, args.artifact_root / Path(config["reference"]["path"]).name)
    manifest_result = manifest(args.artifact_root)
    print(f"PAPER_EXPORT_PASS {len(records_list)} records; 14 score files; {manifest_result['file_count']} hashed files")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "replay", "export"))
    args, remainder = parser.parse_known_args(argv)
    if args.command == "evaluate":
        return evaluate_main(remainder)
    if args.command == "replay":
        return replay_main(remainder)
    return export_main(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
