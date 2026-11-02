import csv
import hashlib
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from my_predictor.paper_replay import (FIELDS, evaluate_main, index_records, read_score_pairs, replay_main, valid_label,)


class PaperReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels = self.root / "records.jsonl.gz"
        with gzip.open(self.labels, "wt", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps({
                "dataset": "tiny", "accession": "P1", "record_key": "record-1",
                "sequence": "ACGT", "labels": "1010",
                "valid": [True, True, False, True],
            }) + "\n")
        self.scores = self.root / "scores.tsv.gz"
        with gzip.open(self.scores, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["dataset", "accession", "position", "residue", "score"])
            for position, residue, score in [
                (1, "A", 0.9), (2, "C", 0.8), (3, "G", 0.7), (4, "T", 0.6)
            ]:
                writer.writerow(["tiny", "P1", position, residue, score])

    def tearDown(self):
        self.temp.cleanup()

    def test_evaluate_cli_uses_mask_and_frozen_threshold(self):
        output = self.root / "metrics.json"
        evaluate_main([
            "--scores", str(self.scores), "--labels", str(self.labels),
            "--threshold", "0.75", "--dataset", "tiny", "--output", str(output),
        ])
        report = json.loads(output.read_text())
        result = report["datasets"]["tiny"]["metrics"]
        self.assertEqual(result["valid_residues"], 3)
        self.assertEqual(result["positive_residues"], 1)
        self.assertEqual(result["threshold"], 0.75)
        self.assertEqual(result["counts"], {"tp": 1, "tn": 1, "fp": 1, "fn": 0})

    def test_coordinate_residue_mismatch_fails(self):
        records = index_records([{
            "dataset": "tiny", "accession": "P1", "record_key": "record-1",
            "sequence": "ACGT", "labels": "1010", "valid": [True, True, False, True],
        }])
        bad = self.root / "bad.tsv"
        bad.write_text(
            "dataset\taccession\tposition\tresidue\tscore\n"
            "tiny\tP1\t1\tX\t0.9\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "residue mismatch"):
            read_score_pairs(bad, records)

    def test_missing_coordinate_fails_full_sequence_coverage(self):
        records = index_records([{
            "dataset": "tiny", "accession": "P1", "record_key": "record-1",
            "sequence": "ACGT", "labels": "1010", "valid": [True, True, False, True],
        }])
        short = self.root / "short.tsv"
        short.write_text(
            "dataset\taccession\tposition\tresidue\tscore\n"
            "tiny\tP1\t1\tA\t0.9\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "incomplete full-sequence"):
            read_score_pairs(short, records)

    def test_complete_six_dataset_fourteen_predictor_replay(self):
        import hashlib
        from my_predictor.paper_replay import FIELDS, replay_main

        artifact = self.root / "artifact"
        (artifact / "scores").mkdir(parents=True)
        dids = [f"d{i}" for i in range(6)]
        mids = [f"m{i}" for i in range(14)]
        datasets = [{"id": d, "name": f"D{i}", "proteins": 1, "full_residues": 4,
                     "valid_residues": 4, "positive_residues": 2} for i, d in enumerate(dids)]
        predictors = []
        for i, mid in enumerate(mids):
            punch = i == 9
            predictors.append({"id": mid, "name": "PUNCH2" if punch else f"Method {i}",
                "source_root": "baseline", "source_run": "fixture", "score_column": "score",
                "probabilities": i not in (0, 11), "validation_policy": "validation threshold",
                **({"official_threshold": 0.39, "official_policy": "official 0.39"} if punch else {})})
        config = {"schema_version": 1,
            "artifact": {"records": "records.jsonl.gz", "thresholds": "thresholds.json", "scores_directory": "scores"},
            "reference": {"path": "paper-v1-reference.tsv", "tolerance": 1e-9},
            "protocol": {"score_comparison": "score >= threshold"}, "datasets": datasets, "predictors": predictors}
        (artifact / "paper-v1-evaluation.json").write_text(json.dumps(config), encoding="utf-8")
        with gzip.open(artifact / "records.jsonl.gz", "wt", encoding="utf-8", newline="") as stream:
            for did in dids:
                stream.write(json.dumps({"dataset": did, "accession": "P1", "record_key": f"{did}:P1",
                    "sequence": "ACGT", "labels": "1010", "valid": [True] * 4}) + "\n")
        (artifact / "thresholds.json").write_text(json.dumps({"predictors": {
            mid: {"source_threshold": 0.5, "source": "fixture validation"} for mid in mids}}), encoding="utf-8")
        for mid in mids:
            with gzip.open(artifact / "scores" / f"{mid}.tsv.gz", "wt", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(["dataset", "accession", "position", "residue", "score"])
                for did in dids:
                    for pos, (res, score) in enumerate(zip("ACGT", (0.9, 0.8, 0.2, 0.1)), 1):
                        writer.writerow([did, "P1", pos, res, score])
        with (artifact / "paper-v1-reference.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for predictor in predictors:
                policies = [(predictor["validation_policy"], 0.5)]
                if predictor.get("official_threshold") is not None:
                    policies.append((predictor["official_policy"], 0.39))
                for dataset in datasets:
                    for policy, threshold in policies:
                        row = {field: "" for field in FIELDS}
                        row.update({"model": predictor["name"], "dataset": dataset["name"], "proteins": "1",
                            "valid_residues": "4", "positive_residues": "2", "threshold_policy": policy,
                            "threshold": str(threshold), "average_precision": repr(5 / 6),
                            "auprc_trapezoid": repr(19 / 24), "auroc": "0.75", "mcc": "0.0", "f1": "0.5",
                            "precision": "0.5", "recall": "0.5", "specificity": "0.5", "accuracy": "0.5",
                            "tp": "1", "tn": "1", "fp": "1", "fn": "1",
                            "brier_score": "" if not predictor["probabilities"] else "0.325",
                            "expected_calibration_error": "" if not predictor["probabilities"] else "0.45",
                            "precision_at_top_1pct": "1.0", "precision_at_top_5pct": "1.0",
                            "precision_at_top_10pct": "1.0", "source_file": f"scores/{predictor['id']}.tsv.gz"})
                        writer.writerow(row)
        files = []
        for path in sorted(p for p in artifact.rglob("*") if p.is_file()):
            data = path.read_bytes()
            files.append({"path": path.relative_to(artifact).as_posix(), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
        (artifact / "MANIFEST.json").write_text(json.dumps({"schema_version": 1, "files": files}), encoding="utf-8")
        output = self.root / "replayed"
        self.assertEqual(replay_main(["--artifact-root", str(artifact), "--output-dir", str(output)]), 0)
        report = json.loads((output / "paper-v1-replayed.json").read_text(encoding="utf-8"))
        self.assertEqual(report["operating_rows"], 90)
        self.assertEqual(report["reference_comparison"]["status"], "PASS")
        with (output / "paper-v1-replayed.tsv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream, delimiter="\t"))), 90)


    def test_native_unknown_labels_and_invalid_characters(self):
        rec = {"dataset": "native", "accession": "P1", "record_key": "native:P1",
               "sequence": "ACGTR", "labels": "1-#20",
               "valid": [True, True, True, True, False]}
        indexed = index_records([rec])
        self.assertEqual([valid_label(rec, i) for i in range(5)], [1, None, None, None, None])
        invalid = dict(rec, labels="1-#X0")
        with self.assertRaisesRegex(ValueError, "Invalid native label"):
            index_records([invalid])

    def test_manifest_rejects_symlinked_artifact(self):
        import hashlib
        from my_predictor.paper_replay import verify_manifest

        root = self.root / "symlink-artifact"
        root.mkdir()
        outside = self.root / "outside.txt"
        outside.write_text("private", encoding="utf-8")
        link = root / "linked.txt"
        link.symlink_to(outside)
        digest = hashlib.sha256(outside.read_bytes()).hexdigest()
        (root / "MANIFEST.json").write_text(json.dumps({"schema_version": 1, "files": [
            {"path": "linked.txt", "bytes": outside.stat().st_size, "sha256": digest}
        ]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            verify_manifest(root)


    def test_real_default_config_and_reference_are_complete(self):
        from my_predictor.paper_replay import DEFAULT_CONFIG, load_config

        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(len(config["datasets"]), 6)
        self.assertEqual(len(config["predictors"]), 14)
        self.assertTrue(all(d["proteins"] > 0 and d["full_residues"] >= d["valid_residues"] > 0
                            and 0 < d["positive_residues"] <= d["valid_residues"] for d in config["datasets"]))
        reference = DEFAULT_CONFIG.parent / config["reference"]["path"]
        with reference.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 90)
        self.assertEqual(set(row["dataset"] for row in rows), {d["name"] for d in config["datasets"]})
        policies = {(row["model"], row["threshold_policy"]) for row in rows}
        self.assertTrue(all((p["name"], p["validation_policy"]) in policies for p in config["predictors"]))
        self.assertEqual(sum(1 for row in rows if row["model"] == "PUNCH2"), 12)
        self.assertTrue(all(row["threshold"] and float(row["threshold"]) == float(row["threshold"]) for row in rows))
        self.assertTrue(all(any(row["model"] == p["name"] for row in rows) for p in config["predictors"]))



if __name__ == "__main__":
    unittest.main()
