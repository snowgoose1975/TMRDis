from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from my_predictor.head_study_models import build
from my_predictor.paper_inference import predict_ensemble, prepare_inputs, verify_models
from my_predictor.sequence_features import sequence_features


FAMILIES = ("prott5", "esmc")
SEEDS = (20260926, 20260927, 20260928, 20260929, 20260930)
WIDTHS = {"prott5": 1066, "esmc": 1194}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_index(path: Path, family: str, embedding: Path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "path", "key", "family"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"id": "p1", "path": embedding.name,
                         "key": "layer_24" if family == "prott5" else "layer_36", "family": family})


def _make_members(root: Path) -> Path:
    models = root / "models"
    models.mkdir()
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            member = f"{family}-s{seed}"
            folder = models / f"{family}-s{seed}"
            folder.mkdir()
            torch.manual_seed(seed + (0 if family == "prott5" else 1))
            spec = {"name": "bigru", "head": "bigru", "width": 256, "msa": False}
            model = build(WIDTHS[family], spec)
            checkpoint = folder / "best.pt"
            torch.save({
                "architecture": "study:bigru",
                "input_dim": WIDTHS[family],
                "selected_epoch": 1,
                "embedding_normalization": {"method": "none"},
                "feature_preprocessing": None,
                "state_dict": model.state_dict(),
            }, checkpoint)
            (folder / "summary.json").write_text(json.dumps({"spec": spec}), encoding="utf-8")
            rows.append({
                "member": member, "branch": family, "seed": seed,
                "learning_rate": "0.0003" if family == "prott5" else "0.001",
                "selected_epoch": 1, "validation_AP": "0.75",
                "checkpoint_sha256": _sha(checkpoint),
                "checkpoint": f"{family}-s{seed}/best.pt",
            })
    table = root / "members.tsv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "member", "branch", "seed", "learning_rate", "selected_epoch",
            "validation_AP", "checkpoint_sha256", "checkpoint",
        ], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return table


def _write_manifest(path: Path, family: str, embedding: Path, length: int):
    key = "layer_24" if family == "prott5" else "layer_36"
    source = {"path": str(embedding), "key": key, "family": family}
    derived = {"path": str(path.parent / "seq.npz"), "key": "embedding", "family": "derived_sequence"}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "length", "embedding", "embedding_key", "embedding_sources",
            "feature_preprocessing", "feature_recipe",
        ], delimiter="\t")
        writer.writeheader()
        writer.writerow({"id": "p1", "length": length, "embedding": "", "embedding_key": "",
                         "embedding_sources": json.dumps([source, derived]),
                         "feature_preprocessing": "", "feature_recipe": "paper_v1_sequence42_centered_window21"})


def test_prepare_inputs_creates_exact_layer_manifests(tmp_path):
    pt_file = tmp_path / "pt.npz"
    esmc_file = tmp_path / "esmc.npz"
    np.savez(pt_file, layer_24=np.zeros((2, 1024), dtype=np.float32), __metadata_json__=np.array("producer=fixture"))
    np.savez(esmc_file, layer_36=np.ones((2, 1152), dtype=np.float32))
    fasta = tmp_path / "seq.fa"
    fasta.write_text(">p1\nAX\n", encoding="utf-8")
    pt_index, esmc_index = tmp_path / "pt.tsv", tmp_path / "esmc.tsv"
    _write_index(pt_index, "prott5", pt_file)
    _write_index(esmc_index, "esmc", esmc_file)

    output = prepare_inputs(fasta, pt_index, esmc_index, tmp_path / "prepared")
    assert (output / "prott5.tsv").is_file()
    assert (output / "esmc.tsv").is_file()
    with (output / "prott5.tsv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    sources = json.loads(row["embedding_sources"])
    assert sources[0]["key"] == "layer_24"
    assert sources[1]["family"] == "derived_sequence"
    preparation = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    protein = preparation["proteins"]["p1"]
    assert protein["sequence_sha256"] == hashlib.sha256(b"AX").hexdigest()
    assert protein["plm_sources"]["prott5"]["width"] == 1024
    assert protein["plm_sources"]["prott5"]["file_sha256"] == _sha(pt_file)
    assert protein["plm_sources"]["prott5"]["native_metadata_literal"] == "producer=fixture"
    assert protein["sequence42_file_sha256"] == _sha(Path(sources[1]["path"]))
    feature = np.load(sources[1]["path"])["embedding"]
    np.testing.assert_array_equal(feature, sequence_features("AX"))


def test_prepare_inputs_rejects_missing_requested_npz_layer(tmp_path):
    pt_file, esmc_file = tmp_path / "pt.npz", tmp_path / "esmc.npz"
    np.savez(pt_file, layer_23=np.zeros((1, 1024), dtype=np.float32))
    np.savez(esmc_file, layer_36=np.zeros((1, 1152), dtype=np.float32))
    fasta = tmp_path / "seq.fa"
    fasta.write_text(">p1\nA\n", encoding="utf-8")
    pt_index, esmc_index = tmp_path / "pt.tsv", tmp_path / "esmc.tsv"
    _write_index(pt_index, "prott5", pt_file)
    _write_index(esmc_index, "esmc", esmc_file)
    with pytest.raises(KeyError, match="layer_24"):
        prepare_inputs(fasta, pt_index, esmc_index, tmp_path / "prepared")


def test_ten_member_inference_preserves_residue_order_and_average(tmp_path):
    members = _make_members(tmp_path)
    pt_file, esmc_file = tmp_path / "pt.npz", tmp_path / "esmc.npz"
    np.savez(pt_file, layer_24=np.full((2, 1024), 0.01, dtype=np.float32))
    np.savez(esmc_file, layer_36=np.full((2, 1152), 0.02, dtype=np.float32))
    pt_manifest, esmc_manifest = tmp_path / "pt-manifest.tsv", tmp_path / "esmc-manifest.tsv"
    with (tmp_path / "seq.npz").open("wb") as handle:
        np.savez_compressed(handle, embedding=sequence_features("AA"))
    _write_manifest(pt_manifest, "prott5", pt_file, 2)
    _write_manifest(esmc_manifest, "esmc", esmc_file, 2)

    output = predict_ensemble(pt_manifest, esmc_manifest, tmp_path / "models", members, tmp_path / "scores.tsv.gz")
    with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [int(row["position"]) for row in rows] == [1, 2]
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["model_id"] == "paper-v1-retrained-ten-member-bigru"
    assert receipt["checkpoint_identity_status"] == "retrained_or_custom"
    assert receipt["output"]["sha256"] == _sha(output)
    assert receipt["records"] == 1 and receipt["residues"] == 2
    assert len(receipt["members"]) == 10
    assert set(receipt["manifests"]) == {"prott5", "esmc"}
    member_fields = [f"ours_{family}_s{seed}" for family in FAMILIES for seed in SEEDS]
    for row in rows:
        expected = np.mean([float(row[name]) for name in member_fields])
        assert float(row["score"]) == pytest.approx(expected, abs=1e-7)
        assert float(row["ours_ensemble"]) == pytest.approx(expected, abs=1e-7)
        assert 0 <= float(row["score"]) <= 1


def test_checkpoint_hash_is_checked_before_model_loading(tmp_path, monkeypatch):
    members = _make_members(tmp_path)
    rows = list(csv.DictReader(members.open(encoding="utf-8"), delimiter="\t"))
    rows[0]["checkpoint_sha256"] = "0" * 64
    with members.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    loaded = False

    def forbidden(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("torch.load must not run before digest verification")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_models(tmp_path / "models", members)
    assert not loaded


def test_real_paper_member_manifest_uses_historical_names():
    from my_predictor.paper_inference import _read_member_table

    manifest = Path(__file__).resolve().parents[1] / "configs" / "paper-v1-members.tsv"
    rows = _read_member_table(manifest)
    assert rows[0]["member"] == "prott5-s20260926"
    assert rows[0]["score_column"] == "ours_prott5_s20260926"
    assert len(rows) == 10


def test_prediction_rejects_npz_missing_requested_layer(tmp_path):
    members = _make_members(tmp_path)
    pt_file, esmc_file = tmp_path / "pt.npz", tmp_path / "esmc.npz"
    np.savez(pt_file, layer_23=np.zeros((2, 1024), dtype=np.float32))
    np.savez(esmc_file, layer_36=np.zeros((2, 1152), dtype=np.float32))
    with (tmp_path / "seq.npz").open("wb") as handle:
        np.savez_compressed(handle, embedding=sequence_features("AA"))
    pt_manifest, esmc_manifest = tmp_path / "pt-manifest.tsv", tmp_path / "esmc-manifest.tsv"
    _write_manifest(pt_manifest, "prott5", pt_file, 2)
    _write_manifest(esmc_manifest, "esmc", esmc_file, 2)

    with pytest.raises(KeyError, match="layer_24"):
        predict_ensemble(pt_manifest, esmc_manifest, tmp_path / "models", members,
                         tmp_path / "scores.tsv.gz")


def test_prepare_inputs_verifies_optional_index_hashes(tmp_path):
    pt_file, esmc_file = tmp_path / "pt.npz", tmp_path / "esmc.npz"
    np.savez(pt_file, layer_24=np.zeros((1, 1024), dtype=np.float32))
    np.savez(esmc_file, layer_36=np.zeros((1, 1152), dtype=np.float32))
    fasta = tmp_path / "seq.fa"
    fasta.write_text(">p1\nA\n", encoding="utf-8")
    pt_index, esmc_index = tmp_path / "pt.tsv", tmp_path / "esmc.tsv"
    fields = ["id", "path", "key", "family", "sha256", "sequence_sha256"]
    for index_path, family, embedding, key in (
        (pt_index, "prott5", pt_file, "layer_24"),
        (esmc_index, "esmc", esmc_file, "layer_36"),
    ):
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerow({"id": "p1", "path": embedding.name, "key": key, "family": family,
                             "sha256": _sha(embedding), "sequence_sha256": hashlib.sha256(b"A").hexdigest()})
    prepare_inputs(fasta, pt_index, esmc_index, tmp_path / "valid")

    with pt_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({"id": "p1", "path": pt_file.name, "key": "layer_24", "family": "prott5",
                         "sha256": "0" * 64, "sequence_sha256": hashlib.sha256(b"A").hexdigest()})
    with pytest.raises(ValueError, match="file SHA256 mismatch"):
        prepare_inputs(fasta, pt_index, esmc_index, tmp_path / "wrong-file-hash")

    with pt_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({"id": "p1", "path": pt_file.name, "key": "layer_24", "family": "prott5",
                         "sha256": _sha(pt_file), "sequence_sha256": "0" * 64})
    with pytest.raises(ValueError, match="sequence SHA256"):
        prepare_inputs(fasta, pt_index, esmc_index, tmp_path / "wrong-sequence-hash")
