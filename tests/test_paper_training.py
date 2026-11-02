from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from my_predictor.data import ManifestDataset, pad_batch, read_manifest
from my_predictor.head_study_models import build
from my_predictor.metrics import average_precision
import my_predictor.paper_training as paper_training
from my_predictor.paper_training import train_main, train_matrix_main
from my_predictor.paper_dataset import prepare_training_inputs
from my_predictor.sequence_features import FEATURE_RECIPE


def _prepared(tmp_path):
    fasta = tmp_path / "sequences.fasta"
    fasta.write_text(">train-protein\nACDE\n>val-protein\nFGHIK\n", encoding="utf-8")
    records = tmp_path / "development.jsonl"
    rows = [
        {"id": "train-protein", "sequence": "ACDE", "split": "train",
         "labels": [0, 1, 2, 1], "mask": [1, 1, 1, 1]},
        {"id": "val-protein", "sequence": "FGHIK", "split": "val",
         "labels": [0, 1, 2, 1, 1], "mask": [1, 1, 1, 0, 1]},
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    indices = {}
    for family, width, layer in (("prott5", 1024, 24), ("esmc", 1152, 36)):
        index = tmp_path / f"{family}.tsv"
        with index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("id", "path", "key", "family"), delimiter="\t")
            writer.writeheader()
            for protein_id, sequence in (("train-protein", "ACDE"), ("val-protein", "FGHIK")):
                path = tmp_path / f"{family}-{protein_id}.npz"
                values = np.arange(len(sequence) * width, dtype=np.float32).reshape(len(sequence), width) / 1000
                np.savez(path, **{f"layer_{layer}": values})
                writer.writerow({"id": protein_id, "path": path.name, "key": f"layer_{layer}", "family": family})
        indices[family] = index
    output = prepare_training_inputs(
        fasta, indices["prott5"], indices["esmc"], records, tmp_path / "prepared"
    )
    return output


def _train_args(output, family="prott5", epochs="1"):
    return [
        "--train-manifest", str(output / f"{family}-train.tsv"),
        "--val-manifest", str(output / f"{family}-val.tsv"),
        "--family", family,
        "--seed", "20260926",
        "--output-dir", str(output / f"run-{family}"),
        "--epochs", epochs,
        "--batch-size", "1",
        "--device", "cpu",
        "--threads", "1",
        "--cache-embeddings",
    ]


@pytest.mark.parametrize(("family", "input_dim", "lr"), [
    ("prott5", 1066, 0.0003),
    ("esmc", 1194, 0.001),
])
def test_public_preparation_trains_reloadable_study_checkpoint(tmp_path, family, input_dim, lr):
    prepared = _prepared(tmp_path)
    train_rows = read_manifest(prepared / f"{family}-train.tsv")
    sequence_source = json.loads(train_rows[0]["embedding_sources"])[1]
    assert sequence_source["family"] == "derived_sequence"
    assert sequence_source["key"] == "embedding"
    assert train_rows[0]["feature_recipe"] == FEATURE_RECIPE

    args = _train_args(prepared, family)
    summary = train_main(args)
    out = prepared / f"run-{family}"
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert summary["learning_rate"] == lr
    assert checkpoint["config"]["learning_rate_overridden"] is False
    assert checkpoint["config"]["feature_recipe"] == FEATURE_RECIPE
    assert checkpoint["architecture"] == "study:bigru"
    assert checkpoint["input_dim"] == input_dim
    assert checkpoint["hidden_dim"] == 256
    assert checkpoint["dropout"] == 0.1
    assert checkpoint["embedding_normalization"] == {"method": "none"}
    assert checkpoint["selected_epoch"] == summary["best_epoch"] == 1
    assert checkpoint["validation_average_precision"] == summary["best_validation_average_precision"]

    model = build(input_dim, {"name": "bigru", "head": "bigru", "width": 256})
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    sample = ManifestDataset(prepared / f"{family}-val.tsv", normalization={"method": "none"}, cache_samples=True)[0]
    batch, _labels, valid, _ids, lengths = pad_batch([sample])
    with torch.inference_mode():
        logits = model(batch, lengths)
    assert logits.shape == (1, 5)
    assert torch.isfinite(logits).all()
    labels = torch.from_numpy(sample["labels"])[None]
    expected_ap = average_precision(logits.numpy(), labels.numpy(), valid.numpy())
    assert expected_ap == summary["best_validation_average_precision"]

    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["validation"]["known_residues"] == 3
    assert history[0]["validation"]["average_precision"] is not None
    assert len(checkpoint["config"]["input_hashes"]["manifests"]) == 2
    assert len(checkpoint["config"]["input_hashes"]["assets"]) == 8
    assert (out / "summary.json").exists()


@pytest.mark.parametrize("bad_source", ["wrong_key", "missing_npz_key", "wrong_sequence_source", "unknown_recipe"])
def test_rejects_sources_that_cannot_be_loaded_by_paper_inference(tmp_path, bad_source):
    prepared = _prepared(tmp_path)
    manifest = prepared / "prott5-train.tsv"
    rows = read_manifest(manifest)
    sources = json.loads(rows[0]["embedding_sources"])
    if bad_source == "wrong_key":
        sources[0]["key"] = "layer_124"
        message = "must be exactly 'layer_24'"
    elif bad_source == "missing_npz_key":
        plm = np.load(sources[0]["path"])["layer_24"]
        npz_path = tmp_path / "missing-layer.npz"
        np.savez(npz_path, layer_124=plm)
        sources[0]["path"] = str(npz_path)
        message = "no required NPZ member 'layer_24'"
    elif bad_source == "wrong_sequence_source":
        sources[1]["family"] = "sequence42"
        message = "sequence42 source must be family=derived_sequence"
    else:
        rows[0]["feature_recipe"] = "unknown-sequence-recipe"
        message = "unsupported sequence feature recipe"
    rows[0]["embedding_sources"] = json.dumps(sources)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises((ValueError, KeyError), match=message):
        train_main(_train_args(prepared))


def test_rejects_labels_outside_native_three_value_contract(tmp_path):
    prepared = _prepared(tmp_path)
    row = read_manifest(prepared / "prott5-train.tsv")[0]
    np.save(row["labels"], np.asarray([0, 1, 3, 1], dtype=np.int8), allow_pickle=False)
    with pytest.raises(ValueError, match="labels must contain only 0, 1, or 2"):
        train_main(_train_args(prepared))


def test_lower_validation_ap_never_replaces_the_best_checkpoint(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    validation_scores = iter((0.8, 0.4))
    snapshots = []

    def fake_epoch(model, _loader, _device, optimizer):
        if optimizer is not None:
            with torch.no_grad():
                model.project.weight.add_(1.0)
            return {"loss": 0.5, "average_precision": 0.5, "known_residues": 3}
        snapshots.append({key: tensor.detach().clone() for key, tensor in model.state_dict().items()})
        return {"loss": 0.5, "average_precision": next(validation_scores), "known_residues": 3}

    monkeypatch.setattr(paper_training, "_epoch", fake_epoch)
    args = _train_args(prepared)
    args[args.index("--epochs") + 1] = "2"
    summary = train_main(args)
    checkpoint = torch.load(prepared / "run-prott5" / "best.pt", map_location="cpu", weights_only=False)
    assert summary["best_epoch"] == checkpoint["selected_epoch"] == 1
    for key, value in snapshots[0].items():
        torch.testing.assert_close(checkpoint["state_dict"][key], value)
    assert not torch.equal(snapshots[0]["project.weight"], snapshots[1]["project.weight"])


def test_rejects_mask_length_mismatch_before_training(tmp_path):
    prepared = _prepared(tmp_path)
    row = read_manifest(prepared / "prott5-train.tsv")[0]
    np.save(row["mask"], np.asarray([1, 1, 1], dtype=np.uint8), allow_pickle=False)
    with pytest.raises(ValueError, match="mask length 3 != labels length 4"):
        train_main(_train_args(prepared))


def test_training_epoch_matches_public_historical_run_epoch_exactly():
    from scripts.train import run_epoch as historical_run_epoch

    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    samples = []
    for identity, length, raw_labels, mask in (
        ("short-a", 4, [0, 1, 2, 1], [1, 1, 1, 0]),
        ("short-b", 3, [1, 0, 1], [1, 0, 1]),
    ):
        labels = np.asarray(raw_labels, dtype=np.float32)
        valid = (labels != 2) & np.asarray(mask, dtype=bool)
        samples.append({
            "id": identity,
            "embedding": rng.normal(size=(length, 1066)).astype(np.float32),
            "labels": (labels == 1).astype(np.float32),
            "valid": valid,
        })
    loader = torch.utils.data.DataLoader(
        samples, batch_size=2, shuffle=False, collate_fn=pad_batch
    )

    spec = {"name": "bigru", "head": "bigru", "width": 256}
    torch.manual_seed(11)
    initialized = build(1066, spec)
    initial_state = {key: value.detach().clone() for key, value in initialized.state_dict().items()}
    current = build(1066, spec)
    historical = build(1066, spec)
    current.load_state_dict(initial_state)
    historical.load_state_dict(initial_state)
    current_optimizer = torch.optim.AdamW(
        current.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    historical_optimizer = torch.optim.AdamW(
        historical.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    torch.manual_seed(20260926)
    current_metrics = paper_training._epoch(
        current, loader, torch.device("cpu"), current_optimizer
    )
    torch.manual_seed(20260926)
    historical_loss, _accuracy, historical_ap = historical_run_epoch(
        historical,
        loader,
        torch.device("cpu"),
        torch.nn.BCEWithLogitsLoss(reduction="none"),
        historical_optimizer,
        "residue",
    )

    maximum_parameter_difference = max(
        float((current.state_dict()[key] - historical.state_dict()[key]).abs().max())
        for key in initial_state
    )
    assert maximum_parameter_difference == 0.0
    assert current_metrics["loss"] == historical_loss
    assert current_metrics["average_precision"] == historical_ap


def test_full_matrix_writes_members_tsv_with_checkpoint_hashes(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)

    def fake_train_one(*, family, seed, output_dir, lr, **_kwargs):
        output_dir.mkdir(parents=True)
        checkpoint = output_dir / "best.pt"
        checkpoint.write_bytes(f"checkpoint:{family}:{seed}".encode())
        summary = {
            "family": family,
            "seed": seed,
            "learning_rate": lr,
            "selected_epoch": (seed - 20260925),
            "metrics": {"average_precision": 0.5 + (seed - 20260926) / 100},
        }
        (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(paper_training, "_train_one", fake_train_one)
    root = tmp_path / "matrix"
    state = train_matrix_main([
        "--prott5-train-manifest", str(prepared / "prott5-train.tsv"),
        "--prott5-val-manifest", str(prepared / "prott5-val.tsv"),
        "--esmc-train-manifest", str(prepared / "esmc-train.tsv"),
        "--esmc-val-manifest", str(prepared / "esmc-val.tsv"),
        "--output-dir", str(root),
    ])
    assert state["phase"] == "complete"
    assert state["member_table"] == "members.tsv"
    with (root / "members.tsv").open(encoding="utf-8", newline="") as handle:
        members = list(csv.DictReader(handle, delimiter="\t"))
    assert len(members) == 10
    assert list(members[0]) == [
        "member", "branch", "seed", "learning_rate", "selected_epoch",
        "validation_AP", "checkpoint_sha256", "checkpoint",
    ]
    for row in members:
        assert row["member"] == f"{row['branch']}-s{row['seed']}"
        run_name = f"{row['branch']}-seed-{row['seed']}"
        assert row["checkpoint"] == f"{run_name}/best.pt"
        checkpoint = root / row["checkpoint"]
        assert row["checkpoint_sha256"] == paper_training._hash(checkpoint, {}, {})
        assert row["selected_epoch"] == str(int(row["seed"]) - 20260925)
        assert float(row["validation_AP"]) > 0.0
