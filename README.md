# Reproducible MSA-free dual-PLM BiGRU disorder prediction

This repository provides the **paper-v1** ProtT5 + ESMC BiGRU ensemble and
code for retraining, checkpoint inference and independent metric replay.
The published model configuration is the original ten-member ensemble that
produced the frozen six-external-dataset comparison.

The two branches use ProtT5 layer 24 (1024 channels) and ESMC layer 36 (1152
channels). Each is concatenated with 42 deterministic sequence features:
21-category one-hot identity and centered 21-residue category frequencies.
Each branch has five study-BiGRU heads, with seeds 20260926--20260930.
The output is the equal average of ten sigmoid probabilities. The frozen
V2 validation-MCC threshold is **0.49059052769999995**.

The paper head includes a learned 256-channel input projection before its
bidirectional GRU. It is different from the older generic `bigru` interface.
The exact forward path has no activation between the projection and GRU.
No MSA, feature normalization, sequence truncation or smoothing is used.

## Install and inspect the release configuration

Use Python 3.11 for the reference paper environment. The project also tests
its source interfaces on the runtime recorded in the verification receipt.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-paper-cpu.txt
python -m pip install --no-deps .
disorder-paper --help
disorder-paper config --output-dir paper-config
```

For the historical CUDA 12.4 runtime use `requirements-paper-cuda124.txt`
instead. CPU inference does not require a GPU or scheduler. A local CUDA
host does not require Slurm; follow your cluster's allocation rules there.

## Resources required for actual paper results

Source installation does **not** download the original training data,
precomputed PLM arrays, ten learned checkpoints or retained baseline scores.
Their identities, access status and formats are documented in
[paper-resources.json](docs/paper-resources.json) and
[the reproduction guide](docs/reproduce-paper.md).

Checkpoint hashes and original selected epochs are in
[paper-v1-members.tsv](configs/paper-v1-members.tsv). Supply the authorized
artifact bundle, with `models/prott5-s20260926/best.pt` and the other nine
members, then verify it:

```bash
disorder-paper verify-models --models-dir artifacts/models \
  --members paper-config/paper-v1-members.tsv
```

No downloadable artifact URL is claimed unless it has actually been
published. The repository does not grant redistribution rights for external
data, third-party models or containers. The resource manifest reports these
separately from code availability.

## Prepare label-free inputs and infer

Provide a FASTA and two TSV indices with columns `id`, `path`, `key`, `family`:

```text
id      path                 key       family
protein1 features/pt.npz     layer_24  prott5
```

The ESMC index uses `layer_36` and `family=esmc`. Paths are relative to the
index file, or absolute user-supplied paths. Both indices and the FASTA must
identify the same complete sequences. NPZ keys are checked explicitly;
missing layers are errors. The software does not replace missing features.

```bash
disorder-paper prepare --fasta inputs/sequences.fasta \
  --prott5-index inputs/prott5-index.tsv --esmc-index inputs/esmc-index.tsv \
  --output-dir artifacts/prepared

disorder-paper predict \
  --prott5-manifest artifacts/prepared/prott5.tsv \
  --esmc-manifest artifacts/prepared/esmc.tsv \
  --models-dir artifacts/models --members paper-config/paper-v1-members.tsv \
  --output results/ensemble.tsv.gz --device cpu --threads 2
```

Preparation computes the exact sequence42 features and creates label-free
manifests. Prediction exports complete 1-based residue coordinates, the ten
member scores, the two branch means and final ensemble. Output paths must
be new. This workflow consumes existing matching PLM arrays; it does not
silently substitute a newly extracted embedding from another encoder revision
or long-sequence recipe. See the provenance and extraction boundary in the
reproduction guide.

## Retrain the ten heads

Supply frozen development records with `id`, `sequence`, `split`, `labels`
and optional `mask` fields in JSONL or JSONL.gz. Splits are `train`, `val` or
`test`; labels are 0, 1 or 2 (unknown). Prepare training inputs separately
from label-free predictor inputs:

```bash
disorder-paper prepare-training --fasta inputs/development.fasta \
  --prott5-index inputs/development-prott5.tsv \
  --esmc-index inputs/development-esmc.tsv \
  --records inputs/development-records.jsonl.gz \
  --output-dir artifacts/development

disorder-paper train-matrix \
  --prott5-train-manifest artifacts/development/prott5-train.tsv \
  --prott5-val-manifest artifacts/development/prott5-val.tsv \
  --esmc-train-manifest artifacts/development/esmc-train.tsv \
  --esmc-val-manifest artifacts/development/esmc-val.tsv \
  --output-dir results/retrained --device cpu --threads 2
```

Defaults match the paper protocol: ten epochs, batch size four, AdamW,
residue-weighted masked BCE, no normalization, ProtT5 learning rate 0.0003
and ESMC learning rate 0.001. Each checkpoint is selected by validation AP
on logits. Unknown and masked positions do not contribute to loss or metrics.
Full input hashes, runtime versions, configuration and epoch histories are
recorded. Read `train --help` for a single family/seed run.

A complete successful paper matrix writes `results/retrained/members.tsv`.
Use `--models-dir results/retrained --members results/retrained/members.tsv`
with the prediction command. Its receipt identifies these as retrained weights.
Freshly trained weights have new hashes; the historical member table stays fixed.
An identical seed does not promise byte-identical training across hardware or
PyTorch versions. No test labels are used for checkpoint or threshold selection.

## Recompute the paper comparison

The six sets are MemDis, CAID3 PDB/NOX, CAID2 PDB/NOX and TriZOD. The portable
replay workflow validates full residue coordinates and native labels/masks,
then calculates metrics from retained continuous scores with frozen validation
thresholds. The complete comparison contains our predictor and 13 baseline
methods, with PUNCH2 also reported at its official 0.39 threshold: **90 rows**.

```bash
disorder-paper replay --config paper-config/paper-v1-evaluation.json \
  --artifact-root artifacts/sixset --output-dir results/replayed
```

The reference table is [paper-v1-reference.tsv](configs/paper-v1-reference.tsv).
AP, trapezoidal AUPRC, AUROC, MCC/F1, calibration and ranking precision are
checked against available full-precision reference fields at tolerance 1e-9.
AP groups tied scores; AP is distinct from trapezoidal AUPRC. Calibration is
undefined for methods whose scores are not probabilities. The frozen
validation threshold is never reselected on an external set.

`export-resources --help` describes how an owner of the retained experiment
artifacts can create a portable, checksummed replay bundle. This is metric
replay from scores; running the 13 third-party predictors again additionally
requires their environments, weights and inputs. The variants, exact source
identity and known gaps are in [baseline-reproduction.md](docs/baseline-reproduction.md).

## Verification and existing interfaces

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

See [verification.md](docs/verification.md) for the actual tested runtimes,
real-checkpoint agreement and full six-set replay evidence. Tests do not
substitute for access to the actual trained models and frozen datasets.

The general MLP, TCN, ESM2 and other older interfaces remain available as
historical source workflows in [legacy-workflows.md](docs/legacy-workflows.md).
They are not the paper-v1 model configuration. Ongoing seed, CNN, positional
and topology experiments do not change the versioned paper-v1 result.
