# Historical general prediction-head workflows

The following is the preserved pre-paper README. Image tags and data contracts
are historical; use the current root README for paper-v1.

# zchen-disorder-predictor

Residue-level intrinsic-disorder prediction heads for protein-language-model embeddings. The project keeps the upstream representation fixed while comparing small prediction heads and evaluating continuous residue scores.

ESM2 is an optional upstream encoder. It is not counted as a separate disorder predictor.

## Run the published Docker image

You do not need to clone this repository just to run a published image.

```bash
docker pull thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1

docker run --rm \
  thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1 \
  scripts.predict --help
```

For a CUDA-capable Docker host:

```bash
docker pull thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1

docker run --rm --gpus all \
  thesnowgoose19750415/zchen-disorder-predictor:gpu-esm2-v1 \
  scripts.train --help
```

The CPU and GPU images contain the same predictor source code and differ in their PyTorch runtime. The GPU tag requires a working NVIDIA Container Toolkit on the host.

## Input contract

The main path is:

```text
residue embedding [L, D] -> prediction head -> disorder scores [L]
```

Training manifests contain at least:

```text
id  split  embedding  embedding_key  labels  mask  length
```

Embeddings may be `.npy` arrays or `.npz` archives. The current ESM2 handoff uses `layer_30` with width 1280; the embedding family, layer, and width must match the checkpoint. Label value `2` means unknown and is excluded from loss and metrics through the mask.

## Prediction heads

- `linear`: residue-wise linear head
- `mlp`: multilayer perceptron
- `tcn`: temporal convolutional head
- `bigru`: bidirectional GRU sequence-context head
- `conv_bigru`: convolution plus bidirectional GRU

## Build a manifest

If your handoff contains `catalogue/records.tsv`,
`development/records.jsonl.gz`, and `catalogue/assets.tsv`, build the shared
train/validation/test manifests as follows:

```bash
mkdir -p manifests
docker run --rm \
  -v "$PWD/data/handoff:/data/handoff:ro" \
  -v "$PWD/manifests:/work/manifests" \
  thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1 \
  scripts.build_manifest \
  --data-root /data/handoff \
  --output-dir /work/manifests/esm2-layer30 \
  --asset-type esm2 \
  --embedding-key layer_30
```

The command writes split TSV files and label/mask assets under `manifests/`.
Because the TSV stores container-visible paths, keep the `/data/handoff` and
`/work/manifests` mount points unchanged in later commands. If you already
have a compatible manifest, skip this step.

## Train, predict, and evaluate

Mount source data read-only and outputs/checkpoints read-write:

```bash
docker run --rm \
  -v "$PWD/data/handoff:/data/handoff:ro" \
  -v "$PWD/manifests:/work/manifests:ro" \
  -v "$PWD/outputs:/work/outputs" \
  thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1 \
  scripts.train \
  --train-manifest /work/manifests/esm2-layer30/train.tsv \
  --val-manifest /work/manifests/esm2-layer30/val.tsv \
  --output-dir /work/outputs/mlp-esm2-layer30 \
  --architecture mlp \
  --epochs 10 \
  --device cpu

docker run --rm \
  -v "$PWD/data/handoff:/data/handoff:ro" \
  -v "$PWD/manifests:/work/manifests:ro" \
  -v "$PWD/outputs:/work/outputs" \
  thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1 \
  scripts.predict_manifest \
  --checkpoint /work/outputs/mlp-esm2-layer30/best.pt \
  --manifest /work/manifests/esm2-layer30/test.tsv \
  --output /work/outputs/mlp-esm2-layer30/test.prediction.tsv \
  --device cpu

docker run --rm \
  -v "$PWD/data/handoff:/data/handoff:ro" \
  -v "$PWD/manifests:/work/manifests:ro" \
  -v "$PWD/outputs:/work/outputs" \
  thesnowgoose19750415/zchen-disorder-predictor:cpu-esm2-v1 \
  scripts.evaluate \
  --prediction /work/outputs/mlp-esm2-layer30/test.prediction.tsv \
  --manifest /work/manifests/esm2-layer30/test.tsv \
  --output-json /work/outputs/mlp-esm2-layer30/test.metrics.json \
  --output-protein-tsv /work/outputs/mlp-esm2-layer30/test.protein.tsv
```

The evaluator checks IDs and residue positions, removes unknown or masked residues, and reports residue-level AUROC, average precision, Brier score, calibration error, MCC, F1, coverage, and protein-level summaries. Select a binary threshold on validation data and freeze it before test evaluation.

## Embeddings and FASTA

Precomputed embeddings skip the ESM2 encoder and do not require ESM2 weights. FASTA-to-ESM2 inference requires a compatible local model directory or model download. The checkpoint and encoder must use the same embedding family, layer, and width.

Do not inject external ESM2 embeddings into an external baseline unless that baseline explicitly documents the same embedding interface. Most published disorder baselines accept FASTA and generate their own internal features.

## Build locally

Clone this repository when you need to inspect or modify source, build an image, or run tests:

```bash
docker build -f Dockerfile.cpu -t my-disorder-predictor:cpu .
docker build -f Dockerfile.gpu -t my-disorder-predictor:gpu .
```

The image entrypoint is `python -m`, so commands such as `scripts.train` and `scripts.evaluate` are Python modules inside the container. Mount input data read-only and keep checkpoints and generated outputs outside the image.

Record the embedding family/layer, model configuration, seed, manifest metadata, exact command, image tag or digest, and output checksums. Do not commit private data, checkpoints, credentials, or generated outputs.
