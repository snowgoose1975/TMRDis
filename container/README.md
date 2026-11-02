# Paper-v1 container execution

Build from this exact source revision:

```bash
docker build -f Dockerfile.cpu -t disorder-paper:cpu .
docker build -f Dockerfile.gpu -t disorder-paper:cuda124 .
docker run --rm disorder-paper:cpu --help
```

The images install the `disorder-paper` command and include the paper
configuration and documentation. CPU requirements pin NumPy 2.1.2 and
PyTorch 2.5.1+cpu. The GPU base pins PyTorch 2.5.1, CUDA 12.4 and cuDNN 9.
Base tags are version constraints; record the resolved image digest when
building. No unverified registry tag is advertised as already published.

Mount user-provided resources read-only and a new output directory read-write.
For example, after exporting a portable replay artifact bundle:

```bash
docker run --rm -v "$PWD/artifacts:/artifacts:ro" -v "$PWD/results:/results" \
  disorder-paper:cpu replay --help
```

Use `--gpus all` on a compatible local Docker GPU host. On a shared cluster,
obtain the required allocation before GPU execution. The software does not
require Slurm on a standalone machine. The images do not contain checkpoints,
PLM weights, data or third-party predictors. Their access requirements are in
`docs/paper-resources.json` and `docs/baseline-reproduction.md`.

The general historical `scripts.*` interfaces remain source entry points.
Their ESM2 workflows are distinct from the paper-v1 dual-PLM model. To inspect
one inside the source-containing image, use an explicit Python entrypoint:

```bash
docker run --rm --entrypoint python disorder-paper:cpu -m scripts.predict --help
```
