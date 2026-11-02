# Publication workspace layout

The paper-v1 release preserves the exact model and evaluation contract in
`configs/paper-v1.json`. `my_predictor/` contains installed runtime code;
`configs/` contains frozen protocol and model identities; `docs/` contains
commands and provenance; `tests/` contains bounded checks. Runtime arrays,
checkpoints, user datasets and result files belong outside tracked source.

The independent Windows benchmark workspace remains the owner of historical
baseline build recipes and project records. Its source is selectively ported
for publication with source hashes in `docs/source-provenance.json`. This Git
repository remains the authoritative own-predictor implementation.

Before acceptance, run the source tests, build/install a wheel, exercise the
CLI outside the checkout and verify the retained real-data predictions and
six-dataset metrics. Preserve existing checkpoint, threshold and prediction
identities. Label 2 and masked residues do not enter loss or metrics.

No generated data, credentials, model binaries or account-specific operational
configurations belong in Git. New artifact outputs must not overwrite accepted
runs. Keep ongoing experimental configurations separately versioned from
paper-v1. A newer research commit is not automatically a newer paper model.
