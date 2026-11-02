# Reproducing the TMRDis paper-v1 six-set baseline comparison

This workflow separates metric replay from model inference. Metric replay needs
only the frozen residue labels/masks, full-length residue score tables, the
versioned threshold evidence, and the standard-library evaluator. It does not
load model weights, containers, PLM embeddings, or any prediction code.

## Metric protocol

The six cohorts are MemDis TEST, CAID3 PDB, CAID3 NOX, CAID2 PDB, CAID2 NOX,
and TriZOD TEST. Metrics pool valid residues within each dataset. A residue is
scored only when its native mask is true and its label is binary 0/1. Score
tables still cover every position in each complete input sequence, including
unlabeled positions.

Each method uses its frozen global residue-level MCC threshold selected on
V2 development validation. External labels never select a threshold. A residue
is predicted disordered when score >= threshold. PUNCH2 is reported at both
its validation-MCC threshold and its official default threshold 0.39; these
are two operating points for one score set. PUNCH2-Light is a separate model
configuration.

Equal scores are grouped for average precision, trapezoidal AUPRC, AUROC,
threshold selection, and ranking precision. Tied AUROC pairs contribute one
half. Precision at a rank boundary that cuts through tied scores is the
fractional expected precision within that score group. Brier score and
10-bin expected calibration error are omitted for ADOPT and SETH because their
scores are not probabilities.

## Portable replay artifact

The export command writes a new artifact directory with:

- `records.jsonl.gz`: dataset, accession, record key, full sequence,
  per-residue labels and valid mask, plus exact-development-overlap flags.
- `scores/<predictor>.tsv.gz`: complete residue coordinates, residue letters,
  and continuous scores for each of the 14 distinct methods.
- `thresholds.json`: frozen threshold values, evidence file hashes, source
  receipt hashes, and threshold protocol.
- `paper-v1-evaluation.json` and `paper-v1-reference.tsv`: the versioned
  replay config and immutable expected 90-row table.
- `MANIFEST.json`: SHA256 and byte size for every artifact file except the
  manifest itself. Replay verifies the manifest before evaluating.

The private source roots must be supplied explicitly to export; the command
does not contain machine-specific paths and does not infer download locations.
If a required score export, label plan, completion receipt, or threshold record
is absent, export fails with the missing resource path. It never substitutes
the precomputed merged metrics table for continuous scores.

From a checkout with Python 3.10 or newer, run:

    python -m my_predictor.paper export-resources \
      --own-root /path/to/external-evaluation-root \
      --baseline-root /path/to/baselines-external-root \
      --punch2-root /path/to/punch2-external-root \
      --handoff-root /path/to/model-handoff-root \
      --artifact-root /path/to/new-paper-v1-artifact

Then recompute and compare all published numeric fields:

    python -m my_predictor.paper replay \
      --artifact-root /path/to/new-paper-v1-artifact \
      --output-dir /path/to/new-metrics-output

The replay expects exactly 1,625 records and checks each dataset's protein,
full-residue, valid-residue, and positive-residue counts against the frozen
config. Every predictor must have one score for every sequence coordinate.
It checks dataset/accession/position uniqueness, sequence letters, masks,
finite values, probability declarations, the 90 expected operating rows, and
all numeric reference fields to an absolute tolerance of 1e-9. Reference
cells that were blank because the source did not report a value (for example,
some confusion counts or non-probability calibration metrics) remain
unconstrained; replay does not invent them.

For a single predictor score file, evaluate accepts the same normalized
score-table schema and either exported records or a native frozen-plan JSON:

    python -m my_predictor.paper evaluate \
      --scores /path/to/predictor.tsv.gz \
      --labels /path/to/records.jsonl.gz \
      --threshold 0.42 \
      --output /path/to/metrics.json

Use the frozen threshold from its evidence record. Do not pass a threshold
selected on the external dataset.

## Metric replay versus raw inference

The portable artifact is sufficient to recompute the reported metrics. It does
not claim to be a self-contained raw-inference release. The current retained
score artifacts and source-run identifiers are evidence of the accepted
prediction variants, not instructions that identify their upstream revisions.
Where the retained handoff provenance pins an upstream URL or commit, it is
listed below. The accepted result may still rely on a different packaging
variant or incomplete runtime metadata; do not treat a tag or image digest as
a source commit.

| Method | Accepted prediction variant / retained run | Upstream URL and commit |
|---|---|---|
| ADOPT | non-MSA ESM-1b; `adopt-esm1b-sixset-20261029-retry1` | [PeptoneLtd/ADOPT](https://github.com/PeptoneLtd/ADOPT), commit `a2d79b30a1f4b04f1134bea366223d5b8e4ca0af` |
| AIUPred | CLI 2.1.0; `aiupred-preoct-c5cd-20261029` | [doszilab/AIUPred](https://github.com/doszilab/AIUPred), historical-matching commit `c5cd78d6d485c09001964e6a98c08b993907eca1` |
| DisPredict3.0 | `dispredict3-sixset-20261030-r1` | [wasicse/Dispredict3.0](https://github.com/wasicse/Dispredict3.0), commit `26ad4e8898cb2fbcb87954b4c3d2189b84be7874` |
| ESMDisPred-DNN | layer-33 DNN variant; `esmdispred-dnn-sixset-recovery-20261030-r1` | [wasicse/ESMDisPred](https://github.com/wasicse/ESMDisPred), commit `9913a9dcfdf6ce7d6fa7e712c0dd2360b33a6701` |
| flDPnn | original flDPnn, not flDPnn3; accepted runtime image digest `sha256:b350367d928f1547d6bf6eb2073eefccd37d290c7444812ff76f660625e574e4` | Source Git commit not established; the accepted container is identified by digest |
| IDP-EDL | task heads in training mode, encoders in eval mode; `idp-edl-complete-20261029` | [joestarXjx/IDP-EDL](https://github.com/joestarXjx/IDP-EDL), source `e3e667b01e47687ba4f877cd91bfd5d57ebf37c2`; ProtT5 `94a6abc029ae13029317b140b7424e012bf8dfbf` |
| IUPred2A | long-mode scores; `iupred2a` | [doszilab/iupred2a](https://github.com/doszilab/iupred2a); exact run commit is not recovered from the historical moving-master build |
| LoRA-DR-suite | raw V2, ESM2-650M LoRA-ID; `lora-dr-suite-raw-v2-20261028` | [GianLMB/LoRA-DR-suite](https://github.com/GianLMB/LoRA-DR-suite), author mirror commit `21aafe39f4f5e47d5d9c34b83bfb09913de6d215`; adapter `028ca2c6411950df3bd1f684c2359a6b39c1e21c`; base `08e4846e537177426273712802403f7ba8261b6c` |
| metapredict V3 | package 3.0.2, canonical V3; `metapredict_v3-canonical-20261028` | [idptools/metapredict](https://github.com/idptools/metapredict), package version 3.0.2; source commit not pinned in the retained record |
| PUNCH2 | full variant; official long-sequence policy uses Light branches above 1,022 residues; validation and official 0.39 points | [deemeng/punch2](https://github.com/deemeng/punch2), commit `815a1040e38bf8c6930c11f51067506932db6d94` |
| PUNCH2-Light | distinct 13-branch configuration; `punch2_light` | [deemeng/punch2_light](https://github.com/deemeng/punch2_light), image source commit `6c7935b3597c056d2e6b3845bb54fc101c5bc574` |
| SETH | direction-normalized continuous CheZOD-derived scores; `seth` | [DagmarIlz/SETH](https://github.com/DagmarIlz/SETH), commit `19265b2e9c1e2d17264a771c93a0915451e7892d` |
| UdonPred | `disprot` head; disorder-positive scores; `udonpred-sixset-20261030-r1` | [DavidWagemann/UdonPred](https://github.com/DavidWagemann/UdonPred), commit `80c5f0abd0debead4659a7b1d45d2cec65f5fb18` |

These run names identify the accepted score receipts, not upstream Git
commits. Exact runtime versions, model-weight revisions and third-party
database snapshots are not uniformly established by the retained receipt set.
They are therefore a material limitation for raw inference reproduction.
This repository does not include the third-party weights, databases or
containers, and it does not supply invented download links.

Re-running our model requires its checkpoint files and the matching frozen
ProtT5/ESMC embeddings; the embeddings are maintained in the separate V2 data
release. The public resource manifest currently has no download URLs for the
author-supplied checkpoints, frozen development records, embeddings, or
six-set replay bundle. Re-running third-party predictors additionally requires
each method's exact source, checkpoint/model files, environment, parameters,
and licensed access to any external database or service it uses.

Public code and weight terms still require author confirmation. Third-party
redistribution terms have not been uniformly verified; IUPred2A is restricted
to academic use, and unverified third-party containers/model binaries were not
republished. Data access and redistribution terms for each external benchmark
must also be confirmed before publishing the portable artifact. No download
URL is asserted here without a verified source and redistribution basis.

The export and replay process preserves source receipt hashes and threshold
evidence hashes. Those hashes establish provenance for existing predictions;
they do not guarantee that each method can be reinstalled or rerun from this
repository alone.
