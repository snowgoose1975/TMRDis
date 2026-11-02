# Paper-v1 reproduction contract

Use the current root README for the tested CLI workflow and
`configs/paper-v1.json` for all model and training settings. Export the
installed configurations with `disorder-paper config --output-dir paper-config`.
The directory must be new. Every CLI also provides `--help`.

## Inputs and model identity

The accepted checkpoints are identified by their bytes, not just seed names.
The member TSV records family, seed, learning rate, selected epoch, validation
AP, relative checkpoint path and SHA256. The corresponding study BiGRU has a
256-dimensional linear input projection, one bidirectional 256-unit GRU layer,
output dropout 0.1 and scalar residue logits. There is no activation between
projection and GRU. A generic `ResidueBiGRU` is a different checkpoint format.

For each family, input order is its PLM array followed by sequence42. The
21 sequence categories are `ACDEFGHIKLMNPQRSTVWY` plus one category for all
other symbols. The second 21 channels are category frequencies in the window
from position i-10 through i+10, shortened at sequence boundaries. NumPy
float64 cumulative sums are used before converting frequencies to float32,
matching the retained feature generator. Labels are not used to compute them.

Accepted PLM inputs are full-length, exact-sequence-matched arrays: ProtT5
layer_24, width 1024; ESMC layer_36, width 1152. Explicit NPZ keys must exist.
The software rejects a missing key rather than selecting another layer. A
standalone NPY input is accompanied by an explicit layer declaration in its
index; its producer is responsible for the declared identity.

## Embedding extraction boundary

The frozen feature release reuses arrays produced by multiple documented
sources. Its NPZ provenance retains native encoder metadata and any
long-sequence windowing/reconstruction. Public code here prepares and consumes
those matching arrays. It does not claim that a generic new download of a
similarly named PLM will regenerate every historical array exactly.

For independent PLM regeneration, supply the exact encoder revision,
tokenization/residue handling, requested hidden layer, BOS/EOS removal,
numerical precision and long-sequence policy from the producer provenance,
then compare arrays and downstream inference with the frozen contract. If
that producer metadata is unavailable for an array, exact raw-FASTA feature
regeneration remains a declared resource limitation. It must not be replaced
by silently truncating a sequence or switching an embedding family/layer.

## Training versus historical predictions

Training reads only train and validation manifests. Defaults reproduce the
registered optimizer/loss/head/seed protocol; validation AP on logits selects
checkpoints. Output configurations include every manifest/input hash and
runtime version. Original and retrained checkpoints are separate artifacts.
Use a different member manifest with actual new hashes to infer from retrained
models; the historical paper-v1 manifest is immutable.

Train/validation IDs must be disjoint. Unknown label 2, mask-false residues and
padding are excluded from both loss and metrics. The package does not choose
thresholds from external labels. Historical selected epochs are evidence from
the retained runs, not a requirement to stop new training at those epochs.

## Evaluation and score replay

Each external dataset retains its native mask and binary-label convention.
MemDis evaluates its native annotated segments within full protein sequences.
CAID PDB and NOX label semantics remain separate. The TriZOD binary derivative
uses the frozen release's G>=0.5 policy and is not presented as an official
universal binary standard. Scores are aligned by full record identity,
1-based residue position and residue letter before pooling valid residues.

The main table uses all native labeled records. Frozen overlap annotations
must be retained and reported; filtering does not automatically establish
independence from development data or baseline training cohorts. Any filtered
comparison has a separate explicit scope rather than replacing main results.

For paper classification, each predictor uses its original frozen V2
validation MCC threshold; PUNCH2's official 0.39 operating point is an
additional row on exactly the same continuous scores. Metric replay from
saved predictions uses a 1e-9 comparison tolerance. That tolerance is not
used as a cross-hardware inference promise. The replay engine uses the Python
standard library and does not require a GPU or trained-model deserialization.

The historic handoff observed maximum CPU-versus-CUDA differences of about
6.96e-6 for the ensemble and 6.96e-5 for a member. Same-runtime agreement is
verified separately in the current validation record. Retraining is not
claimed to recover bit-identical CUDA weights across platforms.

## Artifacts, terms and provenance

`docs/paper-resources.json` states what is supplied separately. File existence
is not evidence of a public download, and a missing source/runtime is never
represented as an empty or zero-filled prediction. `export-resources` reads
source roots passed by the owner and creates a new portable replay directory;
`replay` consumes that directory and versioned configuration.

The original model snapshot and source ports are recorded in
`docs/source-provenance.json`. Data/checkpoint hashes and validation receipts
are independent identifiers from image tags and source commits. Versioned
CPU/CUDA runtime requirements are included. No new third-party redistribution
permission or code license is assigned by this source update.
