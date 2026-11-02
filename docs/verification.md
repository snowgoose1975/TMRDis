# TMRDis paper-v1 verification record

The machine-readable [verification.json](verification.json) records source
fingerprints, the tested wheel, actual coverage and reference comparisons.
Every numerical runtime module matches a SHA256 recorded from the installed
wheel used for real-data acceptance. The package initializer was subsequently
updated only to align `__version__` with distribution version 0.3.0; its
installed metadata is checked separately. The validation source snapshot
identifier is a local immutable archive identity, separate from the later
public GitHub commit containing these same runtime files.

## Source and installation

- **72 tests passed** on Python 3.12.3, NumPy 2.5.3 and PyTorch 2.14.0+cpu.
- The wheel was installed with no dependency downloads into a new target
  outside the source checkout. Package imports and configuration discovery
  were verified to come from that installed target.
- Both real-checkpoint inference and full metric replay ran through the
  installed package without using source-checkout imports.
- The original AP tie-handling defect is corrected. Tests cover tied-score
  order invariance, exact layer selection, masks, malformed coordinates,
  checkpoint hashes, model identity, checkpoint selection and matrix handoff.

## Real ten-member inference

The shortest, median and longest unique frozen external sequences had lengths
**13, 327 and 9,439**, totaling **9,779** residue positions and **97,790** member
probabilities. All ten historical checkpoint hashes were verified before
loading. The receipt identifies the checkpoint set as `original`.

On the tested CPU runtime, the maximum absolute difference was **0** against
both the retained inference implementation and previously retained external
predictions, for every member, both branch means and the final ensemble on
these three proteins. The longest sequence was processed in full.
This sample agreement is not a guarantee of bitwise equality on all hardware.

## Full six-dataset metric replay

The replay artifact contains 1,625 full-sequence records and continuous scores
for TMRDis plus 13 baselines. IDs, full residue coordinates, sequence
letters, native masks, finite scores and cohort counts are checked for each
method. Artifact files are SHA256-verified before evaluation.

Independent replay regenerated **90 operating-point rows** across six datasets.
All **2,142 available reference numeric fields matched within 1e-9**. PUNCH2's
validation-MCC and official 0.39 thresholds were evaluated on the same scores.
The installed-wheel replay repeated this result, independently of the initial
source-checkout replay. Blank historical reference fields are not invented.

## Training and limits

A one-batch regression compares the preserved historical trainer with the
paper training epoch under identical initial weights, dropout RNG, variable
sequence lengths, unknown labels/masks and AdamW settings. All parameter
updates, loss and AP are exactly equal. Bounded training/save/reload tests
cover both PLM families; the full training matrix emits a new hashed member
index and does not relabel retrained checkpoints as the original model.

Full ten-member retraining and third-party baseline reinstallation were not
performed during this release. Those require the separately supplied data,
weights, feature assets and method-specific environments. Original missing
historical logs remain missing. GPU execution and exact raw-FASTA PLM feature
regeneration were not acceptance-tested.

The retained historical runtime was Python 3.11.10, NumPy 2.1.2 and PyTorch
2.5.1+cu124; its version constraints are included separately from the actual
CPU acceptance runtime above. Dockerfiles are build recipes. The Docker
daemon was unavailable, so these Docker images were not built or published
as part of this verification.
