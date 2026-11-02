# Paper-v1 publication worktree rules

This isolated worktree prepares the user-approved reproducibility release.
The repository is the authoritative own-predictor source; Windows PLM holds
benchmark records and baseline build recipes. Preserve existing Git history.

## Directory ownership

- `my_predictor/`: installable model, sequence42, IO, training, inference,
  metrics and publication CLI modules.
- `scripts/`: legacy entry points, preserved for historical/general use.
- `configs/`: versioned paper protocols, exact member identities and resource
  manifests; contain no private operational paths or credentials.
- `docs/`: reproduction commands, baseline provenance, artifact access status
  and historical/general-interface documentation.
- `tests/`: regression, installation and bounded CPU pipeline tests.
- `container/`: reproducible runtime instructions.
- `artifacts/`, `outputs/`, `dist/`, `.venv/`, `build/`: ignored runtime files;
  do not publish data, model binaries, private paths or generated source caches.

Document a purpose before creating a directory. Keep feature/score coordinate
contracts, historical checkpoints, native dataset masks and frozen thresholds.
Do not select parameters or thresholds on external labels. Changes to the AP
implementation require tie-handling regression tests. Generic `bigru` and
paper `study:bigru` are different interfaces.

Before committing run the relevant pytest suite and `git diff --check`; inspect
staged files for accidental resources/credentials. Source publication and push
for this release are explicitly authorized by the user (2026-11-02). Deletion,
force-push, history rewrite, system/credential changes are not authorized.
