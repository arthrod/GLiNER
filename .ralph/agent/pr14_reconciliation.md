# PR14 Reconciliation (vs dev)

Date: 2026-02-18

## Fetch status

- Remote PR14 head fetched with `git fetch origin pull/14/head:pr-14`.
- Local `pr-14` head: `5906b4b61cce38f79de7b8bdc58eab7d092d340a`.
- Unique commits on `pr-14` vs `dev`: `15072b1`, `c4884df`, `531a66d`, `3e5753e`, `5906b4b`.

## Diff reconciliation summary

`pr-14` was branched from an older base. The only PR14-unique functional change is the new `tests/test_validator_integration.py` file, while many other file differences are branch-base drift. The test file is intentionally "bug documenting" and includes assertions that expect broken behavior, so it is not merged directly into `dev`.

## Comment triage

### Relevant and still unresolved (in scope for next tasks)

1. PTBR forwarding gaps remain in `ptbr/training_cli.py`:
- `training.dataloader_pin_memory`
- `training.dataloader_persistent_workers`
- `training.dataloader_prefetch_factor`
- run metadata forwarding (`run.name` -> `run_name`)

2. `train.py` still hardcodes values instead of using config-derived values:
- `output_dir="models"`
- `bf16=True`
- eval batch size forced to train batch size

3. Need deterministic propagation tests proving config reaches:
- PTBR -> `BaseGLiNER.train_model`
- `BaseGLiNER.create_training_args` -> Transformers `TrainingArguments`
- torch-facing pathways via CPU-safe fake/meta checks

### Not applied from PR14 test file

1. Minor nits inside `tests/test_validator_integration.py` (`unused import train_app`, `unused config_ok`, `unused config_ok_with_gc_format`, AST consistency suggestion for `run_tags`) are not applied because this file is not being ported.

2. The large PR14 integration test file asserts current bugs rather than expected production behavior; upcoming tests will be smaller, deterministic, and assertion-positive for fixed propagation.

### Confirmed non-blocking for this objective

1. `remove_unused_columns` default concern is not the primary target here because GLiNER's custom trainer dataloaders bypass HF column-pruning in the train/eval dataloader path.

## Resulting execution path

- Task `task-1771380639-3a36`: add focused deterministic propagation tests.
- Task `task-1771380639-4985`: implement forwarding/hardcode fixes for failing tests.
- Task `task-1771380639-8939`: verify, merge to `staging`, and push.
