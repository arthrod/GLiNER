## Standard HuggingFace Configuration Report

### Executive Summary

Our implementation (`training_cli.py` + `template.yaml`) covers the core GLiNER-specific and standard HuggingFace TrainingArguments fields. After the fixes described below, the key forwarding gaps (dead config fields, missing explicit parameters, Hub integration, resume, `remove_unused_columns`, masking default mismatch) have been resolved. The remaining gaps are primarily in distributed training, model selection, and optional HF TrainingArguments fields that can be passed via `**kwargs`.

**Fixes applied in this revision:**
1. **Dead config removed**: `size_sup`, `shuffle_types`, `random_drop` removed from `_FIELD_SCHEMA` and `template.yaml` (never consumed by any training code)
2. **`remove_unused_columns=False`**: Now an explicit parameter in `create_training_args` (default `False`) and forwarded by `_launch_training()`
3. **Hub fields forwarded**: `push_to_hub` and `hub_model_id` now forwarded from `environment` config to `train_model()`
4. **`resume_from_checkpoint` wired**: `train_model()` now accepts `resume_from_checkpoint` parameter; `_launch_training()` detects latest checkpoint and passes it through
5. **Missing explicit parameters added to `create_training_args`**: `fp16`, `label_smoothing`, `seed`, `gradient_checkpointing`, `run_name`, `push_to_hub`, `hub_model_id`, `eval_steps`, `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor`
6. **Masking default fixed**: `create_training_args` now defaults `masking='global'` matching `TrainingArguments` default (was `'none'`)
7. **`seed` forwarded**: `run.seed` now forwarded to `train_model()` as `seed=`

### Fields Present and Correct

| Field Name | Our Location | Report Recommendation | Status |
|---|---|---|---|
| `learning_rate` (encoder) | `training.lr_encoder` | `training.learning_rate` | PARTIAL -- present but named differently; forwarded correctly to `learning_rate=` kwarg |
| `weight_decay` (encoder) | `training.weight_decay_encoder` | `training.weight_decay` | PARTIAL -- present but named differently; forwarded correctly |
| `others_lr` | `training.lr_others` | `training.others_lr` | PARTIAL -- named differently; forwarded correctly |
| `others_weight_decay` | `training.weight_decay_other` | `training.others_weight_decay` | PARTIAL -- named differently; forwarded correctly |
| `optim` / `optimizer` | `training.optimizer` | `training.optim` | PARTIAL -- named `optimizer` in our schema vs `optim` in HF; forwarded as `optim=` kwarg |
| `lr_scheduler_type` | `training.scheduler_type` | `training.lr_scheduler_type` | PARTIAL -- named differently; forwarded correctly |
| `warmup_ratio` | `training.warmup_ratio` | `training.warmup_ratio` | OK |
| `per_device_train_batch_size` | `training.train_batch_size` | `training.per_device_train_batch_size` | PARTIAL -- named differently; forwarded correctly |
| `per_device_eval_batch_size` | `training.eval_batch_size` | `training.per_device_eval_batch_size` | PARTIAL -- named differently; forwarded correctly |
| `gradient_accumulation_steps` | `training.gradient_accumulation_steps` | `training.gradient_accumulation_steps` | OK |
| `max_grad_norm` | `training.max_grad_norm` | `training.max_grad_norm` | OK |
| `max_steps` | `training.num_steps` | `training.max_steps` | PARTIAL -- named differently; forwarded as `max_steps=` |
| `save_steps` | `training.eval_every` | `training.save_steps` | PARTIAL -- aliased through `eval_every`; forwarded correctly |
| `save_total_limit` | `training.save_total_limit` | `training.save_total_limit` | OK |
| `logging_steps` | `training.logging_steps` | `training.logging_steps` | OK |
| `bf16` | `training.bf16` | `training.bf16` | OK -- forwarded |
| `fp16` | `training.fp16` | `training.fp16` | **FIXED** -- now forwarded to `train_model()` and explicit in `create_training_args` |
| `use_cpu` | `training.use_cpu` | N/A (not in report) | OK -- extra field |
| `report_to` | `environment.report_to` | `training.report_to` | DEVIATION -- placed in `environment` section; forwarded correctly |
| `push_to_hub` | `environment.push_to_hub` | `training.push_to_hub` | **FIXED** -- now forwarded to `train_model()` |
| `hub_model_id` | `environment.hub_model_id` | `training.hub_model_id` | **FIXED** -- now forwarded to `train_model()` |
| `dataloader_num_workers` | `training.dataloader_num_workers` | `training.dataloader_num_workers` | OK |
| `dataloader_pin_memory` | `training.dataloader_pin_memory` | `training.dataloader_pin_memory` | **FIXED** -- now forwarded to `train_model()` |
| `dataloader_persistent_workers` | `training.dataloader_persistent_workers` | `training.dataloader_persistent_workers` | **FIXED** -- now forwarded |
| `dataloader_prefetch_factor` | `training.dataloader_prefetch_factor` | `training.dataloader_prefetch_factor` | **FIXED** -- now forwarded |
| `focal_loss_alpha` | `training.loss_alpha` | `training.focal_loss_alpha` | PARTIAL -- named differently; forwarded correctly |
| `focal_loss_gamma` | `training.loss_gamma` | `training.focal_loss_gamma` | PARTIAL -- named differently; forwarded correctly |
| `focal_loss_prob_margin` | `training.loss_prob_margin` | `training.focal_loss_prob_margin` | PARTIAL -- named differently; forwarded correctly |
| `label_smoothing` | `training.label_smoothing` | `training.label_smoothing` | **FIXED** -- now explicit in `create_training_args` |
| `loss_reduction` | `training.loss_reduction` | `training.loss_reduction` | OK |
| `negatives` | `training.negatives` | `training.negatives` | OK |
| `masking` | `training.masking` | `training.masking` | **FIXED** -- default mismatch resolved (now `'global'`) |
| `seed` | `run.seed` | `experiment.seed` | **FIXED** -- now forwarded to `train_model()` |
| `compile_model` / `torch_compile` | `training.compile_model` | `training.torch_compile` | PARTIAL -- named differently; forwarded correctly |
| `remove_unused_columns` | N/A (was missing) | `training.remove_unused_columns` | **FIXED** -- now explicit in `create_training_args` (default `False`); forwarded by `_launch_training()` |
| `resume_from_checkpoint` | `--resume` CLI flag | `training.resume_from_checkpoint` | **FIXED** -- `train_model()` now accepts and forwards to `trainer.train()` |
| `run_name` | `run.name` | `training.run_name` | **FIXED** -- now forwarded as `run_name=` to `train_model()` |
| `gradient_checkpointing` | N/A (was missing) | `training.gradient_checkpointing` | **FIXED** -- now explicit in `create_training_args` |
| `eval_steps` | `training.eval_every` | `training.eval_steps` | **FIXED** -- now explicit in `create_training_args` |

### Issues Fixed in This Revision

#### 1. Dead Config Fields Removed
**`size_sup`**, **`shuffle_types`**, **`random_drop`** were defined in `_FIELD_SCHEMA` and `template.yaml` but never consumed by any training code. They have been removed entirely.

#### 2. `remove_unused_columns` Set to `False`
HF TrainingArguments defaults `remove_unused_columns=True`, which strips columns not in the model's `forward()` signature. GLiNER uses custom batch dictionaries that require all columns. `create_training_args` now explicitly defaults to `False`, and `_launch_training()` forwards `remove_unused_columns=False`.

#### 3. Hub Fields Now Forwarded
`push_to_hub` and `hub_model_id` from the `environment` config section are now forwarded to `train_model()` and reach `TrainingArguments`. When `push_to_hub: true`, the HF Trainer will push checkpoints to the Hub.

#### 4. `resume_from_checkpoint` Wired
`train_model()` now accepts a `resume_from_checkpoint` parameter (path, or `True` for auto-detect). `_launch_training()` detects the latest checkpoint in the output folder when `--resume` is used and passes it through.

#### 5. Missing Explicit Parameters Added
The following are now explicit named parameters in `create_training_args` (were previously only reachable via `**kwargs`):
- `fp16`, `label_smoothing`, `seed`, `gradient_checkpointing`, `run_name`
- `push_to_hub`, `hub_model_id`
- `eval_steps`, `remove_unused_columns`
- `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor`

#### 6. Masking Default Mismatch Fixed
`create_training_args` previously defaulted `masking='none'` while `TrainingArguments` defaulted to `'global'`. Now both default to `'global'`.

### Remaining Gaps (Not Fixed)

The following fields remain absent from the schema. Most can still be passed via `**kwargs` to `create_training_args` or `train_model()`.

#### Standard TrainingArguments (Medium Priority)

1. **`warmup_steps`** -- integer warmup steps (alternative to `warmup_ratio`)
2. **`num_train_epochs`** -- alternative to `max_steps` for epoch-based training
3. **`evaluation_strategy`** -- not explicit in `create_training_args` (handled dynamically by `train_model()`)
4. **`save_strategy`** -- HF standard (`"steps"`, `"epoch"`, `"no"`)
5. **`overwrite_output_dir`** -- standard TrainingArguments field
6. **`do_train`** / **`do_eval`** -- script convention flags
7. **`eval_delay`** -- delay first evaluation
8. **`eval_on_start`** -- evaluate before training
9. **`auto_find_batch_size`** -- automatic batch size finder
10. **`group_by_length`** -- group samples by length

#### Optimizer Details (Low Priority)

11. **`adam_beta1`**, **`adam_beta2`**, **`adam_epsilon`** -- passable via `**kwargs`

#### Precision and Performance (Low Priority)

12. **`tf32`** -- TF32 precision mode (passable via `**kwargs`)

#### Logging Details (Low Priority)

13. **`logging_strategy`**, **`logging_first_step`**, **`logging_dir`**, **`disable_tqdm`**

#### Hub Details (Low Priority)

14. **`hub_strategy`**, **`hub_private_repo`**, **`hub_always_push`**, **`hub_revision`**

#### Resume and Determinism (Low Priority)

15. **`ignore_data_skip`**, **`full_determinism`**, **`data_seed`**

#### Model Selection (Medium Priority)

16. **`load_best_model_at_end`**, **`metric_for_best_model`**, **`greater_is_better`**

#### Distributed Training (Low Priority)

17. **`deepspeed`**, **`fsdp`**, **`fsdp_config`**, **`ddp_backend`**, **`ddp_timeout`**, **`local_rank`**

### Deviations from Report

#### 1. Section Structure Mismatch

The report recommends a **four-section** layout: `model:`, `training:`, `data:`, `peft:`. Our implementation uses a **six-section** layout: `run:`, `model:`, `data:`, `training:`, `lora:`, `environment:`.

Key differences:
- **`run:` section**: Report puts `name`, `seed`, `tags` under `experiment:`. We use `run:`. Cosmetic difference.
- **`environment:` section**: Hub and W&B fields are segregated into `environment:` rather than `training:`. This is a structural deviation but functionally equivalent since all fields are now forwarded.
- **`lora:` vs `peft:`**: Report uses `peft:` with a nested `lora:` sub-section.

#### 2. Field Naming Inconsistencies

Our CLI uses GLiNER-legacy field names instead of standard HF TrainingArguments names:

| Our Name | HF Standard Name | Notes |
|---|---|---|
| `training.num_steps` | `max_steps` | Legacy GLiNER naming |
| `training.train_batch_size` | `per_device_train_batch_size` | Legacy GLiNER naming |
| `training.eval_batch_size` | `per_device_eval_batch_size` | Legacy GLiNER naming |
| `training.lr_encoder` | `learning_rate` | GLiNER-specific but maps correctly |
| `training.scheduler_type` | `lr_scheduler_type` | Shortened name |
| `training.eval_every` | `save_steps` + `eval_steps` | Conflates save and eval intervals |
| `training.loss_alpha` | `focal_loss_alpha` | Shortened name |
| `training.optimizer` | `optim` | Different name |
| `training.compile_model` | `torch_compile` | Different name |

#### 3. `eval_every` Conflates Two Distinct HF Concepts

Our `training.eval_every` is used for both `save_steps` and `eval_steps`. The report recommends separate fields. In practice, `_launch_training()` uses `eval_every` for both, and `train_model()` auto-enables evaluation at the `save_steps` cadence when an eval dataset is provided.

#### 4. `report_to` Type

Our schema declares `report_to` as `str`. HF TrainingArguments also accepts `List[str]`. The `create_training_args` signature now uses `Union[str, list]` but the YAML schema still types it as `str`.

### Test Coverage Summary

After fixes: **185 tests pass, 12 xfail** across 4 test files:
- `tests/test_training_validation.py` -- validates `create_training_args`, `TrainingArguments`, `Trainer` wiring
- `tests/test_config_forwarding.py` -- validates `create_training_args` signature and `train.py` forwarding
- `tests/test_validator_integration.py` -- cross-validates `training_cli.py` and `config_cli.py` integration
- `ptbr/tests/test_training_cli.py` -- validates `_launch_training` forwarding, schema validation, semantic checks

Remaining xfail tests document:
- `evaluation_strategy` not explicit in `create_training_args` (handled dynamically)
- `size_sup`/`shuffle_types`/`random_drop` consumers in legacy `config.yaml` files
- `label_smoothing` collision with HF's `label_smoothing_factor`
- Legacy `train.py` gaps (fp16, eval_steps, logging_steps conflation)
