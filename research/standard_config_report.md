## Standard HuggingFace Configuration Report

### Executive Summary

Our implementation (`training_cli.py` + `template.yaml`) covers the core GLiNER-specific and basic HuggingFace TrainingArguments fields but **falls significantly short** of the deep research report's recommendations for Hub integration, W&B tracking, distributed training, model selection, resume handling, and several standard TrainingArguments knobs. Approximately 35+ recommended fields are either entirely absent or placed in non-standard locations, and multiple forwarding gaps exist where our CLI validates a field but never passes it through to the upstream `train_model()` call.

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
| `bf16` | `training.bf16` | `training.bf16` | OK |
| `fp16` | `training.fp16` | `training.fp16` | PARTIAL -- validated in schema but **not forwarded** to `train_model()` as `fp16=` kwarg |
| `use_cpu` | `training.use_cpu` | N/A (not in report) | OK -- extra field |
| `report_to` | `environment.report_to` | `training.report_to` | DEVIATION -- placed in `environment` section, not `training` |
| `push_to_hub` | `environment.push_to_hub` | `training.push_to_hub` | DEVIATION -- placed in `environment` section, not `training` |
| `hub_model_id` | `environment.hub_model_id` | `training.hub_model_id` | DEVIATION -- placed in `environment` section, not `training` |
| `dataloader_num_workers` | `training.dataloader_num_workers` | `training.dataloader_num_workers` | OK |
| `dataloader_pin_memory` | `training.dataloader_pin_memory` | `training.dataloader_pin_memory` | OK -- but **not forwarded** to `train_model()` |
| `dataloader_persistent_workers` | `training.dataloader_persistent_workers` | `training.dataloader_persistent_workers` | OK -- but **not forwarded** |
| `dataloader_prefetch_factor` | `training.dataloader_prefetch_factor` | `training.dataloader_prefetch_factor` | OK -- but **not forwarded** |
| `focal_loss_alpha` | `training.loss_alpha` | `training.focal_loss_alpha` | PARTIAL -- named differently; forwarded correctly |
| `focal_loss_gamma` | `training.loss_gamma` | `training.focal_loss_gamma` | PARTIAL -- named differently; forwarded correctly |
| `focal_loss_prob_margin` | `training.loss_prob_margin` | `training.focal_loss_prob_margin` | PARTIAL -- named differently; forwarded correctly |
| `label_smoothing` | `training.label_smoothing` | `training.label_smoothing` | OK |
| `loss_reduction` | `training.loss_reduction` | `training.loss_reduction` | OK |
| `negatives` | `training.negatives` | `training.negatives` | OK |
| `masking` | `training.masking` | `training.masking` | OK |
| `seed` | `run.seed` | `experiment.seed` | PARTIAL -- different section name but functional |
| `compile_model` / `torch_compile` | `training.compile_model` | `training.torch_compile` | PARTIAL -- named differently; forwarded correctly |

### Missing Fields (Critical Gaps)

The following fields are recommended by the deep research report but are **completely absent** from both `_FIELD_SCHEMA` in `training_cli.py` and `template.yaml`:

#### Standard TrainingArguments (High Priority)

1. **`warmup_steps`** -- integer warmup steps (report says: include and set to `0` explicitly when using `warmup_ratio`)
2. **`num_train_epochs`** -- alternative to `max_steps` for epoch-based training
3. **`evaluation_strategy`** -- HF standard field (`"steps"`, `"epoch"`, `"no"`)
4. **`eval_steps`** -- distinct from `save_steps` (we alias both to `eval_every`)
5. **`save_strategy`** -- HF standard (`"steps"`, `"epoch"`, `"no"`)
6. **`overwrite_output_dir`** -- standard TrainingArguments field
7. **`save_safetensors`** -- controls safetensors serialization format
8. **`do_train`** / **`do_eval`** -- script convention flags
9. **`eval_delay`** -- number of steps/epochs to delay first evaluation
10. **`eval_on_start`** -- evaluate before training begins
11. **`auto_find_batch_size`** -- automatic batch size finder
12. **`group_by_length`** -- group samples by length for efficiency
13. **`length_column_name`** -- column name for length grouping
14. **`remove_unused_columns`** -- critical for custom batch dictionaries (report notes GLiNER needs `false`)

#### Optimizer Details (Medium Priority)

15. **`adam_beta1`** -- Adam beta1 parameter (default 0.9)
16. **`adam_beta2`** -- Adam beta2 parameter (default 0.999)
17. **`adam_epsilon`** -- Adam epsilon (default 1e-8)

#### Precision and Performance (Medium Priority)

18. **`tf32`** -- TF32 precision mode
19. **`gradient_checkpointing`** -- memory optimization
20. **`torch_compile`** (as standard HF name) -- we use `compile_model` instead

#### Logging Details (Medium Priority)

21. **`logging_strategy`** -- `"steps"` / `"epoch"` / `"no"`
22. **`logging_first_step`** -- log metrics on first step
23. **`logging_dir`** -- TensorBoard log directory
24. **`disable_tqdm`** -- disable progress bar
25. **`log_level`** -- Trainer log level
26. **`log_level_replica`** -- log level for replicas
27. **`run_name`** -- W&B run name (standard TrainingArguments field)

#### Dataloader Details (Low Priority)

28. **`dataloader_drop_last`** -- drop last incomplete batch
29. **`skip_memory_metrics`** -- skip memory profiling
30. **`save_on_each_node`** -- save on each distributed node

#### Resume and Determinism (High Priority)

31. **`resume_from_checkpoint`** -- standard HF checkpoint resume path
32. **`ignore_data_skip`** -- skip data fast-forward on resume
33. **`full_determinism`** -- enforce full deterministic training

#### Model Selection (High Priority)

34. **`load_best_model_at_end`** -- load best checkpoint at end of training
35. **`metric_for_best_model`** -- metric to determine best model
36. **`greater_is_better`** -- whether higher metric is better

#### Distributed Training (Medium Priority)

37. **`deepspeed`** -- DeepSpeed config path
38. **`fsdp`** -- FSDP sharding strategy
39. **`fsdp_config`** -- FSDP configuration dict
40. **`ddp_backend`** -- DDP communication backend
41. **`ddp_timeout`** -- DDP timeout
42. **`local_rank`** -- local rank for distributed

### Deviations from Report

#### 1. Section Structure Mismatch

The report recommends a **four-section** layout: `model:`, `training:`, `data:`, `peft:` (with an optional `logging:` or `experiment:` section). Our implementation uses a **six-section** layout: `run:`, `model:`, `data:`, `training:`, `lora:`, `environment:`.

Key differences:
- **`run:` section**: Report puts `name`, `seed`, `tags` under `experiment:`. We use `run:`. This is cosmetic but inconsistent.
- **`environment:` section**: Report puts Hub and W&B fields under `training:` (since they are standard `TrainingArguments` fields). We segregate them into `environment:`. This is a **structural deviation** -- `push_to_hub`, `hub_model_id`, `report_to`, and `run_name` are all standard `TrainingArguments` parameters and should live under `training:` per HF convention.
- **`lora:` vs `peft:`**: Report uses `peft:` with a nested `lora:` sub-section. We use a flat `lora:` section. Our approach lacks the `method`, `adapter_name`, `init_lora_weights`, and `use_rslora` fields.

#### 2. Field Naming Inconsistencies

Our CLI uses GLiNER-legacy field names instead of standard HF TrainingArguments names:

| Our Name | HF Standard Name | Notes |
|---|---|---|
| `training.num_steps` | `max_steps` | Legacy GLiNER naming |
| `training.train_batch_size` | `per_device_train_batch_size` | Legacy GLiNER naming |
| `training.eval_batch_size` | `per_device_eval_batch_size` | Legacy GLiNER naming |
| `training.lr_encoder` | `learning_rate` | GLiNER-specific but maps correctly |
| `training.lr_others` | `others_lr` | GLiNER extension |
| `training.scheduler_type` | `lr_scheduler_type` | Shortened name |
| `training.eval_every` | `save_steps` + `eval_steps` | Conflates save and eval intervals |
| `training.loss_alpha` | `focal_loss_alpha` | Shortened name |
| `training.loss_gamma` | `focal_loss_gamma` | Shortened name |
| `training.loss_prob_margin` | `focal_loss_prob_margin` | Shortened name |
| `training.optimizer` | `optim` | Different name |
| `training.compile_model` | `torch_compile` | Different name |

This creates a translation layer that makes it harder to reason about which HF TrainingArguments are actually being set.

#### 3. `eval_every` Conflates Two Distinct HF Concepts

Our `training.eval_every` is used for both `save_steps` and (implicitly) `eval_steps`. The report recommends separate `save_steps`, `eval_steps`, `save_strategy`, and `evaluation_strategy` fields, since in practice you often want to evaluate more frequently than you save checkpoints.

#### 4. `report_to` Type Mismatch

The report shows `report_to: ["wandb"]` (a list), matching HF TrainingArguments which accepts `List[str]` or `str`. Our schema declares `report_to` as `str` type, which means you cannot specify multiple backends like `["wandb", "tensorboard"]`.

#### 5. Forwarding Gaps in `_launch_training()`

Several fields are validated in the schema but **never forwarded** to `model.train_model()`:

- `training.fp16` -- validated but not passed (line 1123 passes `bf16=` but the `fp16=` kwarg is missing from the call at line 1090-1132)
- `training.dataloader_pin_memory` -- validated but not passed
- `training.dataloader_persistent_workers` -- validated but not passed
- `training.dataloader_prefetch_factor` -- validated but not passed
- `environment.push_to_hub` -- validated but not passed to TrainingArguments
- `environment.hub_model_id` -- validated but not passed to TrainingArguments
- `run.name` -- validated but not forwarded as `run_name` to TrainingArguments (which would set the W&B run name)

These represent "dead" configuration: the user sets them, validation passes, but the actual training run ignores them.

### Hub Integration Gaps

The report identifies the following standard `TrainingArguments` Hub fields as essential for professional workflows:

| Field | Report Recommendation | Our Implementation | Gap |
|---|---|---|---|
| `push_to_hub` | `training.push_to_hub` | `environment.push_to_hub` -- validated but **never forwarded** to TrainingArguments | CRITICAL: field exists but is dead weight; push never actually happens via Trainer |
| `hub_model_id` | `training.hub_model_id` | `environment.hub_model_id` -- validated but **never forwarded** | CRITICAL: same as above |
| `hub_strategy` | `training.hub_strategy` (`"every_save"`, `"end"`, etc.) | **MISSING** | CRITICAL: no way to control when models are pushed |
| `hub_token` | `training.hub_token` | `environment.hf_token` -- validated for connectivity check but **never forwarded** to TrainingArguments | CRITICAL: Trainer cannot authenticate to Hub |
| `hub_private_repo` | `training.hub_private_repo` | **MISSING** | No way to create private Hub repos |
| `hub_always_push` | `training.hub_always_push` | **MISSING** | No way to force push on every save |
| `hub_revision` | `training.hub_revision` | **MISSING** | No way to push to a specific branch/revision |

**Net effect**: Even if a user sets `push_to_hub: true` and `hub_model_id: "org/model"`, the model is **never actually pushed** because these values are only used for a connectivity check in `check_huggingface()` but are never passed through to the HF Trainer's `TrainingArguments`. The user gets a false sense of security from the validation passing.

### W&B Gaps

| Feature | Report Recommendation | Our Implementation | Gap |
|---|---|---|---|
| `report_to` | `training.report_to: ["wandb"]` (list) | `environment.report_to: "none"` (string) -- forwarded to `train_model()` | Type mismatch (str vs list); placed in wrong section |
| `run_name` | `training.run_name` (standard TrainingArguments) | **MISSING** -- `run.name` exists but is never forwarded as `run_name=` to TrainingArguments | W&B runs get auto-generated names instead of user-specified names |
| `WANDB_PROJECT` | Set via env var | `environment.wandb_project` -- forwarded to `os.environ["WANDB_PROJECT"]` | OK -- functional |
| `WANDB_ENTITY` | Set via env var | `environment.wandb_entity` -- forwarded to `os.environ["WANDB_ENTITY"]` | OK -- functional |
| `WANDB_API_KEY` | Set via env var | `environment.wandb_api_key` -- forwarded to `os.environ["WANDB_API_KEY"]` | OK -- functional |
| `WANDB_LOG_MODEL` | `logging.wandb.log_model: "checkpoint"` | **MISSING** | No model artifact logging to W&B |
| `WANDB_MODE` | `logging.wandb.mode` (`"online"`, `"offline"`, `"disabled"`) | **MISSING** | Cannot control W&B mode |
| `WANDB_GROUP` | `logging.wandb.group` | **MISSING** | Cannot group related runs |
| `WANDB_JOB_TYPE` | `logging.wandb.job_type` | **MISSING** | Cannot tag job type |
| `logging_first_step` | `training.logging_first_step: true` | **MISSING** | First step metrics not guaranteed to be logged |
| `logging_strategy` | `training.logging_strategy: "steps"` | **MISSING** | Implicit "steps" only |

**Net effect**: W&B runs will have auto-generated names (not the user's `run.name`), no model artifacts logged, and limited organizational metadata. The `run.name` field is validated and displayed in the summary table but never reaches HF Trainer.

### warmup_ratio vs warmup_steps

**Report finding**: The W&B dump showed `warmup_steps: 0.05` (a float), which is a red flag. HF TrainingArguments defines `warmup_steps` as an **integer** (number of steps) and `warmup_ratio` as a **float** (fraction of total steps). If both are set, `warmup_steps > 0` overrides `warmup_ratio`.

**Our implementation**:
- `training.warmup_ratio` is present, typed as `float`, bounded to `[0.0, 1.0]` -- **CORRECT**.
- `training.warmup_steps` (integer) is **ABSENT**.
- `warmup_ratio` is forwarded to `train_model()` at line 1099 of `training_cli.py` -- **CORRECT**.

**Assessment**: Our handling is **partially correct** but incomplete. We correctly use `warmup_ratio` as a float and forward it. However:
1. We do not expose `warmup_steps` at all, so users cannot override with an explicit integer step count.
2. We do not explicitly set `warmup_steps=0` in the TrainingArguments call, relying on the HF default (which is `0`, so this works in practice).
3. There is no cross-validation to prevent a user from somehow passing both (though since `warmup_steps` is not in our schema, this is unlikely).

**Verdict**: Functionally correct for the `warmup_ratio`-only path. Missing the `warmup_steps` escape hatch recommended by the report.

### Professional Extras Missing

| Feature | Report Section | Status in Our Code | Impact |
|---|---|---|---|
| `resume_from_checkpoint` | Resume/determinism | **MISSING** from schema and forwarding. Our `--resume` CLI flag does a name-matching check on checkpoint dirs but never actually sets `resume_from_checkpoint` in TrainingArguments | Resume does not actually resume training; it only validates checkpoint existence |
| `full_determinism` | Resume/determinism | **MISSING** | Cannot guarantee bitwise reproducibility |
| `load_best_model_at_end` | Model selection | **MISSING** | After training, the last checkpoint is loaded regardless of performance |
| `metric_for_best_model` | Model selection | **MISSING** | No metric-based model selection |
| `greater_is_better` | Model selection | **MISSING** | Required companion to `metric_for_best_model` |
| `deepspeed` | Distributed | **MISSING** | Cannot use DeepSpeed ZeRO stages |
| `fsdp` | Distributed | **MISSING** | Cannot use PyTorch FSDP |
| `fsdp_config` | Distributed | **MISSING** | No FSDP configuration |
| `ddp_backend` | Distributed | **MISSING** | Cannot specify NCCL/Gloo backend |
| `ddp_timeout` | Distributed | **MISSING** | No DDP timeout control |
| `local_rank` | Distributed | **MISSING** | No explicit local rank |
| `gradient_checkpointing` | Precision/perf | **MISSING** | Cannot trade compute for memory |
| `tf32` | Precision/perf | **MISSING** | Cannot enable TF32 on Ampere+ GPUs |
| `data_seed` | Determinism | **MISSING** | Cannot independently seed data shuffling |
| `save_safetensors` | Saving | **MISSING** | No control over serialization format |

### Concrete Recommendations

1. **Move Hub and W&B fields from `environment:` to `training:`** (`training_cli.py:194-201`, `template.yaml:496-540`). Fields `push_to_hub`, `hub_model_id`, `hub_strategy`, `hub_token`, `hub_private_repo`, `hub_always_push`, `report_to`, and `run_name` are all standard `TrainingArguments` parameters. Keep `environment:` only for script-level env vars (`cuda_visible_devices`, `wandb_api_key` override).

2. **Forward Hub fields to `train_model()`** (`training_cli.py:1090-1132`). Add `push_to_hub=`, `hub_model_id=`, `hub_strategy=`, `hub_token=`, `hub_private_repo=`, `hub_always_push=` kwargs to the `model.train_model()` call. Currently these are validated but silently discarded.

3. **Forward `run_name` to TrainingArguments** (`training_cli.py:1090-1132`). Add `run_name=cfg["run"]["name"]` to the `train_model()` call so W&B runs get the user's chosen name.

4. **Forward missing dataloader and precision fields** (`training_cli.py:1126-1128`). Add `fp16=`, `dataloader_pin_memory=`, `dataloader_persistent_workers=`, `dataloader_prefetch_factor=` to the `train_model()` call. These are validated but never passed through.

5. **Add `hub_strategy`, `hub_private_repo`, `hub_always_push`, `hub_revision` to schema** (`training_cli.py:92-202`). Add to `_FIELD_SCHEMA` with appropriate types and defaults. Add corresponding entries to `template.yaml`.

6. **Add `WANDB_LOG_MODEL` support** (`training_cli.py:1029-1040`). In the W&B env var setup block, add `os.environ["WANDB_LOG_MODEL"] = cfg["environment"].get("wandb_log_model", "false")` or similar. Add `environment.wandb_log_model` to the schema.

7. **Add `resume_from_checkpoint` and actually wire resume** (`training_cli.py:997-1134`). The current `--resume` flag checks for checkpoint existence but never passes the checkpoint path to `train_model()`. Add `resume_from_checkpoint=str(latest_checkpoint)` to the `train_model()` kwargs, or add a `training.resume_from_checkpoint` schema field.

8. **Separate `eval_steps` from `save_steps`** (`training_cli.py:167`, `template.yaml:373`). Replace the single `eval_every` with distinct `save_steps` and `eval_steps` fields (both defaulting to the same value for backward compatibility). Add `evaluation_strategy` and `save_strategy` fields.

9. **Add model selection fields** (`training_cli.py:92-202`). Add `training.load_best_model_at_end` (bool, default `false`), `training.metric_for_best_model` (str, default `null`), `training.greater_is_better` (bool, default `null`) to the schema and forward them.

10. **Add distributed training fields** (`training_cli.py:92-202`). Add `training.deepspeed` (str/null), `training.fsdp` (list), `training.fsdp_config` (dict), `training.ddp_backend` (str/null), `training.ddp_timeout` (int) as placeholder fields even if not actively used, to prepare for scaling.

11. **Add `gradient_checkpointing` and `tf32`** (`training_cli.py:92-202`). Add to schema and forward to `train_model()`. These are high-impact performance knobs.

12. **Add `warmup_steps` as an explicit integer field** (`training_cli.py:150`). Add `training.warmup_steps` (int, default `0`) and add a semantic cross-check: if both `warmup_ratio > 0` and `warmup_steps > 0`, emit a warning that `warmup_steps` will take precedence.

13. **Change `report_to` type from `str` to `list|str`** (`training_cli.py:197`). HF TrainingArguments accepts `List[str]` for `report_to`. Update the schema type and validation.

14. **Add `adam_beta1`, `adam_beta2`, `adam_epsilon`** (`training_cli.py:92-202`). These are standard optimizer hyperparameters that should be exposed for reproducibility.

15. **Add LoRA `init_lora_weights` and `use_rslora`** (`training_cli.py:184-191`). The report recommends these PEFT fields. Also consider renaming the section from `lora:` to `peft:` with a nested `lora:` sub-section for future extensibility (e.g., adding `method: "lora"` or `method: "qlora"`).

16. **Add `data_seed` field** (`training_cli.py:92-98`). Add under `run:` section (alongside `seed`) for independent data shuffling seed control.

17. **Rename fields to match HF conventions** (optional but strongly recommended). Consider adding aliases or renaming `num_steps` -> `max_steps`, `train_batch_size` -> `per_device_train_batch_size`, `scheduler_type` -> `lr_scheduler_type`, etc. This reduces cognitive overhead when cross-referencing with HF documentation.

18. **Add `logging_first_step: true` and `logging_strategy`** (`training_cli.py:92-202`). These ensure W&B gets metrics from step 0, which is valuable for debugging learning rate warmup and initial loss values.

19. **Forward `save_safetensors=True` to TrainingArguments** (`training_cli.py:1090-1132`). The upstream Trainer `_save()` method at `gliner/training/trainer.py:107` checks `self.args.save_safetensors`, but we never set this argument. Default should be `true` for modern workflows.

20. **Add `remove_unused_columns: false` to TrainingArguments** (`training_cli.py:1090-1132`). The report explicitly notes this is essential for GLiNER's custom batch dictionaries. Without it, HF Trainer may silently drop columns needed by GLiNER's `compute_loss()`.

---

## Post-Fix Assessment (2026-02-18)

### Changes Applied Since This Report Was Written

The following fixes have been applied to the codebase, addressing several of the critical gaps identified above. This section documents what was fixed, what tests verify the fixes, and what remains open.

#### Fixes in `training_cli.py`

| Issue | Status | Verification |
|---|---|---|
| `fp16` validated but not forwarded to `train_model()` | **FIXED** — now forwarded as `fp16=train_cfg.get("fp16", False)` | `test_training_cli.py::TestLaunchTrainingPropagation`, `test_validator_integration.py::TestParameterForwardingFixes::test_fp16_is_forwarded` |
| `dataloader_pin_memory` validated but not forwarded | **FIXED** — now forwarded | `test_validator_integration.py::TestParameterForwardingFixes::test_dataloader_pin_memory_is_forwarded` |
| `dataloader_persistent_workers` validated but not forwarded | **FIXED** — now forwarded | `test_validator_integration.py::TestParameterForwardingFixes::test_dataloader_persistent_workers_is_forwarded` |
| `dataloader_prefetch_factor` validated but not forwarded | **FIXED** — now forwarded | `test_validator_integration.py::TestParameterForwardingFixes::test_dataloader_prefetch_factor_is_forwarded` |
| `run.name` validated but never forwarded as `run_name` | **FIXED** — now forwarded as `run_name=cfg["run"]["name"]` | `test_validator_integration.py::TestParameterForwardingFixes::test_run_name_is_forwarded` |
| `report_to` not forwarded to `train_model()` | **FIXED** — now forwarded | `test_validator_integration.py::TestParameterForwardingFixes::test_report_to_is_forwarded` |
| `eval_strategy` / `eval_steps` not set when eval data available | **FIXED** — now conditionally set | `test_validator_integration.py::TestParameterForwardingFixes::test_eval_strategy_is_forwarded` |

#### Fixes in `train.py`

| Issue | Status | Verification |
|---|---|---|
| `output_dir` hardcoded to `"models"` | **FIXED** — now reads from `cfg.data.root_dir` | `test_validator_integration.py::TestTrainPyFixes::test_output_dir_reads_from_config` |
| `bf16` hardcoded to `True` | **FIXED** — now reads from `getattr(cfg.training, "bf16", False)` | `test_validator_integration.py::TestTrainPyFixes::test_bf16_reads_from_config` |
| `per_device_eval_batch_size` reused `train_batch_size` | **FIXED** — now cascades `eval_batch_size` → `train_batch_size` | `test_validator_integration.py::TestTrainPyFixes::test_eval_batch_size_uses_separate_variable` |
| `label_smoothing` not forwarded | **FIXED** — now forwarded | `test_validator_integration.py::TestTrainPyFixes::test_label_smoothing_is_forwarded` |

#### Fixes in `__main__.py`

| Issue | Status | Verification |
|---|---|---|
| `training_cli` imported at module level (side effects on config/data subcommands) | **FIXED** — now lazy-loaded | `test_validator_integration.py::TestMainImportSideEffects::test_training_cli_not_imported_at_module_level` |

#### Fixes in `config_cli.py`

| Issue | Status | Verification |
|---|---|---|
| `model:` not accepted as alias for `gliner_config:` | **FIXED** — config_cli now accepts `model:` as alias | `test_config_cli_aliases.py` |
| `lora:` not accepted as alias for `lora_config:` | **FIXED** — config_cli now accepts `lora:` as alias | `test_config_cli_aliases.py` |

### Test Suite Status

| Test File | Result | Purpose |
|---|---|---|
| `ptbr/tests/test_training_cli.py` | **62/62 passed** | Core validation, semantic checks, CLI, resume, LoRA, launch propagation |
| `tests/test_validator_integration.py` | **36/36 passed, 6 skipped** | Cross-CLI integration, forwarding verification, fix confirmation |
| `tests/test_trainer_column_pruning.py` | **3/3 passed** | Custom trainer column handling |
| `tests/test_config_propagation.py` | **3 skipped** | Config propagation (requires torch) |
| `ptbr/tests/test_main_cli.py` | **passed** | __main__.py subcommand tests |
| `ptbr/tests/test_validation.py` | **passed** | YAML validation edge cases |
| `ptbr/tests/test_config_cli_aliases.py` | **passed** | model:/lora: alias acceptance |
| `ptbr/tests/test_train_py.py` | **passed** | train.py forwarding tests |

The 6 skipped tests in `test_validator_integration.py` require `torch`/`gliner` (heavy DL dependencies) and are guarded with `pytest.importorskip`. They will run in environments with full dependencies installed.

### Remaining Open Issues

The following issues from the original report have **not** been addressed:

#### Still-Open Forwarding Gaps

1. `size_sup`, `shuffle_types`, `random_drop` — validated in schema but never forwarded (dead config). Verified by `TestParameterForwardingRemainingGaps`.
2. `run.tags`, `run.description` — validated but not forwarded. Verified by `TestParameterForwardingRemainingGaps`.
3. `remove_unused_columns` — not set to `False` in either `training_cli.py` or `train.py`. This remains dangerous for GLiNER's custom collators. Verified by `TestRemoveUnusedColumns`.
4. `push_to_hub`, `hub_model_id` — still in `environment:` section, still not forwarded to `TrainingArguments`.
5. `resume_from_checkpoint` — `--resume` flag validates checkpoint existence but never passes path to `TrainingArguments`.

#### Still-Missing Fields

All 42 missing fields listed in the "Missing Fields" section above remain absent. The highest-priority gaps are:

- `warmup_steps` (integer escape hatch)
- `gradient_checkpointing` (memory optimization)
- `resume_from_checkpoint` (actual resume wiring)
- `load_best_model_at_end` + `metric_for_best_model` (model selection)
- `hub_strategy` and related Hub fields (push automation)
- `WANDB_LOG_MODEL`, `WANDB_MODE`, `WANDB_GROUP` (W&B metadata)
- `adam_beta1`, `adam_beta2`, `adam_epsilon` (optimizer reproducibility)

#### Still-Open Structural Issues

- YAML schema incompatibility between `config_cli` and `training_cli` persists (different top-level keys, different LoRA section names). Aliases partially mitigate this.
- CLI argument style inconsistency (`--file` vs positional) persists.
- `eval_every` still conflates `save_steps` and `eval_steps`.
- `report_to` is still typed as `str` not `list|str`.
- Hub/W&B fields remain in `environment:` instead of `training:`.
