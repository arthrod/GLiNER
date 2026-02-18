## Training Parameters and Loss Configuration Report

### Executive Summary

This report was originally drafted before several key fixes were applied. After reviewing the current codebase (commit `5921772`, branch `claude/prevent-cuda-init-Emjw4`), running static analysis (`py_compile` on all key modules), and executing the full `test_training_validation.py` test suite (20 passed, 22 xfailed), the findings below reflect the **current state** of the code.

**Fixes applied since the original report:**
- `_launch_training` now forwards `eval_strategy="steps"` and `eval_steps` when an eval dataset is present (line 1127).
- `train_model()` auto-enables evaluation when `eval_dataset` is provided but `eval_strategy` is `"no"` (model.py lines 1119-1130).
- `_launch_training` now forwards `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor` (lines 1135-1137).
- `_launch_training` now forwards `run_name=cfg["run"]["name"]` (line 1140).
- `_launch_training` now forwards `fp16` (line 1131).
- `_launch_training` now forwards `gradient_accumulation_steps` (line 1142).
- `train.py` also forwards `eval_strategy` and `eval_steps` (line 100).

**Remaining critical gaps:**
- `remove_unused_columns` is **never set to `False`**. HF defaults to `True`, which silently drops custom batch keys. This is the highest-priority remaining bug.
- `run.seed` is only used for `torch.manual_seed()`, never forwarded as `seed` to TrainingArguments.
- `label_smoothing` passes through `**kwargs` only (not explicit in `create_training_args`).
- Three dead fields (`size_sup`, `shuffle_types`, `random_drop`) remain in `_FIELD_SCHEMA` with no consumer.
- Three LoRA fields (`init_lora_weights`, `use_rslora`, `fan_in_fan_out`) are missing from `_apply_lora`.

---

### Static Analysis Results

All key modules compile cleanly:

| File | `py_compile` | Notes |
|---|---|---|
| `gliner/training/trainer.py` | OK | 407 lines, no syntax errors |
| `gliner/model.py` | OK | 3409 lines, no syntax errors |
| `ptbr/training_cli.py` | OK | ~1200 lines, no syntax errors |
| `train.py` | OK | 115 lines, no syntax errors |

No import errors detected at module level (tested via `pytest --collect-only`).

---

### Test Suite Assessment

**File**: `tests/test_training_validation.py` (42 tests)
**Run**: `python -m pytest tests/test_training_validation.py -v --tb=short`
**Result**: 20 passed, 22 xfailed, 0 failures

#### Tests That MUST Pass (20/20 passing)

| # | Test | Status | What It Validates |
|---|---|---|---|
| 1 | `TestSmoke::test_training_args_import` | PASS | Import works |
| 2 | `TestSmoke::test_trainer_import` | PASS | Import works |
| 3 | `TestSmoke::test_training_args_instantiation` | PASS | Can create TA with defaults |
| 4 | `TestSmoke::test_create_training_args_callable` | PASS | Factory method exists |
| 5 | `TestSmoke::test_create_training_args_returns_training_args` | PASS | Returns correct type |
| 6 | `TestTrainingArgumentsDefaults::test_masking_default_is_global` | PASS | TA.masking defaults to "global" |
| 7 | `TestTrainingArgumentsDefaults::test_label_smoothing_field_exists` | PASS | Field declared |
| 8 | `TestTrainingArgumentsDefaults::test_loss_reduction_default_is_sum` | PASS | Correct default |
| 9 | `TestTrainingArgumentsDefaults::test_focal_loss_defaults` | PASS | alpha=-1, gamma=0, prob_margin=0 |
| 10 | `TestTrainingArgumentsFieldCompleteness::test_all_gliner_fields_declared` | PASS | All 9 custom fields exist |
| 11 | `TestTrainingArgumentsFieldCompleteness::test_gliner_fields_have_defaults` | PASS | All have defaults |
| 12 | `TestComputeLossWiring::test_compute_loss_passes_all_loss_params` | PASS | 7 loss params forwarded |
| 13 | `TestTrainerDataloaderWiring::test_get_train_dataloader_uses_pin_memory` | PASS | Custom dataloader correct |
| 14 | `TestTrainerDataloaderWiring::test_get_train_dataloader_uses_persistent_workers` | PASS | Custom dataloader correct |
| 15 | `TestTrainerDataloaderWiring::test_get_train_dataloader_uses_prefetch_factor` | PASS | Custom dataloader correct |
| 16 | `TestKwargsPassThrough::test_label_smoothing_via_kwargs_reaches_training_args` | PASS | kwargs passthrough works |
| 17 | `TestKwargsPassThrough::test_fp16_via_kwargs_reaches_training_args` | PASS | kwargs passthrough works |
| 18 | `TestKwargsPassThrough::test_seed_via_kwargs_reaches_training_args` | PASS | kwargs passthrough works |
| 19 | `TestHFDefaults::test_hf_remove_unused_columns_defaults_to_true` | PASS | Documents the risk |
| 20 | `TestHFDefaults::test_hf_label_smoothing_factor_defaults_to_zero` | PASS | No double smoothing by default |

#### Known-Gap Tests (22/22 xfailing as expected)

Each `xfail(strict=True)` test documents a specific gap. When the gap is fixed, the test will unexpectedly pass (XPASS) and CI will flag it, forcing the marker to be removed. This is the correct pattern.

**Do the xfail tests correctly identify real issues?** Analysis per test:

| # | Test | xfail reason | Verified still accurate? |
|---|---|---|---|
| 1 | `test_label_smoothing_is_explicit` | label_smoothing not in create_training_args signature | **YES** — still in `**kwargs` only |
| 2 | `test_fp16_is_explicit` | fp16 not explicit, only bf16 | **YES** — fp16 still implicit |
| 3 | `test_seed_is_explicit` | seed not explicit | **YES** — seed still missing |
| 4 | `test_gradient_checkpointing_is_explicit` | gradient_checkpointing not explicit | **YES** — still missing entirely |
| 5 | `test_run_name_is_explicit` | run_name not explicit | **YES** — still implicit (passes via kwargs from CLI) |
| 6 | `test_push_to_hub_is_explicit` | push_to_hub not explicit | **YES** — still missing |
| 7 | `test_hub_model_id_is_explicit` | hub_model_id not explicit | **YES** — still missing |
| 8 | `test_evaluation_strategy_is_explicit` | eval_strategy not explicit | **YES** — but mitigated at runtime in `train_model()` |
| 9 | `test_eval_steps_is_explicit` | eval_steps not explicit | **YES** — but mitigated at runtime in `train_model()` |
| 10 | `test_create_training_args_masking_matches_training_args_default` | masking: "none" vs "global" | **YES** — mismatch persists |
| 11 | `test_create_training_args_sets_remove_unused_columns_false` | remove_unused_columns not set | **YES** — CRITICAL, still True |
| 12 | `test_train_model_source_references_remove_unused_columns` | train_model never mentions it | **YES** — CRITICAL, completely absent |
| 13 | `test_create_training_args_enables_evaluation` | eval_strategy not set by create_training_args | **YES** — but mitigated by `train_model()` runtime logic |
| 14 | `test_create_training_args_forwards_eval_steps` | eval_steps not set | **YES** — but mitigated by `train_model()` runtime logic |
| 15 | `test_size_sup_is_consumed` | size_sup dead | **YES** — confirmed via grep, zero hits in gliner/ |
| 16 | `test_shuffle_types_is_consumed` | shuffle_types dead | **YES** — confirmed via grep, zero hits in gliner/ |
| 17 | `test_random_drop_is_consumed` | random_drop dead | **YES** — confirmed via grep, zero hits in gliner/ |
| 18 | `test_no_dual_label_smoothing_fields` | Both label_smoothing and label_smoothing_factor exist | **YES** — both fields present on TA |
| 19 | `test_train_model_accepts_resume_from_checkpoint` | resume_from_checkpoint not in signature | **YES** — parameter missing |
| 20 | `test_train_model_forwards_resume_to_trainer` | resume_from_checkpoint not in source | **YES** — completely absent |
| 21 | `test_all_custom_fields_are_explicit_params` | label_smoothing (at minimum) implicit | **YES** — label_smoothing still in kwargs |
| 22 | `test_all_config_training_fields_have_consumers` | dead fields in config.yaml | **YES** — dead fields confirmed |

**Verdict: All 22 xfail tests correctly identify real, verified issues.** None are false positives. None should currently be removed.

---

### GLiNER Extension Fields

All nine GLiNER-specific extension fields are correctly identified and placed under `training:` in the schema, which matches the report's recommendation. The naming differs from upstream (e.g., `loss_alpha` vs `focal_loss_alpha`) but the mapping in `_launch_training` correctly translates between them.

| Field | In schema? | Correctly classified? | Notes |
|---|---|---|---|
| `others_lr` | Yes (`training.lr_others`) | Yes | Correctly treated as training, mapped to `others_lr` kwarg |
| `others_weight_decay` | Yes (`training.weight_decay_other`) | Yes | Correctly treated as training |
| `focal_loss_alpha` | Yes (`training.loss_alpha`) | Yes | Mapped to `focal_loss_alpha` kwarg |
| `focal_loss_gamma` | Yes (`training.loss_gamma`) | Yes | Mapped to `focal_loss_gamma` kwarg |
| `focal_loss_prob_margin` | Yes (`training.loss_prob_margin`) | Yes | Mapped to `focal_loss_prob_margin` kwarg |
| `label_smoothing` | Yes (`training.label_smoothing`) | Yes | Mapped correctly |
| `loss_reduction` | Yes (`training.loss_reduction`) | Yes | Mapped correctly |
| `negatives` | Yes (`training.negatives`) | Yes | Mapped correctly |
| `masking` | Yes (`training.masking`) | Yes | Mapped correctly |

**Issue**: The default for `masking` in `training_cli.py` is `"none"` (line 169), while GLiNER's `TrainingArguments` defaults to `"global"` (trainer.py line 86). Also `create_training_args` defaults masking to `"none"` (model.py line 1013). This means the CLI and factory method override GLiNER's own default. This should be documented or aligned.

---

### LoRA Configuration

#### `training_cli.py` LoRA schema (lines 187-194):

| Field | Our default | Status |
|---|---|---|
| `r` | 8 | OK |
| `lora_alpha` | 16 | OK |
| `lora_dropout` | 0.05 | OK |
| `bias` | `"none"` | OK |
| `target_modules` | `["q_proj", "v_proj"]` | OK |
| `task_type` | `"TOKEN_CLS"` | OK |
| `modules_to_save` | `None` | OK |
| `init_lora_weights` | **MISSING** | MISSING from `_FIELD_SCHEMA` and `_apply_lora` |
| `use_rslora` | **MISSING** | MISSING from `_FIELD_SCHEMA` and `_apply_lora` |
| `fan_in_fan_out` | **MISSING** | MISSING from `_FIELD_SCHEMA` and `_apply_lora` |

The `config_cli.py` already has all three missing fields in `_LORA_RULES`. The two CLI modules remain inconsistent.

The `task_type` mapping in `_apply_lora` (line 1165) only includes 4 options and misses `FEATURE_EXTRACTION` which is what `config_cli.py` defaults to.

---

### _launch_training Parameter Mapping

#### Fields validated AND correctly forwarded (UPDATED):

| Schema field | Forwarded as | Line(s) |
|---|---|---|
| `training.num_steps` | `max_steps` | 1100 |
| `training.scheduler_type` | `lr_scheduler_type` | 1101 |
| `training.warmup_ratio` | `warmup_ratio` | 1102 |
| `training.train_batch_size` | `per_device_train_batch_size` | 1104 |
| `training.eval_batch_size` | `per_device_eval_batch_size` | 1105 |
| `training.lr_encoder` | `learning_rate` | 1107 |
| `training.lr_others` | `others_lr` | 1108 |
| `training.weight_decay_encoder` | `weight_decay` | 1109 |
| `training.weight_decay_other` | `others_weight_decay` | 1110 |
| `training.max_grad_norm` | `max_grad_norm` | 1111 |
| `training.optimizer` | `optim` | 1112 |
| `training.loss_alpha` | `focal_loss_alpha` | 1114 |
| `training.loss_gamma` | `focal_loss_gamma` | 1115 |
| `training.loss_prob_margin` | `focal_loss_prob_margin` | 1116 |
| `training.label_smoothing` | `label_smoothing` | 1117 |
| `training.loss_reduction` | `loss_reduction` | 1118 |
| `training.negatives` | `negatives` | 1119 |
| `training.masking` | `masking` | 1120 |
| `training.eval_every` | `save_steps` + `eval_steps` | 1122, 1127 |
| `training.logging_steps` | `logging_steps` | 1123 |
| `training.save_total_limit` | `save_total_limit` | 1124 |
| `training.bf16` | `bf16` | 1130 |
| `training.fp16` | `fp16` | 1131 |
| `training.use_cpu` | `use_cpu` | 1133 |
| `training.dataloader_num_workers` | `dataloader_num_workers` | 1134 |
| `training.dataloader_pin_memory` | `dataloader_pin_memory` | 1135 |
| `training.dataloader_persistent_workers` | `dataloader_persistent_workers` | 1136 |
| `training.dataloader_prefetch_factor` | `dataloader_prefetch_factor` | 1137 |
| `training.gradient_accumulation_steps` | `gradient_accumulation_steps` | 1142 |
| `training.freeze_components` | `freeze_components` | 1097 |
| `training.compile_model` | `compile_model` | 1098 |
| `training.prev_path` | Model loading logic | 1050-1055 |
| `environment.report_to` | `report_to` | 1139 |
| `run.name` | `run_name` | 1140 |
| (eval_dataset presence) | `eval_strategy="steps"` | 1127 |

#### Fields validated but still NOT forwarded:

| Schema field | Line in schema | Impact |
|---|---|---|
| `training.size_sup` | 182 | **DEAD FIELD** — not consumed anywhere in `gliner/` |
| `training.shuffle_types` | 183 | **DEAD FIELD** — not consumed anywhere in `gliner/` |
| `training.random_drop` | 184 | **DEAD FIELD** — not consumed anywhere in `gliner/` |
| `run.description` | 98 | Not forwarded anywhere |
| `run.tags` | 99 | Not forwarded anywhere |
| `run.seed` | 100 | `torch.manual_seed()` only, NOT forwarded as `seed` to TrainingArguments |
| `environment.push_to_hub` | 197 | Validated but never forwarded to TrainingArguments |
| `environment.hub_model_id` | 198 | Validated but never forwarded to TrainingArguments |

---

### Missing Training Parameters (still absent)

| Missing field | Impact | Priority |
|---|---|---|
| `remove_unused_columns` | Must be `False` for GLiNER. HF defaults to `True`. **Silent data loss risk.** | **CRITICAL** |
| `seed` (as TrainingArguments field) | Trainer-internal seed differs from user seed | HIGH |
| `gradient_checkpointing` | Critical for large backbones on limited GPU memory | HIGH |
| `resume_from_checkpoint` | CLI detects checkpoints but never passes path to Trainer | HIGH |
| `push_to_hub` / `hub_model_id` / `hub_token` | Validated but never reach Trainer | MEDIUM |
| `label_smoothing` (explicit in create_training_args) | Works via kwargs but fragile | MEDIUM |
| `adam_beta1` / `adam_beta2` / `adam_epsilon` | Cannot tune optimizer hyperparameters | MEDIUM |
| `load_best_model_at_end` / `metric_for_best_model` | No model selection support | MEDIUM |
| `num_train_epochs` | Only step-based training supported | LOW |
| `auto_find_batch_size` / `group_by_length` | Throughput optimizations | LOW |
| `deepspeed` / `fsdp` | Distributed training | LOW |

---

### Loss Function Wiring

**Focal loss**: Correctly wired end-to-end. Verified via `TestComputeLossWiring::test_compute_loss_passes_all_loss_params`.

**Label smoothing**: Wired but fragile.
1. `training_cli.py` validates `training.label_smoothing` (line 166)
2. `_launch_training` forwards as `label_smoothing=float(...)` (line 1117)
3. Falls through `**kwargs` in `create_training_args` (not an explicit parameter)
4. Reaches `TrainingArguments.label_smoothing` (trainer.py line 83)
5. `compute_loss` forwards it to model forward (trainer.py line 184)

**Name collision risk**: HF's `TrainingArguments` has `label_smoothing_factor` (applied in HF Trainer's loss computation) while GLiNER adds `label_smoothing` (applied in model forward). Both exist on the same object and could be active simultaneously. Verified by `TestLabelSmoothingCollision::test_no_dual_label_smoothing_fields` (xfail).

**`loss_reduction`**: Correctly wired end-to-end.

---

### Concrete Recommendations (Updated)

#### Still Outstanding

1. **CRITICAL: Add `remove_unused_columns=False`**.
   - In `create_training_args` (model.py) OR in `train_model()` via `training_args.remove_unused_columns = False`.
   - In `_launch_training` (training_cli.py) as an explicit kwarg.
   - GLiNER uses custom batch dictionaries. Without this, HF Trainer drops columns not in the model's `forward()` signature.
   - **Tested by**: `TestRemoveUnusedColumns::test_create_training_args_sets_remove_unused_columns_false` (xfail) and `TestRemoveUnusedColumns::test_train_model_source_references_remove_unused_columns` (xfail).

2. **HIGH: Forward `run.seed` as TrainingArguments `seed`**.
   - Currently only `torch.manual_seed(seed)` is called. The Trainer has its own seed logic.
   - **Tested by**: `TestCreateTrainingArgsSignature::test_seed_is_explicit` (xfail).

3. **HIGH: Add `gradient_checkpointing` to schema and forward it**.
   - Critical for training large backbones on limited GPU memory.
   - **Tested by**: `TestCreateTrainingArgsSignature::test_gradient_checkpointing_is_explicit` (xfail).

4. **HIGH: Remove or document dead fields `size_sup`, `shuffle_types`, `random_drop`**.
   - Confirmed via full-source grep: zero hits in `gliner/` directory.
   - **Tested by**: `TestDeadConfigFields` (3 xfail tests) and `TestConfigYamlDeadFields` (1 xfail test).

5. **HIGH: Wire `--resume` to `resume_from_checkpoint`**.
   - `check_resume()` detects checkpoints but the path is never passed to `trainer.train(resume_from_checkpoint=...)`.
   - **Tested by**: `TestTrainModelResumeSupport` (2 xfail tests).

6. **MEDIUM: Make `label_smoothing` explicit in `create_training_args`**.
   - **Tested by**: `TestCreateTrainingArgsSignature::test_label_smoothing_is_explicit` (xfail) and `TestCreateTrainingArgsCoversCustomFields` (xfail).

7. **MEDIUM: Add missing LoRA fields to `_FIELD_SCHEMA` and `_apply_lora`**.
   - `init_lora_weights`, `use_rslora`, `fan_in_fan_out`
   - Add `FEATURE_EXTRACTION` to the `task_map`.

8. **MEDIUM: Forward Hub-related fields to TrainingArguments**.
   - `push_to_hub`, `hub_model_id`, `hub_token` validated but never forwarded.
   - **Tested by**: `TestCreateTrainingArgsSignature::test_push_to_hub_is_explicit` and `test_hub_model_id_is_explicit` (xfail).

9. **LOW: Align `masking` default (`"none"` in CLI/create_training_args vs `"global"` in TrainingArguments)**.
   - **Tested by**: `TestMaskingDefaultMismatch::test_create_training_args_masking_matches_training_args_default` (xfail).

10. **LOW: Make `fp16` explicit in `create_training_args`**.
    - `bf16` is explicit, but `fp16` is not, creating an asymmetry.
    - **Tested by**: `TestCreateTrainingArgsSignature::test_fp16_is_explicit` (xfail).

#### Already Fixed (compared to original report)

| Original Recommendation | Status | Evidence |
|---|---|---|
| Add `evaluation_strategy="steps"` and `eval_steps` | **FIXED** | `_launch_training` line 1127; `train_model()` lines 1119-1130 |
| Forward `dataloader_pin_memory/persistent_workers/prefetch_factor` | **FIXED** | `_launch_training` lines 1135-1137 |
| Forward `run.name` as `run_name` | **FIXED** | `_launch_training` line 1140 |
| Forward `fp16` | **FIXED** | `_launch_training` line 1131 (via kwargs) |
| Forward `gradient_accumulation_steps` | **FIXED** | `_launch_training` line 1142 |

---

### Test Suite Quality Assessment

The test suite in `tests/test_training_validation.py` is well-structured:

1. **Smoke tests** (5 tests) verify basic functionality and prevent import/instantiation regressions.
2. **Default tests** (4 tests) lock down critical default values.
3. **Completeness tests** (2 tests) ensure all GLiNER-specific fields are declared.
4. **Wiring tests** (4 tests) verify source-level presence of required references in `compute_loss` and `get_train_dataloader`.
5. **Pass-through tests** (3 tests) verify that kwargs actually reach TrainingArguments.
6. **HF default documentation** (2 tests) document risky HF defaults.
7. **Gap documentation** (22 xfail tests) systematically cover every known issue from this report.

**Strengths:**
- `xfail(strict=True)` means tests will XPASS (and alert CI) when issues are fixed, preventing stale markers.
- Source inspection-based tests (inspecting `compute_loss`, `get_train_dataloader`) are resilient to refactoring.
- Each xfail test has a descriptive `reason` string that serves as inline documentation.

**Gaps in test coverage:**
- No test verifies that `_launch_training` in `training_cli.py` forwards all validated fields. The tests focus on `create_training_args` and `train_model`, not the CLI layer.
- No test verifies LoRA field completeness in `_apply_lora`.
- No integration test runs an actual (small) training loop to verify end-to-end parameter flow. This would catch issues like `remove_unused_columns=True` causing column drops at runtime.
- The `TestConfigYamlDeadFields` test depends on `configs/config.yaml` existing, which may not be present in CI.

---

### Conclusion

The training parameter pipeline is fundamentally sound. The code correctly forwards all 9 GLiNER-specific loss/optimization fields from YAML through `_launch_training` → `create_training_args` → `TrainingArguments` → `compute_loss` → model forward. Key fixes since the original report addressed evaluation strategy, dataloader parameters, run naming, fp16, and gradient accumulation steps.

The single most important remaining issue is **`remove_unused_columns`**, which defaults to `True` in HF Trainer and can cause silent data loss with GLiNER's custom batch dictionaries. This should be the next fix applied.

The test suite correctly identifies and tracks all 22 known gaps using `xfail(strict=True)` markers. All 22 xfail tests were verified to be accurate against the current codebase. No false positives were found.
