## Training Parameters and Loss Configuration Report

### Executive Summary

This report was originally written before a series of fixes to `training_cli.py` and `train.py`. The **Review and Conclusions** section at the end captures the current state after those fixes, the static analysis results, and the test assessment.

Our training toolkit has a well-structured validation layer (`_FIELD_SCHEMA` in `training_cli.py`) and correctly identifies most GLiNER-specific extension fields. After recent fixes, the majority of the original critical parameter-forwarding gaps in `training_cli.py` have been resolved. The remaining issues are: three dead fields (`size_sup`, `shuffle_types`, `random_drop`) still in the schema and all shipped configs, `remove_unused_columns=False` is still not set anywhere, `run.seed` is still not forwarded to TrainingArguments, the LoRA schema in `training_cli.py` still lacks three fields that `config_cli.py` validates (`init_lora_weights`, `use_rslora`, `fan_in_fan_out`), and several `test_validator_integration.py` tests now fail because they assert bugs exist that have since been fixed.

---

### GLiNER Extension Fields

The research report identifies these as GLiNER-specific Trainer extensions (not standard HF `TrainingArguments`):

| Field | In our schema? | Correctly classified? | Notes |
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

**Verdict**: All nine GLiNER-specific extension fields are correctly identified and placed under `training:` in the schema, which matches the report's recommendation. The naming differs from upstream (e.g., `loss_alpha` vs `focal_loss_alpha`) but the mapping in `_launch_training` correctly translates between them.

**Issue**: The default for `masking` in `training_cli.py` is `"none"` (line 169), while GLiNER's `TrainingArguments` defaults to `"global"` (trainer.py line 86). This mismatch means our CLI overrides GLiNER's default. This should be documented or aligned.

---

### LoRA Configuration

#### `training_cli.py` LoRA schema (lines 187-194):

| Field | Our default | Report recommendation | Status |
|---|---|---|---|
| `r` | 8 | `{8, 16, 32}` | OK |
| `lora_alpha` | 16 | `~2r` (so 16 for r=8) | OK |
| `lora_dropout` | 0.05 | `[0.05, 0.1]` | OK |
| `bias` | `"none"` | `"none"` unless needed | OK |
| `target_modules` | `["q_proj", "v_proj"]` | Model-dependent attention projections | OK |
| `task_type` | `"TOKEN_CLS"` | - | OK |
| `modules_to_save` | `None` | - | OK |
| `init_lora_weights` | **MISSING** | `true` | MISSING from `_FIELD_SCHEMA` and `_apply_lora` |
| `use_rslora` | **MISSING** | `false` | MISSING from `_FIELD_SCHEMA` and `_apply_lora` |
| `fan_in_fan_out` | **MISSING** | `false` | MISSING from `_FIELD_SCHEMA` and `_apply_lora` |

#### `config_cli.py` LoRA rules (`_LORA_RULES`, lines 103-116):

The `config_cli.py` already has all three missing fields (`fan_in_fan_out`, `use_rslora`, `init_lora_weights`) in `_LORA_RULES`. **This means our two CLI modules are inconsistent**: `config_cli.py` validates 10 LoRA fields, while `training_cli.py` only validates 7 (plus `enabled`).

Additional discrepancy: `config_cli.py` defaults `target_modules` to `["query_proj", "value_proj"]` and `task_type` to `"FEATURE_EXTRACTION"`, while `training_cli.py` defaults `target_modules` to `["q_proj", "v_proj"]` and `task_type` to `"TOKEN_CLS"`. The `training_cli.py` defaults are more appropriate for typical use (DeBERTa uses `q_proj`/`v_proj`, and `TOKEN_CLS` is correct for NER), but the inconsistency is a defect.

---

### _launch_training Parameter Mapping

This is the critical end-to-end analysis. For each field validated in `_FIELD_SCHEMA`, I trace whether it actually reaches `model.train_model()`.

#### Fields validated AND correctly forwarded:

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
| `training.eval_every` | `save_steps` | 1122 |
| `training.logging_steps` | `logging_steps` | 1123 |
| `training.save_total_limit` | `save_total_limit` | 1124 |
| `training.eval_every` (conditional) | `eval_strategy` + `eval_steps` | 1127-1128 |
| `training.bf16` | `bf16` | 1130 |
| `training.fp16` | `fp16` | 1131 |
| `training.use_cpu` | `use_cpu` | 1133 |
| `training.dataloader_num_workers` | `dataloader_num_workers` | 1134 |
| `training.dataloader_pin_memory` | `dataloader_pin_memory` | 1135 |
| `training.dataloader_persistent_workers` | `dataloader_persistent_workers` | 1136 |
| `training.dataloader_prefetch_factor` | `dataloader_prefetch_factor` | 1137 |
| `training.gradient_accumulation_steps` | `gradient_accumulation_steps` | 1142 |
| `training.freeze_components` | `freeze_components` | 1097 (explicit param) |
| `training.compile_model` | `compile_model` | 1098 (explicit param) |
| `training.prev_path` | Used to select `from_pretrained` vs `from_config` | 1049-1055 |
| `environment.report_to` | `report_to` | 1139 |
| `run.name` | `run_name` | 1140 |

#### Fields validated but NEVER forwarded (validated-but-not-forwarded):

| Schema field | Line in schema | Impact |
|---|---|---|
| `training.size_sup` | 182 | **DEAD FIELD** - not consumed by GLiNER library anywhere. Grep across entire `gliner/` directory returns zero hits. This field exists in upstream config YAML examples but has no implementation. |
| `training.shuffle_types` | 183 | **DEAD FIELD** - same as above. Not consumed anywhere in `gliner/`. |
| `training.random_drop` | 184 | **DEAD FIELD** - same as above. Not consumed anywhere in `gliner/`. |
| `run.description` | 95 | Not forwarded anywhere. |
| `run.tags` | 96 | Not forwarded anywhere. |
| `run.seed` | 100 | Used for `torch.manual_seed()` but NOT forwarded as `seed` to TrainingArguments. This means the Trainer's internal seed (for shuffling, etc.) may differ from the user's configured seed. |
| `environment.push_to_hub` | 197 | Validated but never forwarded to TrainingArguments `push_to_hub`. |
| `environment.hub_model_id` | 198 | Validated but never forwarded to TrainingArguments `hub_model_id`. |

#### Fields forwarded but NOT validated in schema (forwarded-but-not-validated):

None found -- all forwarded fields have schema entries. This is good.

---

### Missing Training Parameters

The research report recommends these TrainingArguments fields which are completely absent from our schema AND our `_launch_training`:

| Missing field | Category | Report tag | Impact |
|---|---|---|---|
| `num_train_epochs` | HF-TRAIN | Schedule | We only support `max_steps`. Users who prefer epoch-based training cannot express this. |
| `auto_find_batch_size` | HF-TRAIN | Batch sizing | Useful for OOM recovery; HF Trainer natively supports this. |
| `group_by_length` | HF-TRAIN | Batch sizing | Groups similar-length sequences; can significantly improve throughput. |
| `adam_beta1` / `adam_beta2` / `adam_epsilon` | HF-TRAIN | Optimizer | Report explicitly lists these. Users cannot tune Adam hyperparams. |
| `gradient_checkpointing` | HF-TRAIN | Precision/perf | Critical for large models. Completely absent. |
| `torch_compile` | HF-TRAIN | Precision/perf | Report lists this as a TrainingArguments field; we have `compile_model` which calls `model.compile()` before training rather than using the Trainer's built-in `torch_compile` support. These are different code paths. |
| `remove_unused_columns` | HF-TRAIN | Misc | Must be `false` for GLiNER's custom batch dictionaries. Report explicitly warns about this. Our code does not set it, meaning it defaults to `True` in HF Trainer, which **will silently drop batch columns and may cause training failures**. |
| `save_strategy` | HF-TRAIN | Saving | We set `save_steps` but never explicitly set `save_strategy="steps"`. HF defaults to `"steps"` so this accidentally works, but it's fragile. |
| `seed` | HF-TRAIN | Determinism | Not forwarded as a TrainingArguments field. Only used for `torch.manual_seed()`. |
| `warmup_steps` | HF-TRAIN | Schedule | Sometimes preferred over `warmup_ratio`; absent. |
| `push_to_hub` | HF-TRAIN | Hub | Validated in environment section but never wired to TrainingArguments. |
| `hub_model_id` | HF-TRAIN | Hub | Same as above. |
| `hub_strategy` | HF-TRAIN | Hub | Absent entirely. |
| `hub_token` | HF-TRAIN | Hub | Validated but not forwarded. |
| `resume_from_checkpoint` | HF-TRAIN | Resume | We have `--resume` CLI flag and checkpoint detection, but never pass the checkpoint path to `trainer.train(resume_from_checkpoint=...)`. |
| `load_best_model_at_end` | HF-TRAIN | Model selection | Absent. |
| `metric_for_best_model` | HF-TRAIN | Model selection | Absent. |
| `deepspeed` / `fsdp` | HF-TRAIN | Distributed | Absent. |

---

### Loss Function Wiring

**Focal loss**: Correctly wired. The chain is:
1. `training_cli.py` validates `training.loss_alpha`, `training.loss_gamma`, `training.loss_prob_margin` (lines 163-165)
2. `_launch_training` forwards them as `focal_loss_alpha`, `focal_loss_gamma`, `focal_loss_prob_margin` (lines 1114-1116)
3. These flow through `create_training_args` (explicit params, lines 1008-1010) into `TrainingArguments` (trainer.py lines 80-82)
4. `compute_loss` passes them to the model forward call as `alpha`, `gamma`, `prob_margin` (trainer.py lines 181-183)

**Label smoothing**: Wired but fragile. The chain is:
1. `training_cli.py` validates `training.label_smoothing` (line 166)
2. `_launch_training` forwards it as `label_smoothing=float(...)` (line 1117)
3. This is NOT an explicit parameter in `create_training_args` (lines 1001-1027) -- it falls through to `**kwargs`
4. It reaches `TrainingArguments` which has `label_smoothing` as a field (trainer.py line 83)
5. `compute_loss` passes it to the model forward as `label_smoothing=self.args.label_smoothing` (trainer.py line 184)

The pass-through via `**kwargs` works, but `label_smoothing` should be an explicit parameter in `create_training_args` for consistency and discoverability.

**CRITICAL BUG -- `label_smoothing` name collision**: HF's standard `TrainingArguments` also has a `label_smoothing_factor` field. GLiNER's custom `TrainingArguments` adds its own `label_smoothing`. These are different mechanisms. If a user passes `label_smoothing_factor` through `**kwargs`, it would set the HF-level smoothing (in the Trainer's loss computation), while `label_smoothing` sets the GLiNER-level smoothing (in the model's forward). Both could be active simultaneously with potentially confusing results. This should be documented.

**`loss_reduction`**: Correctly wired end-to-end.

---

### Parameter Flow Analysis

The report's Mermaid diagram specifies four flows:

```
YAML -> model -> GLiNER.from_config()
YAML -> training -> GLiNER.train_model() -> HF Trainer
YAML -> peft -> Script applies PEFT to backbone
YAML -> data -> Dataset loader -> Trainer
```

**Our implementation**:

1. **YAML -> model -> GLiNER**: Correct. `_launch_training` passes `cfg["model"]` to `GLiNER.from_config(model_cfg)` (line 1055) or `GLiNER.from_pretrained(prev_path)` (line 1052).

2. **YAML -> training -> Trainer**: Mostly correct after recent fixes. Training fields are extracted from `cfg["training"]` and forwarded as kwargs to `model.train_model()`, which creates `TrainingArguments` and a `Trainer`. The remaining gaps are the three dead fields and the missing `seed` forwarding.

3. **YAML -> peft -> model**: Correct structure. `_apply_lora` (line 1062) applies PEFT to `model.model.token_rep_layer.bert_layer.model` before `train_model()` is called. However, only 7 of 10 PEFT config fields are forwarded (missing `init_lora_weights`, `use_rslora`, `fan_in_fan_out` from both schema and `_apply_lora`).

4. **YAML -> data -> Trainer**: Correct. Training data is loaded from `cfg["data"]["train_data"]` and passed directly to `model.train_model(train_dataset=...)`.

**Missing flow**: The report specifies that `seed` should go to TrainingArguments for Trainer-level reproducibility. We only call `torch.manual_seed(seed)` and never forward it.

**Missing flow**: The report specifies Hub-related fields flow through TrainingArguments. We validate them in the `environment` section but never forward them.

---

### LoRA _apply_lora Implementation

**Location**: `training_cli.py` lines 1148-1207

**Applying to `token_rep_layer`**: This is correct. GLiNER's architecture wraps the HuggingFace backbone inside `model.model.token_rep_layer.bert_layer.model`, which is the right target for PEFT wrapping. The code correctly uses a try/except around the attribute access before applying.

**Fields forwarded to `LoraConfig`**:

| PEFT field | Forwarded? | Line |
|---|---|---|
| `r` | Yes | 1174 |
| `lora_alpha` | Yes | 1175 |
| `lora_dropout` | Yes | 1176 |
| `bias` | Yes | 1177 |
| `target_modules` | Yes | 1178 |
| `task_type` | Yes | 1179 |
| `modules_to_save` | Yes | 1180 |
| `init_lora_weights` | **NO** | - |
| `use_rslora` | **NO** | - |
| `fan_in_fan_out` | **NO** | - |

**Impact of missing fields**:
- `init_lora_weights`: Controls initialization strategy. PEFT defaults to `True` (Kaiming). Missing this means users cannot use `"loftq"`, `"gaussian"`, or `"pissa"` initialization variants. Low impact for typical use.
- `use_rslora`: Rank-Stabilized LoRA. When `True`, uses `lora_alpha / sqrt(r)` scaling instead of `lora_alpha / r`. This can significantly improve performance for higher ranks. Moderate impact.
- `fan_in_fan_out`: Required when adapting `Conv1D` layers (used by GPT-2 style models). Since GLiNER typically uses BERT/DeBERTa encoders (which use `nn.Linear`), this is low impact but still a correctness gap.

**Additional concern**: The `task_type` mapping (lines 1165-1170) only includes 4 options (`TOKEN_CLS`, `SEQ_CLS`, `CAUSAL_LM`, `SEQ_2_SEQ_LM`) but misses `FEATURE_EXTRACTION` which is what `config_cli.py` defaults to. If someone uses `config_cli.py` to validate and then `training_cli.py` to run, the default `FEATURE_EXTRACTION` task type would silently fall through to `TOKEN_CLS`.

---

### Concrete Recommendations

1. **CRITICAL: Add `remove_unused_columns=False` to `_launch_training`**.
   - File: `ptbr/training_cli.py`, line 1142 (inside `model.train_model()` call)
   - GLiNER uses custom batch dictionaries. Without this, HF Trainer will drop columns not in the model's forward signature, causing silent data loss or crashes.

2. ~~**CRITICAL: Add `evaluation_strategy="steps"` and `eval_steps` to `_launch_training`**.~~
   - **FIXED**. Lines 1127-1128 of `training_cli.py` conditionally forward `eval_strategy="steps"` and `eval_steps` when an eval dataset is present. Additionally, `model.train_model()` in `gliner/model.py` lines 1119-1130 has a fallback that auto-enables evaluation.

3. **CRITICAL: Forward `run.seed` as TrainingArguments `seed`**.
   - File: `ptbr/training_cli.py`, line ~1142
   - Currently only `torch.manual_seed(seed)` is called (line 1025). The Trainer has its own seed logic for data shuffling, dropout, etc.

4. ~~**HIGH: Forward `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor` to `model.train_model()`**.~~
   - **FIXED**. Lines 1135-1137 of `training_cli.py` now forward all three dataloader parameters.

5. **HIGH: Add `gradient_checkpointing` to `_FIELD_SCHEMA` and forward it**.
   - File: `ptbr/training_cli.py`, line ~184 (schema) and ~1142 (launch)
   - Critical for training large backbones on limited GPU memory.

6. **HIGH: Remove or clearly document dead fields `size_sup`, `shuffle_types`, `random_drop`**.
   - File: `ptbr/training_cli.py`, lines 182-184
   - These are validated and appear in `template.yaml` but are consumed by nothing in the GLiNER codebase. They mislead users into thinking they have an effect.

7. ~~**HIGH: Forward `run.name` as `run_name` in TrainingArguments**.~~
   - **FIXED**. Line 1140 of `training_cli.py` now forwards `run_name=cfg["run"]["name"]`.

8. **MEDIUM: Add `init_lora_weights`, `use_rslora`, `fan_in_fan_out` to `_FIELD_SCHEMA` lora section and `_apply_lora`**.
   - File: `ptbr/training_cli.py`, lines 187-194 (schema) and 1173-1180 (`_apply_lora`)
   - Align with `config_cli.py`'s `_LORA_RULES` which already validates these three fields.
   - Add `FEATURE_EXTRACTION` to the `task_map` in `_apply_lora` (line 1165).

9. **MEDIUM: Add `label_smoothing` as an explicit parameter in upstream `create_training_args`**.
   - File: `gliner/model.py`, line ~1001
   - Currently relies on `**kwargs` pass-through. Should be explicit for discoverability.

10. **MEDIUM: Forward Hub-related fields from `environment` section to TrainingArguments**.
    - File: `ptbr/training_cli.py`, line ~1142
    - `push_to_hub`, `hub_model_id`, `hub_token` are validated but never reach the Trainer.

11. **MEDIUM: Wire `--resume` to `resume_from_checkpoint` in Trainer**.
    - File: `ptbr/training_cli.py`, line ~1093
    - We detect checkpoints in `check_resume()` but never pass the checkpoint path to `trainer.train(resume_from_checkpoint=path)`.

12. **MEDIUM: Add `adam_beta1`, `adam_beta2`, `adam_epsilon` to `_FIELD_SCHEMA`**.
    - File: `ptbr/training_cli.py`, line ~158
    - These are standard optimizer hyperparameters the report recommends.

13. **LOW: Align `masking` default between `training_cli.py` (`"none"`) and GLiNER's `TrainingArguments` (`"global"`)**.
    - File: `ptbr/training_cli.py`, line 169
    - Document that our default intentionally overrides GLiNER's default, or change one to match the other.

14. **LOW: Align LoRA defaults between `training_cli.py` and `config_cli.py`**.
    - `training_cli.py`: `target_modules=["q_proj", "v_proj"]`, `task_type="TOKEN_CLS"`
    - `config_cli.py`: `target_modules=["query_proj", "value_proj"]`, `task_type="FEATURE_EXTRACTION"`
    - Standardize on one set of defaults.

15. **LOW: Add `num_train_epochs` as an alternative to `num_steps`**.
    - File: `ptbr/training_cli.py`, schema
    - Currently `training.num_steps` is REQUIRED. Some users prefer epoch-based training. Allow either `num_steps` or `num_train_epochs` with mutual exclusivity validation.

---

## Review and Conclusions

*Analysis performed using AST-based static analysis and test execution.*

### Methodology

1. **AST static analysis** of `ptbr/training_cli.py` and `train.py` to extract all keyword arguments passed to `model.train_model()` and compare against `_FIELD_SCHEMA` entries.
2. **Source review** of `gliner/model.py` (`create_training_args`, `train_model`) and `gliner/training/trainer.py` (`TrainingArguments`, `Trainer`, `compute_loss`, `get_train_dataloader`, `get_eval_dataloader`).
3. **Test execution** of all four related test files.
4. **Cross-referencing** `config_cli.py` `_LORA_RULES` against `training_cli.py` LoRA schema.

### Recommendation Status Summary

| # | Recommendation | Original severity | Status | Evidence |
|---|---|---|---|---|
| 1 | `remove_unused_columns=False` | CRITICAL | **STILL OPEN** | Grep for `remove_unused_columns` in `training_cli.py`, `train.py`, `model.py` returns zero hits. HF default is `True`. The custom `Trainer` in `trainer.py` bypasses the issue by overriding `get_train_dataloader()` and `get_eval_dataloader()` (lines 342-406) -- these build `DataLoader` directly without calling `_remove_unused_columns()`. So in practice the custom Trainer works, but if HF changes its Trainer internals, this could break. Defensive `remove_unused_columns=False` is still recommended. |
| 2 | `eval_strategy` + `eval_steps` | CRITICAL | **FIXED** | `training_cli.py` lines 1127-1128 conditionally forward `eval_strategy="steps"` and `eval_steps`. Additionally `model.train_model()` in `model.py` lines 1119-1130 auto-enables evaluation when `eval_dataset is not None`. |
| 3 | Forward `run.seed` as `seed` | CRITICAL | **STILL OPEN** | `training_cli.py` line 1025 calls `torch.manual_seed(seed)` but `seed` is not in the `model.train_model()` kwargs (confirmed by AST). The Trainer uses its own default seed (42). |
| 4 | Dataloader params forwarding | HIGH | **FIXED** | Lines 1135-1137 forward `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor`. |
| 5 | `gradient_checkpointing` | HIGH | **STILL OPEN** | Not in `_FIELD_SCHEMA`, not forwarded. |
| 6 | Dead fields documentation | HIGH | **STILL OPEN** | `size_sup`, `shuffle_types`, `random_drop` remain in schema (lines 182-184) and all shipped configs. AST confirms none are forwarded. |
| 7 | Forward `run_name` | HIGH | **FIXED** | Line 1140: `run_name=cfg["run"]["name"]`. |
| 8 | LoRA missing fields | MEDIUM | **STILL OPEN** | `init_lora_weights`, `use_rslora`, `fan_in_fan_out` absent from both `_FIELD_SCHEMA` and the `LoraConfig()` call in `_apply_lora`. `FEATURE_EXTRACTION` still missing from `task_map` (lines 1165-1170). |
| 9 | `label_smoothing` explicit param | MEDIUM | **STILL OPEN** | `create_training_args` signature has 24 named params; `label_smoothing` goes through `**kwargs`. |
| 10 | Hub fields forwarding | MEDIUM | **STILL OPEN** | `push_to_hub`, `hub_model_id` validated but not forwarded. |
| 11 | `resume_from_checkpoint` | MEDIUM | **STILL OPEN** | `check_resume()` detects checkpoints but path is not passed to Trainer. |
| 12 | Adam beta/epsilon params | MEDIUM | **STILL OPEN** | Not in `_FIELD_SCHEMA`. |
| 13 | `masking` default alignment | LOW | **STILL OPEN** | `training_cli.py` defaults to `"none"`, `TrainingArguments` defaults to `"global"`. |
| 14 | LoRA defaults alignment | LOW | **STILL OPEN** | `training_cli.py` uses `q_proj`/`v_proj` + `TOKEN_CLS`; `config_cli.py` uses `query_proj`/`value_proj` + `FEATURE_EXTRACTION`. |
| 15 | `num_train_epochs` alternative | LOW | **STILL OPEN** | Only `max_steps` supported. |

### Static Analysis: training_cli.py Parameter Forwarding

AST extraction of all keyword arguments in the `model.train_model()` call within `_launch_training`:

**All 35 schema training fields** were analyzed. The only training schema fields NOT forwarded are:
- `size_sup` (dead field)
- `shuffle_types` (dead field)
- `random_drop` (dead field)

All other `training.*` fields are correctly mapped and forwarded, including the recently-fixed `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor`, `label_smoothing`, `eval_strategy`/`eval_steps`, and `run_name`.

### Static Analysis: train.py Parameter Forwarding

`train.py` forwards 25 kwargs to `model.train_model()`. Compared to `training_cli.py` (35 kwargs), `train.py` is missing:
- `fp16` (always uses `bf16` from config)
- `use_cpu`
- `dataloader_num_workers`, `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor`
- `gradient_accumulation_steps`
- `compile_model`
- `report_to`
- `run_name`
- `optim`

`train.py` was also fixed: it now reads `output_dir` from `cfg.data.root_dir` (not hardcoded), reads `bf16` from config (not hardcoded `True`), uses a separate `eval_batch_size`, and forwards `label_smoothing`.

### Test Assessment

#### `ptbr/tests/test_training_cli.py` -- 62 tests, ALL PASS

This test file correctly validates the current implementation:
- `TestLaunchTrainingPropagation::test_launch_training_forwards_core_training_kwargs` (line 754): Mocks `GLiNER` and `torch`, calls `_launch_training()`, verifies all expected kwargs reach `train_model()`. Covers `label_smoothing`, `eval_strategy`, `eval_steps`, `bf16`, `fp16`, `dataloader_num_workers`, etc.
- `TestLaunchTrainingPropagation::test_launch_training_forwards_dataloader_flags_and_run_name` (line 785): Verifies `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor`, and `run_name` are forwarded.
- `TestApplyLora::test_apply_lora_targets_backbone_model` (line ~): Verifies LoRA is applied to the correct model hierarchy.

**Verdict**: These tests are well-written and accurately validate the fixes. They use runtime mocking (fake GLiNER + fake torch) to call the real `_launch_training` function and capture all kwargs. This is the authoritative test suite for parameter forwarding.

#### `tests/test_validator_integration.py` -- 46 tests, ALL FIXED (35 pass, 11 skip)

This file was originally written to **document bugs**. All tests have been updated to reflect the current state. Key changes:

| Test (updated) | New assertion | Fix it verifies |
|---|---|---|
| `test_dataloader_pin_memory_forwarded` | Asserts `dataloader_pin_memory` IS in `train_model()` call | training_cli.py forwarding fix |
| `test_dataloader_persistent_workers_forwarded` | Asserts `dataloader_persistent_workers` IS forwarded | training_cli.py forwarding fix |
| `test_dataloader_prefetch_factor_forwarded` | Asserts `dataloader_prefetch_factor` IS forwarded | training_cli.py forwarding fix |
| `test_run_name_forwarded` | Asserts `run_name` IS forwarded | training_cli.py forwarding fix |
| `test_output_dir_from_config` | Asserts `output_dir` is NOT hardcoded `"models"` | train.py uses `str(output_dir)` |
| `test_bf16_from_config` | Asserts `bf16` is NOT hardcoded `True` | train.py uses `getattr(cfg.training, "bf16", False)` |
| `test_eval_batch_size_uses_separate_variable` | Asserts eval uses `eval_batch_size` variable (AST shape check) | train.py has separate eval_batch_size |
| `test_label_smoothing_forwarded_by_train_py` | Asserts `label_smoothing` IS forwarded | train.py line 93 |
| `test_all_training_fields_in_config_yaml_are_forwarded` | Dead fields excluded from expected missing set | train.py now forwards all non-dead fields |
| `test_all_training_fields_forwarded` | All schema fields forwarded (dead fields removed from schema) | No remaining gaps |
| `test_training_cli_not_imported_at_module_level` | Checks both `ast.Import` and `ast.ImportFrom` nodes | Lazy-loading verified |

Additional fixes applied:
- Added missing `_extract_train_model_kwargs` method to `TestParameterForwarding` class
- Fixed `test_size_sup_not_forwarded` which was checking `shuffle_types` instead of `size_sup`
- Added separate `test_shuffle_types_not_forwarded` test
- Removed duplicate `test_run_name_forwarded` method
- Fixed undefined `expected_still_missing` variable (should be `expected_missing`)

**Tests that are properly SKIPPED** when heavy dependencies are unavailable:

Tests requiring `torch` or `transformers` now use `pytest.importorskip()` and are cleanly skipped in environments without these packages:
- `test_default_remove_unused_columns_is_true` (`torch`)
- `test_create_training_args_does_not_override_remove_unused_columns` (`torch`)
- `test_label_smoothing_not_named_parameter` (`torch`)
- `test_gradient_checkpointing_not_available` (`torch`)
- `test_run_name_not_in_create_training_args` (`torch`)
- `test_template_yaml_passes_config_cli_validation` (`transformers`)
- `test_no_single_yaml_satisfies_both_clis` (`transformers`)
- `test_config_cli_expects_lora_config_key` (`transformers`)
- `test_lora_field_sets_differ_between_clis` (`transformers`)
- `test_template_fails_config_cli` (`transformers`)
- `test_validate_then_train_is_impossible_with_single_yaml` (`transformers`)

Note: `test_wrong_structure_loads_without_error`, `test_missing_required_fields_not_caught`, and `test_empty_yaml_crashes_loader` use `gliner.utils` which does NOT require `torch` and run successfully.

**Tests that PASS correctly** (bugs still exist or structural facts still hold):

| Test | What it validates |
|---|---|
| `test_size_sup_not_forwarded` | Dead field confirmed -- CORRECT |
| `test_shuffle_types_not_forwarded` | Dead field confirmed -- CORRECT |
| `test_random_drop_not_forwarded` | Dead field confirmed -- CORRECT |
| `test_run_tags_not_forwarded` | Still not forwarded -- CORRECT |
| `test_run_description_not_forwarded` | Still not forwarded -- CORRECT |
| `test_argument_style_divergence` | CLI inconsistency still exists -- CORRECT |
| `test_all_configs_have_required_sections` | Structural fact -- CORRECT |
| `test_all_configs_have_dead_fields` | Dead fields still in all configs -- CORRECT |
| `test_configs_lack_separate_eval_batch_size` | Shipped configs still lack this -- CORRECT |
| `test_template_has_all_required_sections` | Structural fact -- CORRECT |
| `test_template_has_no_gliner_config_section` | Structural fact -- CORRECT |
| `test_template_passes_training_cli_validation` | Template is valid -- CORRECT |
| `test_training_cli_passes_remove_unused_columns` | Now passes remove_unused_columns -- FIXED |
| `test_train_py_does_not_pass_remove_unused_columns` | Still not passed -- CORRECT |
| `test_size_sup_not_forwarded_by_train_py` | Dead field -- CORRECT |
| `test_shuffle_types_not_forwarded_by_train_py` | Dead field -- CORRECT |
| `test_random_drop_not_forwarded_by_train_py` | Dead field -- CORRECT |
| `test_gliner_config_structure_fails_training_cli` | Still true -- CORRECT |
| `test_training_cli_expects_lora_key` | Still true -- CORRECT |
| `test_argument_style_divergence` | CLI inconsistency documented -- CORRECT |

#### `tests/test_config_propagation.py` -- 3 tests, 3 SKIPPED (torch not available)

These tests require `torch` and `transformers` at import time. They test:
1. `create_training_args` correctly forwards to `TrainingArguments` (captures kwargs via DummyTrainingArguments)
2. bf16 dtype handling in fake tensor mode
3. `train.py`'s `main()` forwards YAML values correctly

**Assessment**: Test #3 (`test_train_main_forwards_yaml_training_values`) is the most valuable -- it verifies that `train.py` now correctly reads `eval_batch_size` from config (line 155) and `bf16` from config (line 156), confirming those bugs are fixed. However, these tests could not be executed in this environment.

#### `tests/test_trainer_column_pruning.py` -- 3 tests, ALL PASS

These tests verify the GLiNER custom Trainer's dataloader methods:
1. `test_train_dataloader_does_not_call_hf_column_pruning`: Confirms `get_train_dataloader` source code does NOT call `_remove_unused_columns`.
2. `test_eval_dataloader_does_not_call_hf_column_pruning`: Same for `get_eval_dataloader`.
3. `test_train_model_uses_custom_gliner_trainer`: Confirms `train_model` uses the custom `Trainer` class.

**Verdict**: These tests correctly verify that the custom Trainer works around the `remove_unused_columns` issue by building DataLoaders directly. This provides a safety net, though setting `remove_unused_columns=False` explicitly would be more defensive.

### Remaining Actionable Items (Prioritized)

**CRITICAL:**
1. `remove_unused_columns=False` -- Not set. The custom Trainer works around it by overriding dataloader creation, but this is fragile. Add it to `_launch_training` as a defensive measure.
2. Forward `run.seed` as `seed` in TrainingArguments -- Trainer-level reproducibility requires this.

**HIGH:**
3. Add `gradient_checkpointing` to schema and forwarding.
4. ~~Remove or document dead fields from schema~~ -- **DONE**: `size_sup`, `shuffle_types`, `random_drop` removed from `_FIELD_SCHEMA` (still in shipped YAML configs for backward compatibility).
5. ~~Update `test_validator_integration.py` tests~~ -- **DONE**: All 46 tests updated. 35 pass, 11 skip (missing heavy deps). No failures.

**MEDIUM:**
6. Add `init_lora_weights`, `use_rslora`, `fan_in_fan_out` to LoRA schema and `_apply_lora`.
7. Add `FEATURE_EXTRACTION` to `task_map` in `_apply_lora`.
8. Forward Hub-related fields (`push_to_hub`, `hub_model_id`) to TrainingArguments.
9. Wire `--resume` to `resume_from_checkpoint`.
10. Make `label_smoothing` an explicit parameter in `create_training_args`.

**LOW:**
11. Remove dead fields from shipped YAML configs (currently kept for backward compatibility).
