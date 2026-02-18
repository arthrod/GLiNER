## Training Parameters and Loss Configuration Report

### Executive Summary

Our training toolkit has a well-structured validation layer (`_FIELD_SCHEMA` in `training_cli.py`) and correctly identifies most GLiNER-specific extension fields. However, there are critical gaps: three validated fields (`size_sup`, `shuffle_types`, `random_drop`) are never forwarded to training, three forwarded dataloader fields (`dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor`) rely on `**kwargs` pass-through without explicit documentation, `label_smoothing` is similarly implicit, the LoRA implementation is missing three PEFT fields (`init_lora_weights`, `use_rslora`, `fan_in_fan_out`) that `config_cli.py` already validates, and several high-impact TrainingArguments fields recommended by the research report are completely absent from the schema.

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

**Issue**: The default for `masking` in `training_cli.py` is `"none"` (line 166), while GLiNER's `TrainingArguments` defaults to `"global"` (trainer.py line 85). This mismatch means our CLI overrides GLiNER's default. This should be documented or aligned.

---

### LoRA Configuration

#### `training_cli.py` LoRA schema (lines 184-191):

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

The `config_cli.py` already has all three missing fields (`fan_in_fan_out`, `use_rslora`, `init_lora_weights`) in `_LORA_RULES`. **This means our two CLI modules are inconsistent**: `config_cli.py` validates 10 LoRA fields, while `training_cli.py` only validates 8.

Additional discrepancy: `config_cli.py` defaults `target_modules` to `["query_proj", "value_proj"]` and `task_type` to `"FEATURE_EXTRACTION"`, while `training_cli.py` defaults `target_modules` to `["q_proj", "v_proj"]` and `task_type` to `"TOKEN_CLS"`. The `training_cli.py` defaults are more appropriate for typical use (DeBERTa uses `q_proj`/`v_proj`, and `TOKEN_CLS` is correct for NER), but the inconsistency is a defect.

---

### _launch_training Parameter Mapping

This is the critical end-to-end analysis. For each field validated in `_FIELD_SCHEMA`, I trace whether it actually reaches `model.train_model()`.

#### Fields validated AND correctly forwarded:

| Schema field | Forwarded as | Line(s) |
|---|---|---|
| `training.num_steps` | `max_steps` | 1097 |
| `training.scheduler_type` | `lr_scheduler_type` | 1098 |
| `training.warmup_ratio` | `warmup_ratio` | 1099 |
| `training.train_batch_size` | `per_device_train_batch_size` | 1101 |
| `training.eval_batch_size` | `per_device_eval_batch_size` | 1102 |
| `training.lr_encoder` | `learning_rate` | 1104 |
| `training.lr_others` | `others_lr` | 1105 |
| `training.weight_decay_encoder` | `weight_decay` | 1106 |
| `training.weight_decay_other` | `others_weight_decay` | 1107 |
| `training.max_grad_norm` | `max_grad_norm` | 1108 |
| `training.optimizer` | `optim` | 1109 |
| `training.loss_alpha` | `focal_loss_alpha` | 1111 |
| `training.loss_gamma` | `focal_loss_gamma` | 1112 |
| `training.loss_prob_margin` | `focal_loss_prob_margin` | 1113 |
| `training.label_smoothing` | `label_smoothing` | 1114 |
| `training.loss_reduction` | `loss_reduction` | 1115 |
| `training.negatives` | `negatives` | 1116 |
| `training.masking` | `masking` | 1117 |
| `training.eval_every` | `save_steps` | 1119 |
| `training.logging_steps` | `logging_steps` | 1120 |
| `training.save_total_limit` | `save_total_limit` | 1121 |
| `training.bf16` | `bf16` | 1123 |
| `training.fp16` | `fp16` | 1124 |
| `training.use_cpu` | `use_cpu` | 1126 |
| `training.dataloader_num_workers` | `dataloader_num_workers` | 1127 |
| `training.gradient_accumulation_steps` | `gradient_accumulation_steps` | 1131 |
| `training.freeze_components` | `freeze_components` | 1094 (explicit param) |
| `training.compile_model` | `compile_model` | 1095 (explicit param) |
| `training.prev_path` | Used to select `from_pretrained` vs `from_config` | 1046-1052 |
| `environment.report_to` | `report_to` | 1129 |

#### Fields validated but NEVER forwarded (validated-but-not-forwarded):

| Schema field | Line in schema | Impact |
|---|---|---|
| `training.size_sup` | 179 | **DEAD FIELD** - not consumed by GLiNER library anywhere. Grep across entire `gliner/` directory returns zero hits. This field exists in upstream config YAML examples but has no implementation. |
| `training.shuffle_types` | 180 | **DEAD FIELD** - same as above. Not consumed anywhere in `gliner/`. |
| `training.random_drop` | 181 | **DEAD FIELD** - same as above. Not consumed anywhere in `gliner/`. |
| `training.dataloader_pin_memory` | 174 | **NOT FORWARDED** - validated in schema but never passed to `model.train_model()`. The Trainer would use its default (`True` from HF). Since our default is also `True` this accidentally works, but explicit values set by the user would be silently ignored. |
| `training.dataloader_persistent_workers` | 175 | **NOT FORWARDED** - same issue. Default is `False`, which matches HF default, but user overrides are silently ignored. |
| `training.dataloader_prefetch_factor` | 176 | **NOT FORWARDED** - same issue. |
| `run.name` | 94 | Used for resume matching but not forwarded to `run_name` in TrainingArguments. W&B runs would not have a meaningful name. |
| `run.description` | 95 | Not forwarded anywhere. |
| `run.tags` | 96 | Not forwarded anywhere. |
| `run.seed` | 97 | Used for `torch.manual_seed()` but NOT forwarded as `seed` to TrainingArguments. This means the Trainer's internal seed (for shuffling, etc.) may differ from the user's configured seed. |
| `environment.push_to_hub` | 194 | Validated but never forwarded to TrainingArguments `push_to_hub`. |
| `environment.hub_model_id` | 195 | Validated but never forwarded to TrainingArguments `hub_model_id`. |

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
| `evaluation_strategy` / `eval_steps` | HF-TRAIN | Evaluation | We set `save_steps` but never set `evaluation_strategy="steps"` or `eval_steps`. This means **evaluation never runs** unless the user passes it via... nothing -- there is no way to pass it through our CLI. |
| `save_strategy` | HF-TRAIN | Saving | We set `save_steps` but never explicitly set `save_strategy="steps"`. HF defaults to `"steps"` so this accidentally works, but it's fragile. |
| `run_name` | HF-TRAIN | Logging | Not forwarded from `run.name`. W&B runs get auto-generated names. |
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
1. `training_cli.py` validates `training.loss_alpha`, `training.loss_gamma`, `training.loss_prob_margin` (lines 160-162)
2. `_launch_training` forwards them as `focal_loss_alpha`, `focal_loss_gamma`, `focal_loss_prob_margin` (lines 1111-1113)
3. These flow through `create_training_args` (explicit params, lines 1008-1010) into `TrainingArguments` (trainer.py lines 79-81)
4. `compute_loss` passes them to the model forward call as `alpha`, `gamma`, `prob_margin` (trainer.py lines 150-152)

**Label smoothing**: Wired but fragile. The chain is:
1. `training_cli.py` validates `training.label_smoothing` (line 163)
2. `_launch_training` forwards it as `label_smoothing=float(...)` (line 1114)
3. This is NOT an explicit parameter in `create_training_args` (line 1001-1027) -- it falls through to `**kwargs`
4. It reaches `TrainingArguments` which has `label_smoothing` as a field (trainer.py line 82)
5. `compute_loss` passes it to the model forward as `label_smoothing=self.args.label_smoothing` (trainer.py line 153)

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

1. **YAML -> model -> GLiNER**: Correct. `_launch_training` passes `cfg["model"]` to `GLiNER.from_config(model_cfg)` (line 1052) or `GLiNER.from_pretrained(prev_path)` (line 1049).

2. **YAML -> training -> Trainer**: Partially correct. Training fields are extracted from `cfg["training"]` and passed as kwargs to `model.train_model()`, which creates `TrainingArguments` and a `Trainer`. However, several validated training fields are dropped (see "validated-but-not-forwarded" above), and several important HF TrainingArguments are never populated.

3. **YAML -> peft -> model**: Correct structure. `_apply_lora` (line 1059) applies PEFT to `model.model.token_rep_layer` before `train_model()` is called. However, only 7 of 10 PEFT config fields are forwarded (missing `init_lora_weights`, `use_rslora`, `fan_in_fan_out`).

4. **YAML -> data -> Trainer**: Correct. Training data is loaded from `cfg["data"]["train_data"]` and passed directly to `model.train_model(train_dataset=...)`.

**Missing flow**: The report specifies that `seed` should go to TrainingArguments for Trainer-level reproducibility. We only call `torch.manual_seed(seed)` and never forward it.

**Missing flow**: The report specifies Hub-related fields flow through TrainingArguments. We validate them in the `environment` section but never forward them.

---

### LoRA _apply_lora Implementation

**Location**: `training_cli.py` lines 1137-1183

**Applying to `token_rep_layer`**: This is correct. GLiNER's architecture wraps the HuggingFace backbone inside `model.model.token_rep_layer`, which is the right target for PEFT wrapping. The code correctly checks `hasattr(model, "model") and hasattr(model.model, "token_rep_layer")` before applying.

**Fields forwarded to `LoraConfig`**:

| PEFT field | Forwarded? | Line |
|---|---|---|
| `r` | Yes | 1163 |
| `lora_alpha` | Yes | 1164 |
| `lora_dropout` | Yes | 1165 |
| `bias` | Yes | 1166 |
| `target_modules` | Yes | 1167 |
| `task_type` | Yes | 1168 |
| `modules_to_save` | Yes | 1169 |
| `init_lora_weights` | **NO** | - |
| `use_rslora` | **NO** | - |
| `fan_in_fan_out` | **NO** | - |

**Impact of missing fields**:
- `init_lora_weights`: Controls initialization strategy. PEFT defaults to `True` (Kaiming). Missing this means users cannot use `"loftq"`, `"gaussian"`, or `"pissa"` initialization variants. Low impact for typical use.
- `use_rslora`: Rank-Stabilized LoRA. When `True`, uses `lora_alpha / sqrt(r)` scaling instead of `lora_alpha / r`. This can significantly improve performance for higher ranks. Moderate impact.
- `fan_in_fan_out`: Required when adapting `Conv1D` layers (used by GPT-2 style models). Since GLiNER typically uses BERT/DeBERTa encoders (which use `nn.Linear`), this is low impact but still a correctness gap.

**Additional concern**: The `task_type` mapping (lines 1154-1159) only includes 4 options (`TOKEN_CLS`, `SEQ_CLS`, `CAUSAL_LM`, `SEQ_2_SEQ_LM`) but misses `FEATURE_EXTRACTION` which is what `config_cli.py` defaults to. If someone uses `config_cli.py` to validate and then `training_cli.py` to run, the default `FEATURE_EXTRACTION` task type would silently fall through to `TOKEN_CLS`.

---

### Concrete Recommendations

1. **CRITICAL: Add `remove_unused_columns=False` to `_launch_training`**.
   - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line 1131 (inside `model.train_model()` call)
   - GLiNER uses custom batch dictionaries. Without this, HF Trainer will drop columns not in the model's forward signature, causing silent data loss or crashes.

2. **CRITICAL: Add `evaluation_strategy="steps"` and `eval_steps` to `_launch_training`**.
   - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line ~1119
   - Without this, evaluation never runs during training even though we validate `eval_every`. Add:
     ```python
     evaluation_strategy="steps",
     eval_steps=train_cfg["eval_every"],
     ```

3. **CRITICAL: Forward `run.seed` as TrainingArguments `seed`**.
   - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line ~1131
   - Currently only `torch.manual_seed(seed)` is called. The Trainer has its own seed logic for data shuffling, dropout, etc.

4. **HIGH: Forward `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor` to `model.train_model()`**.
   - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line ~1127
   - These are validated in `_FIELD_SCHEMA` (lines 174-176) but never forwarded. User-specified values are silently ignored.

5. **HIGH: Add `gradient_checkpointing` to `_FIELD_SCHEMA` and forward it**.
   - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line ~176 (schema) and ~1131 (launch)
   - Critical for training large backbones on limited GPU memory.

6. **HIGH: Remove or clearly document dead fields `size_sup`, `shuffle_types`, `random_drop`**.
   - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, lines 179-181
   - These are validated and appear in `template.yaml` but are consumed by nothing in the GLiNER codebase. They mislead users into thinking they have an effect.

7. **HIGH: Forward `run.name` as `run_name` in TrainingArguments**.
   - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line ~1129
   - This enables meaningful W&B run names and checkpoint folder names.

8. **MEDIUM: Add `init_lora_weights`, `use_rslora`, `fan_in_fan_out` to `_FIELD_SCHEMA` lora section and `_apply_lora`**.
   - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, lines 184-191 (schema) and 1162-1169 (`_apply_lora`)
   - Align with `config_cli.py`'s `_LORA_RULES` which already validates these three fields.
   - Add `FEATURE_EXTRACTION` to the `task_map` in `_apply_lora` (line 1154).

9. **MEDIUM: Add `label_smoothing` as an explicit parameter in upstream `create_training_args`**.
   - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/gliner/model.py`, line ~1001
   - Currently relies on `**kwargs` pass-through. Should be explicit for discoverability.

10. **MEDIUM: Forward Hub-related fields from `environment` section to TrainingArguments**.
    - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line ~1131
    - `push_to_hub`, `hub_model_id`, `hub_token` are validated but never reach the Trainer.

11. **MEDIUM: Wire `--resume` to `resume_from_checkpoint` in Trainer**.
    - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line ~1090
    - We detect checkpoints in `check_resume()` but never pass the checkpoint path to `trainer.train(resume_from_checkpoint=path)`.

12. **MEDIUM: Add `adam_beta1`, `adam_beta2`, `adam_epsilon` to `_FIELD_SCHEMA`**.
    - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line ~155
    - These are standard optimizer hyperparameters the report recommends.

13. **LOW: Align `masking` default between `training_cli.py` (`"none"`) and GLiNER's `TrainingArguments` (`"global"`)**.
    - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, line 166
    - Document that our default intentionally overrides GLiNER's default, or change one to match the other.

14. **LOW: Align LoRA defaults between `training_cli.py` and `config_cli.py`**.
    - `training_cli.py`: `target_modules=["q_proj", "v_proj"]`, `task_type="TOKEN_CLS"`
    - `config_cli.py`: `target_modules=["query_proj", "value_proj"]`, `task_type="FEATURE_EXTRACTION"`
    - Standardize on one set of defaults.

15. **LOW: Add `num_train_epochs` as an alternative to `num_steps`**.
    - File: `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, schema
    - Currently `training.num_steps` is REQUIRED. Some users prefer epoch-based training. Allow either `num_steps` or `num_train_epochs` with mutual exclusivity validation.
