## Integration and End-to-End Architecture Report

### Executive Summary

The `ptbr` toolkit contains a **showstopper configuration format incompatibility** between its two core CLIs: `config_cli.py` expects a YAML with top-level `gliner_config:` and `lora_config:` sections, while `training_cli.py` (and the shipped `template.yaml`) uses `model:`, `training:`, `lora:`, `environment:` sections. A user cannot use the same YAML file for both `python -m ptbr config` and `python -m ptbr train`. Beyond this fatal split, the toolkit is missing significant data features recommended by the research report (document schema fields, long-document chunking, preprocessing configuration), does not forward several parameters that `create_training_args` accepts via `**kwargs`, and deviates from the report's canonical YAML structure in naming and section organization.

---

### CRITICAL: Incompatible YAML Structures

This is the most severe defect in the toolkit. The two CLIs that are both routed from the same `__main__.py` entry point expect **mutually exclusive** YAML layouts.

**config_cli.py** (`/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/config_cli.py`):
- Line 417: Checks for `raw["gliner_config"]` -- hard error if missing.
- Line 448: Checks for `raw["lora_config"]` when in LoRA mode.
- Validates fields like `gliner_config.model_name`, `gliner_config.span_mode`, etc.
- Has **no awareness** of `model:`, `training:`, `data:`, `run:`, `environment:` sections.

**training_cli.py** (`/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`):
- Line 100: Schema expects `model.model_name`, `model.span_mode`, etc.
- Line 142-145: Schema expects `data.root_dir`, `data.train_data`, `data.val_data_dir`.
- Line 147-181: Schema expects `training.num_steps`, `training.lr_encoder`, etc.
- Line 184-191: Schema expects `lora.enabled`, `lora.r`, etc.
- Line 194-201: Schema expects `environment.push_to_hub`, `environment.wandb_project`, etc.
- Has **no awareness** of `gliner_config:` or `lora_config:` sections.

**template.yaml** (`/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/template.yaml`):
- Uses the `training_cli.py` structure: `run:`, `model:`, `data:`, `training:`, `lora:`, `environment:`.
- **Cannot** be validated by `config_cli.py` -- the config subcommand will immediately error with: `"Missing 'gliner_config' section in YAML file."`

**Reproduction of the bug:**

```bash
# Step 1: Validate config (FAILS because template.yaml has no gliner_config: section)
python -m ptbr config --file ptbr/template.yaml --validate

# Step 2: Train (WORKS because training_cli.py expects model: section)
python -m ptbr train ptbr/template.yaml --validate
```

This means the `config` subcommand is completely non-functional for any YAML file that works with the `train` subcommand, and vice versa. The two CLIs validate **different schemas for different YAML layouts** and there is no bridge, adapter, or shared format between them.

**Root cause:** `config_cli.py` was written to validate the raw `GLiNERConfig` fields directly (as they appear in `gliner_config.json` on the Hub), while `training_cli.py` was written to validate a full experiment YAML with sectioned structure. Neither module knows about the other's format.

**Additional incompatibility -- LoRA section naming:**
- `config_cli.py` expects `lora_config:` (line 448), with fields like `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `bias`, `task_type`, `modules_to_save`, `fan_in_fan_out`, `use_rslora`, `init_lora_weights`.
- `training_cli.py` expects `lora:` (line 184), with fields like `enabled`, `r`, `lora_alpha`, `lora_dropout`, `bias`, `target_modules`, `task_type`, `modules_to_save`.
- Even the field sets differ: `config_cli.py` has `fan_in_fan_out`, `use_rslora`, `init_lora_weights`; `training_cli.py` has `enabled` (master switch). They are not interchangeable.

---

### __main__.py Routing Analysis

**File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/__main__.py`

The central dispatcher correctly registers three subcommands (`data`, `config`, `train`) but introduces an **inconsistent interface**:

1. **Config subcommand** (lines 95-113): Defined as a Typer callback, takes `--file` as a named option (`typer.Option(...)`). Usage: `python -m ptbr config --file config.yaml --validate`.

2. **Train subcommand** (line 120-122): Imports the `app` from `training_cli.py` and adds it as a sub-app. The training CLI's `main()` function (training_cli.py line 862-863) takes `config` as a **positional argument** (`typer.Argument(...)`). Usage: `python -m ptbr train config.yaml --validate`.

**Inconsistency:** The config path is a named option (`--file`) for the config subcommand but a positional argument for the train subcommand. This is a UX inconsistency that will confuse users.

3. **Data subcommand** (lines 27-86): Also uses `--file-or-repo` as a named option. This is consistent with config but not with train.

4. **No shared validation path:** The `config_cmd` callback (line 96) calls `load_and_validate_config()` which expects the `gliner_config:` format. The `train` subcommand calls `validate_config()` which expects the `model:` format. There is no common validation function, no shared schema, and no way to run `config` validation as a pre-flight check for `train`.

5. **Top-level import of training_cli** (line 120): `from ptbr.training_cli import app as _train_app` is executed at module load time. This means importing `__main__.py` (even just for the `config` or `data` subcommands) will execute `training_cli.py`'s module-level code, including setting up Rich logging handlers (training_cli.py lines 48-61). This is a side-effect that could cause issues with log output even when the user only wants to validate a config.

---

### Report's Canonical YAML vs Our Structure

Side-by-side comparison:

| Report Recommendation | Our `template.yaml` | Status |
|---|---|---|
| `experiment:` (metadata, seed, tags) | `run:` (name, description, tags, seed) | Renamed but functionally equivalent. Missing `data_seed`, `notes`. |
| `model:` (GLiNER config) | `model:` (GLiNER config) | Structurally aligned. |
| `peft:` with nested `lora:` sub-section | `lora:` (flat, no method/adapter_name) | Divergent. Report recommends `peft.enabled`, `peft.method`, `peft.adapter_name`, `peft.lora.r/alpha/...`. We have flat `lora.enabled`, `lora.r`, etc. Missing `method`, `adapter_name`, `init_lora_weights`, `use_rslora`. |
| `data:` with `format`, `fields`, `preprocessing` sub-sections | `data:` with only `root_dir`, `train_data`, `val_data_dir` | **Major gap.** No schema fields, no format declaration, no preprocessing/chunking config. |
| `training:` (full HF TrainingArguments + GLiNER extensions) | `training:` (subset of HF TrainingArguments) | Partial. Missing: `output_dir`, `overwrite_output_dir`, `save_safetensors`, `num_train_epochs`, `auto_find_batch_size`, `group_by_length`, `adam_beta1/beta2/epsilon`, `gradient_checkpointing`, `torch_compile` (as HF field), `evaluation_strategy`, `eval_steps`, `eval_delay`, `save_strategy`, `logging_strategy`, `logging_first_step`, `disable_tqdm`, `resume_from_checkpoint`, `remove_unused_columns`, `full_determinism`, `deepspeed`, `fsdp`, `hub_strategy`, `hub_private_repo`, `hub_always_push`, `run_name`, `load_best_model_at_end`, `metric_for_best_model`. |
| `logging:` (W&B detailed config) | `environment:` (partial W&B config) | Divergent naming and missing fields. Report recommends `logging.wandb.project/entity/group/job_type/mode/log_model`. We have `environment.wandb_project/entity/api_key` only. Missing `group`, `job_type`, `mode`, `log_model`. |

**Key structural divergences:**

1. Report uses `experiment:` for metadata; we use `run:`. This is cosmetic.
2. Report uses `peft:` with a nested `lora:` sub-section allowing future extensibility (e.g., QLoRA, prefix-tuning). We use a flat `lora:` section. This limits future extensibility.
3. Report's `data:` section is significantly richer, with document schema and preprocessing. Ours is minimal.
4. Report's `training:` section is a near-complete `TrainingArguments` mirror. Ours covers roughly 60% of the fields.
5. Report separates `logging:` from environment. We merge W&B settings into `environment:`.

---

### Missing Data Features

The research report explicitly recommends several data capabilities that are entirely absent from our implementation:

**1. Document schema fields** (report's `data.fields`):

The report recommends:
```yaml
data:
  fields:
    tokens: "tokenized_text"
    ner: "ner"
    relations: "relations"
```

Our `data.py` (`/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/data.py`) hardcodes the column names:
- `load_data()` (line 26-31): Takes `text_column` and `ner_column` as parameters but they default to `"tokenized_text"` and `"ner"`.
- The `data:` section in `template.yaml` has no field mapping configuration.
- The `training_cli.py` schema (lines 142-145) only validates `data.root_dir`, `data.train_data`, `data.val_data_dir` -- no field mapping at all.

**Impact:** If a user has a dataset with different column names (e.g., `"tokens"` instead of `"tokenized_text"`), the `data` subcommand supports remapping via CLI options, but the `train` subcommand has no mechanism for it. The training launcher (`_launch_training` at training_cli.py line 1064) does `json.load(f)` and passes the raw list directly to `model.train_model()` with no column mapping.

**2. Long document chunking** (report's `data.preprocessing.long_doc_chunking`):

The report recommends:
```yaml
data:
  preprocessing:
    long_doc_chunking:
      enabled: false
      max_words: 200
      stride: 50
      drop_empty_chunks: true
```

Our toolkit has **zero** preprocessing or chunking capabilities. Documents longer than `model.max_len` will simply be truncated by the tokenizer. There is no stride-based chunking, no option to split long documents into overlapping windows, and no way to configure this behavior.

**3. Missing preprocessing section entirely:**

The `data.py` module performs validation but not transformation. There is no:
- Sentence splitting
- Token normalization
- Entity boundary adjustment after chunking
- Train/validation split creation from a single file
- Data augmentation (entity mention substitution, negative sampling pre-processing)

---

### Parameter Forwarding Gaps

**Parameters validated by `training_cli.py` but NOT forwarded to `model.train_model()`:**

Examining `_launch_training()` (training_cli.py lines 997-1132) and `create_training_args()` (model.py lines 1001-1087):

| Parameter in Our Schema | Forwarded to `train_model()`? | Notes |
|---|---|---|
| `training.dataloader_pin_memory` | NO | Validated at line 174 but not passed at lines 1090-1132. |
| `training.dataloader_persistent_workers` | NO | Validated at line 175 but not passed. |
| `training.dataloader_prefetch_factor` | NO | Validated at line 176 but not passed. |
| `training.size_sup` | NO | Validated at line 179 but never used. Dead config. |
| `training.shuffle_types` | NO | Validated at line 180 but never forwarded. GLiNER internally handles this but our CLI doesn't forward it. |
| `training.random_drop` | NO | Validated at line 181 but never forwarded. Same issue. |
| `training.fp16` | Passed to `train_model()` at line 1124 | But `create_training_args()` does NOT have `fp16` as a named parameter (model.py line 1001-1027). It would need to go through `**kwargs`. |
| `training.label_smoothing` | Passed to `train_model()` at line 1114 | But `create_training_args()` does NOT have `label_smoothing` as a named parameter. It goes through `**kwargs` and reaches `TrainingArguments` only if the custom `TrainingArguments` in `trainer.py` accepts it. It does (trainer.py line 82). This works but is fragile. |
| `training.optim` | Passed as `optim=` (line 1109) | But `create_training_args()` does not have an `optim` named parameter. Uses `**kwargs`. Works but undocumented. |
| `run.name` | NO | Validated, used for resume checking and log naming, but never passed to `TrainingArguments.run_name`. W&B will not see the run name. |
| `run.tags` | NO | Validated and logged but never forwarded to W&B or TrainingArguments. |
| `run.description` | NO | Validated but unused beyond logging. |

**Parameters recommended by report but absent from our schema entirely:**

- `training.output_dir` -- our CLI uses `--output-folder` CLI flag instead, which is good, but means the YAML cannot specify it.
- `training.num_train_epochs` -- our schema only supports `num_steps`, not epoch-based training.
- `training.adam_beta1`, `training.adam_beta2`, `training.adam_epsilon` -- not in schema, not forwarded.
- `training.gradient_checkpointing` -- absent, important for large models.
- `training.torch_compile` (as HF TrainingArguments field) -- we have `training.compile_model` which calls `model.compile()` directly rather than using the HF `torch_compile` arg.
- `training.hub_strategy` -- absent, always pushes at end only.
- `training.run_name` -- absent; `run.name` exists but is never forwarded.
- `training.save_strategy`, `training.evaluation_strategy` -- absent; implied by `eval_every` field.
- `training.resume_from_checkpoint` -- we use `--resume` CLI flag, not a config field.
- `training.remove_unused_columns` -- absent. The report notes this is "often essential" for custom batch dictionaries.
- `training.load_best_model_at_end`, `training.metric_for_best_model` -- absent.
- `training.deepspeed`, `training.fsdp` -- absent, no distributed training support.
- `logging.wandb.group`, `logging.wandb.job_type`, `logging.wandb.mode`, `logging.wandb.log_model` -- absent.

---

### Upstream train.py Compatibility

**File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/train.py`

The upstream `train.py` uses `load_config_as_namespace()` from `gliner/utils.py`, which converts any YAML into nested `argparse.Namespace` objects by key. It then accesses:
- `cfg.model` -> passed to `GLiNER.from_config(model_cfg)` as a dict
- `cfg.training` -> used for training parameters
- `cfg.data.root_dir`, `cfg.data.train_data`, `cfg.data.val_data_dir`

**Compatibility assessment:**

Our `template.yaml` uses the same top-level section names (`model:`, `training:`, `data:`) that `train.py` expects. Therefore, `template.yaml` **can** be used with the upstream `train.py` with these caveats:

1. **Extra sections ignored:** The `run:`, `lora:`, and `environment:` sections in `template.yaml` are loaded into the namespace but never accessed by `train.py`. They are harmless.

2. **Missing fields cause `AttributeError`:** The upstream `train.py` accesses `cfg.training.loss_prob_margin` via `getattr(cfg.training, "loss_prob_margin", 0.0)` (line 79), which handles missing fields. But it directly accesses `cfg.training.num_steps`, `cfg.training.scheduler_type`, etc. (lines 65-86) without `getattr` fallbacks. If any of these are missing, it crashes.

3. **output_dir is hardcoded:** Upstream `train.py` passes `output_dir="models"` (line 63), ignoring `data.root_dir`. Our `training_cli.py` correctly uses the `--output-folder` flag.

4. **bf16 is hardcoded:** Upstream `train.py` passes `bf16=True` (line 91), ignoring the config's `training.bf16` field. Our `training_cli.py` correctly reads from config.

5. **eval_batch_size not forwarded:** Upstream `train.py` passes `per_device_eval_batch_size=cfg.training.train_batch_size` (line 70), always using the train batch size. Our `training_cli.py` has a proper fallback (line 1083).

6. **label_smoothing not forwarded:** Upstream `train.py` does not forward `label_smoothing` at all. Our `training_cli.py` does (line 1114).

**Verdict:** `template.yaml` works with upstream `train.py` for the basic case but several parameters validated by our toolkit will be silently ignored by upstream.

---

### End-to-End Workflow Gaps

**Scenario: A user tries the documented workflow from `__init__.py`:**

```bash
python -m ptbr config --file config.yaml --validate && python -m ptbr train config.yaml
```

**What happens:**

1. **If `config.yaml` follows `template.yaml` structure (model:/training:/data:):**
   - `python -m ptbr config --file config.yaml --validate` **FAILS** immediately.
   - Error: `"Missing 'gliner_config' section in YAML file."` (config_cli.py line 418)
   - The `&&` short-circuits. Training never starts.
   - The user sees a confusing error about a section they never intended to write.

2. **If `config.yaml` follows the `config_cli.py` structure (gliner_config:/lora_config:):**
   - `python -m ptbr config --file config.yaml --validate` **SUCCEEDS** (assuming valid GLiNER fields).
   - `python -m ptbr train config.yaml` **FAILS** at multiple points:
     - `validate_config()` will report all `model.*`, `run.*`, `data.*`, `training.*` fields as MISSING/REQUIRED errors.
     - Even if validation were skipped, `_launch_training()` would crash with `KeyError: 'model'` at line 1043.

3. **There is no YAML format that satisfies both CLIs simultaneously.** The `config` subcommand demands `gliner_config:` at the top level (which would be an unknown/extra key to `training_cli.py`), and the `train` subcommand demands `model:` at the top level (which `config_cli.py` would ignore completely while erroring on the absent `gliner_config:` key).

**Additional workflow gaps:**

- **No `init` or `template` command:** There is no way to generate a starter YAML from the CLI. Users must find and copy `template.yaml` manually.
- **No format migration:** If a user has a `gliner_config.json` from the Hub, there is no tool to convert it into our YAML format.
- **No dry-run for training:** The `--validate` flag on the train subcommand exits after validation (training_cli.py line 967-968), but it still requires `--output-folder` to be unset or empty (line 971-973), meaning you cannot validate a config that is already partially trained.
- **Log file pollution:** Both CLIs create log/summary files in the config directory. Running validation repeatedly creates `validation_*.log`, `summary_*.txt`, and `config_validation_*.json` files with no cleanup mechanism.

---

### Concrete Recommendations

**Priority 1 (Showstoppers):**

1. **Unify the YAML format between `config_cli.py` and `training_cli.py`.** The most pragmatic approach is to make `config_cli.py` understand the `model:` structure used by `training_cli.py` and `template.yaml`. Specifically:
   - `config_cli.py` line 417: Change `if "gliner_config" not in raw:` to also accept `if "model" not in raw:` and treat the `model:` section as the source of GLiNER config fields.
   - `config_cli.py` line 448: Change `if "lora_config" not in raw:` to also accept `if "lora" not in raw:`.
   - Alternatively, rewrite `config_cli.py` to consume the same `_FIELD_SCHEMA` structure from `training_cli.py`, extracting only the `model.*` and `lora.*` fields for its GLiNER/LoRA validation.
   - **Files:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/config_cli.py` lines 417-464.

2. **Forward `run.name` as `run_name` to `TrainingArguments`.** Without this, W&B runs are unnamed.
   - **File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`, `_launch_training()`, around line 1090. Add `run_name=cfg["run"]["name"]` to the `model.train_model()` call.

**Priority 2 (Important):**

3. **Forward missing dataloader parameters.** `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor` are validated but never passed.
   - **File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py` lines 1090-1132. Add the three parameters to the `model.train_model()` call.

4. **Remove dead config fields or wire them.** `training.size_sup`, `training.shuffle_types`, `training.random_drop` are validated but never forwarded. Either remove them from `_FIELD_SCHEMA` (training_cli.py lines 179-181) or forward them to the trainer.

5. **Add `remove_unused_columns=False` as a default.** The report explicitly warns this is "often essential for custom batch dictionaries." GLiNER uses custom data collators, making this critical.
   - **File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py` line 1090-1132. Add `remove_unused_columns=False` to the `model.train_model()` call.

6. **Add `data.fields` schema to template.yaml and training_cli schema.** Allow users to specify custom column name mappings so that `_launch_training()` can remap columns before passing to the trainer.
   - **Files:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/template.yaml` (add `fields:` sub-section under `data:`), `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py` (add schema entries, update `_launch_training()`).

7. **Standardize CLI argument style.** The `config` subcommand uses `--file` (a named option) while `train` uses a positional argument. Either make both positional or both named.
   - **Files:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/__main__.py` lines 97 and 120-122.

8. **Add `gradient_checkpointing` to the schema.** This is essential for fine-tuning large models on consumer GPUs and is a standard `TrainingArguments` field.
   - **File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py` `_FIELD_SCHEMA`, add entry for `training.gradient_checkpointing`.

**Priority 3 (Nice-to-have):**

9. **Add long-document chunking to `data.py`.** Implement the report's recommended `data.preprocessing.long_doc_chunking` with `max_words`, `stride`, and `drop_empty_chunks` parameters.
   - **File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/data.py`.

10. **Add a `peft:` wrapper section** around the LoRA config to allow future extensibility (QLoRA, prefix-tuning, IA3). Include `method` and `adapter_name` fields as the report recommends.
    - **Files:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/template.yaml`, `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`.

11. **Add distributed training placeholders** (`deepspeed`, `fsdp`) to the schema, even if not yet wired. This prevents users from having to modify the schema later.
    - **File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py` `_FIELD_SCHEMA`.

12. **Add `hub_strategy`, `hub_private_repo`, `hub_always_push`** to the environment section to match the report's Hub integration recommendations.
    - **Files:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/template.yaml`, `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py`.

13. **Add a `ptbr init` command** that generates a starter YAML from the template, optionally with a model name pre-filled.
    - **File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/__main__.py`.

14. **Lazy-import training_cli in __main__.py** to avoid module-level side effects (Rich handler setup) when only using `config` or `data` subcommands.
    - **File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/__main__.py` line 120. Move the import inside a function or use Typer's lazy loading pattern.

15. **Add W&B run tags forwarding.** The `run.tags` field is validated but never forwarded to W&B via `WANDB_TAGS` environment variable or TrainingArguments.
    - **File:** `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/training_cli.py` `_launch_training()`, around line 1030-1040.
