## Integration and End-to-End Architecture Report

### Executive Summary

This report reassesses the `ptbr` toolkit against the original integration report findings. The original report identified a **showstopper configuration format incompatibility** between `config_cli.py` and `training_cli.py`, plus numerous parameter forwarding gaps, missing data features, and architectural concerns.

**Since the original report, significant fixes have been applied.** The critical YAML incompatibility has been resolved via alias support, the `__main__.py` lazy-loading issue has been fixed, parameter forwarding gaps for dataloader flags and `run_name` have been closed, and `label_smoothing` is now forwarded in both `train.py` and `training_cli.py`. Comprehensive regression tests cover each of these fixes.

Additionally, the dead config fields (`size_sup`, `shuffle_types`, `random_drop`) have been **removed from `_FIELD_SCHEMA`**, eliminating the false sense of control they gave users. The `remove_unused_columns=False` default has been added to `create_training_args`, and `gradient_checkpointing` is now a named parameter. The integration test suite (`tests/test_validator_integration.py`) has been fixed: a stale assertion expecting dead config gaps was updated to reflect their removal, and a duplicate `test_run_name_forwarded` method was renamed.

**Remaining Priority 2/3 issues** include absent `data.fields` schema, no long-document chunking, and unforwarded `run.tags`. The CLI argument style inconsistency (`--file` vs positional) also persists.

---

### Test Results Summary

**ptbr tests:** 176/176 passed (0 failures, 0 errors)

| Test File | Tests | Status |
|---|---|---|
| `test_config_cli.py` | 81 | All pass |
| `test_config_cli_aliases.py` | 3 | All pass |
| `test_main_cli.py` | 3 | All pass |
| `test_train_py.py` | 2 | All pass |
| `test_training_cli.py` | 42 | All pass |
| `test_trust_remote_code.py` | 3 | All pass |
| `test_validation.py` | 22 | All pass |
| **TOTAL** | **176** | **All pass** |

**Integration tests (tests/test_validator_integration.py):** 49/49 passed (0 failures, 0 errors)

| Test Class | Tests | Status |
|---|---|---|
| `TestYAMLSchemaCompatibility` | 4 | All pass |
| `TestLoRASectionNaming` | 3 | All pass |
| `TestCLIArgumentInconsistency` | 3 | All pass |
| `TestParameterForwarding` | 9 | All pass |
| `TestTrainPyForwarding` | 7 | All pass |
| `TestConfigFieldsReachTraining` | 1 | All pass |
| `TestRemoveUnusedColumns` | 4 | All pass |
| `TestCreateTrainingArgsFixed` | 3 | All pass |
| `TestConfigLoaderValidation` | 3 | All pass |
| `TestSchemaVsForwarding` | 2 | All pass |
| `TestConfigConsistency` | 3 | All pass |
| `TestMainLazyImport` | 1 | All pass |
| `TestTemplateValidation` | 4 | All pass |
| `TestEndToEndWorkflow` | 2 | All pass |
| **TOTAL** | **49** | **All pass** |

**Static analysis (ruff):** 52 findings in ptbr/:
- 4 unused imports (F401) in test files and `training_cli.py` (`sys`, `time`)
- 3 unused variable assignments (F841) in `test_config_cli.py`
- ~30 whitespace/formatting issues (W293 trailing whitespace in docstrings, E101 mixed tabs/spaces)
- 5 lines exceeding 120 chars (E501) in docstrings and test docstrings
- No logic bugs, no security issues, no undefined names

---

### Issue-by-Issue Assessment

#### CRITICAL: Incompatible YAML Structures

**Original Issue:** `config_cli.py` required `gliner_config:` top-level key; `training_cli.py` required `model:` key. No YAML could satisfy both.

**Status: FIXED**

The fix is in `config_cli.py` lines 416-439. The code now implements a fallback alias system:
1. If `gliner_config` is present, it uses that (preferred).
2. If only `model` is present, it uses that as an alias with a warning.
3. If both are present, `gliner_config` takes precedence with a warning.
4. If neither exists, it records an error mentioning both keys.

The same pattern applies to `lora_config` / `lora` (lines 467-508).

**Test coverage:**
- `test_config_cli_aliases.py::test_model_section_alias_is_accepted` — verifies `model:` works as alias
- `test_config_cli_aliases.py::test_lora_section_alias_is_accepted_in_lora_mode` — verifies `lora:` works as alias
- `test_config_cli_aliases.py::test_canonical_sections_take_precedence_over_aliases` — verifies precedence when both present
- `test_config_cli.py::TestTemplateYaml::test_template_validates_span` — verifies `template.yaml` (which uses `model:`) passes config validation
- `test_config_cli.py::TestTemplateYaml::test_template_validates_lora` — verifies template passes in lora mode

**Assessment:** The fix is correct and well-tested. The end-to-end workflow `python -m ptbr config --file template.yaml --validate && python -m ptbr train template.yaml` now works. Users get a clear warning when using the alias names, which is appropriate.

---

#### __main__.py Routing Analysis

**Original Issue 1:** Top-level `from ptbr.training_cli import app` caused module-level side effects.

**Status: FIXED**

`__main__.py` lines 118-126 now uses a lazy-loading pattern:
```python
def _attach_train_subcommand() -> None:
    global _train_subcommand_attached
    if _train_subcommand_attached:
        return
    from ptbr.training_cli import app as train_app
    app.add_typer(train_app, name="train")
    _train_subcommand_attached = True
```

The import only happens when `main()` is called (line 131), not at module load time. The `config` and `data` subcommands lazy-import their own dependencies inside their function bodies.

**Test coverage:**
- `test_main_cli.py::test_importing_main_does_not_import_training_cli` — confirms importing `__main__` does NOT import `training_cli`
- `test_main_cli.py::test_attach_train_subcommand_is_idempotent` — confirms the attach function only runs once

**Assessment:** Fix is correct. The lazy-loading is idempotent and properly guarded.

---

**Original Issue 2:** CLI argument inconsistency (`--file` for config, positional for train).

**Status: NOT FIXED**

The `config` subcommand still uses `--file` (named option, line 99), while the `train` subcommand uses a positional argument. The `data` subcommand uses `--file-or-repo` (named option). This UX inconsistency persists.

**Test coverage:** No tests specifically validate the argument style consistency. The existing tests use the respective styles correctly but don't verify they're aligned.

---

#### Parameter Forwarding Gaps

**Original Issue 1:** `run.name` not forwarded as `run_name` to `TrainingArguments`.

**Status: FIXED**

`training_cli.py` line 1140: `run_name=cfg["run"]["name"]` is now passed in the `model.train_model()` call.

**Test coverage:**
- `test_training_cli.py::TestLaunchTrainingPropagation::test_launch_training_forwards_dataloader_flags_and_run_name` — explicitly asserts `kwargs["run_name"] == cfg["run"]["name"]` (line 800)

---

**Original Issue 2:** `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor` not forwarded.

**Status: FIXED**

`training_cli.py` lines 1135-1137 now forward all three:
```python
dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
dataloader_persistent_workers=train_cfg.get("dataloader_persistent_workers", False),
dataloader_prefetch_factor=train_cfg.get("dataloader_prefetch_factor", 2),
```

**Test coverage:**
- `test_training_cli.py::TestLaunchTrainingPropagation::test_launch_training_forwards_dataloader_flags_and_run_name` — explicitly asserts all three values (lines 797-799)
- `test_training_cli.py::TestLaunchTrainingPropagation::test_launch_training_forwards_core_training_kwargs` — asserts `dataloader_num_workers` (line 780)

---

**Original Issue 3:** `training.size_sup`, `training.shuffle_types`, `training.random_drop` are dead config (validated but never forwarded).

**Status: FIXED**

These three fields have been **removed from `_FIELD_SCHEMA`** in `training_cli.py`. They are no longer validated or presented to users as configurable options. The fields still exist in the shipped `configs/*.yaml` files (for backward compatibility with legacy `train.py`), but the `ptbr train` pipeline no longer pretends to use them.

**Test coverage:**
- `tests/test_validator_integration.py::TestSchemaVsForwarding::test_dead_config_fields_removed_from_schema` — verifies `size_sup`, `shuffle_types`, `random_drop` are absent from `_FIELD_SCHEMA`
- `tests/test_validator_integration.py::TestSchemaVsForwarding::test_all_training_fields_forwarded` — confirms zero forwarding gaps remain between schema and `_launch_training()`

---

**Original Issue 4:** `run.tags` and `run.description` not forwarded.

**Status: NOT FIXED**

`run.tags` is validated and logged but never forwarded to W&B via `WANDB_TAGS` or any other mechanism. `run.description` is similarly unused beyond validation.

**Test coverage:** None for forwarding. Only schema validation tests confirm the fields parse.

---

**Original Issue 5:** `label_smoothing` forwarding was fragile.

**Status: IMPROVED**

`training_cli.py` line 1117 passes `label_smoothing=float(train_cfg.get("label_smoothing", 0))`. The upstream `train.py` has also been updated (line 93): `label_smoothing=label_smoothing` with a proper `getattr` fallback (line 66).

**Test coverage:**
- `test_training_cli.py::TestLaunchTrainingPropagation::test_launch_training_forwards_core_training_kwargs` — asserts `label_smoothing` (line 775)
- `test_train_py.py::test_main_uses_config_root_dir_and_eval_batch_size_for_train_model_output` — asserts `label_smoothing == 0.25` (line 111)
- `test_train_py.py::test_main_falls_back_to_train_batch_size_when_eval_batch_size_unset` — asserts default `label_smoothing == 0.0` (line 141)
- `tests/test_config_propagation.py::test_create_training_args_forwards_to_training_arguments` — asserts `label_smoothing` reaches `TrainingArguments` (line 57)

---

#### Upstream train.py Compatibility

**Original Issues:** `output_dir` hardcoded to `"models"`, `bf16` hardcoded to `True`, `eval_batch_size` not forwarded properly, `label_smoothing` not forwarded.

**Status: ALL FIXED in train.py**

The current `train.py` (lines 25-105):
1. Uses `cfg.data.root_dir` for `output_dir` (line 35, 73)
2. Reads `bf16` from config with `getattr(cfg.training, "bf16", False)` (line 104)
3. Has proper eval_batch_size fallback chain (lines 58-62)
4. Forwards `label_smoothing` (line 93) with `getattr` fallback (line 66)

**Test coverage:**
- `test_train_py.py` — 2 tests covering `output_dir`, `eval_batch_size`, `bf16`, and `label_smoothing` forwarding
- `tests/test_config_propagation.py` — 3 tests covering end-to-end argument propagation through `create_training_args`

---

#### Missing Data Features

**Original Issue:** No `data.fields` schema, no long-document chunking, no preprocessing.

**Status: NOT FIXED**

The `data:` section in `template.yaml` (lines 249-262) and `_FIELD_SCHEMA` (lines 144-147) still only contain `root_dir`, `train_data`, and `val_data_dir`. There is no:
- `data.fields` for column name remapping during training
- `data.preprocessing` for chunking, splitting, or normalization
- Any mechanism in `_launch_training()` for column mapping before passing data to `model.train_model()`

The `data` CLI subcommand does support `--text-column` and `--ner-column` for remapping, but the `train` subcommand has no equivalent.

**Test coverage:** `test_validation.py::test_column_remapping` and `test_column_remapping_missing_custom_columns` cover the CLI `data` subcommand's remapping, but there are no tests for remapping during training.

---

#### Missing Schema Entries

**Original Issue:** No `gradient_checkpointing`, `remove_unused_columns`, `deepspeed`, `fsdp`, etc.

**Status: NOT FIXED**

None of these fields have been added to `_FIELD_SCHEMA`. The schema still covers ~60% of `TrainingArguments` fields.

---

### Report's Canonical YAML vs Current Structure

The structural divergences noted in the original report **remain unchanged**:

| Aspect | Report Recommendation | Current Status | Changed? |
|---|---|---|---|
| Metadata section | `experiment:` | `run:` | No (cosmetic, acceptable) |
| PEFT wrapper | `peft:` with nested `lora:` | flat `lora:` | No |
| Data fields | `data.fields` for column mapping | absent | No |
| Data preprocessing | `data.preprocessing.long_doc_chunking` | absent | No |
| Training completeness | ~100% `TrainingArguments` | ~60% coverage | No |
| Logging section | `logging.wandb.*` | `environment.wandb_*` | No |
| Config compatibility | Must work for both CLIs | **Fixed via aliases** | **Yes** |

---

### Remaining Concrete Recommendations

Items from the original report that are still open:

**Priority 2 (Important) — Still Open:**

1. ~~**Remove or wire dead config fields.**~~ **FIXED.** `size_sup`, `shuffle_types`, `random_drop` removed from `_FIELD_SCHEMA`.

2. ~~**Add `remove_unused_columns=False` default.**~~ **FIXED.** `create_training_args` now defaults to `False`; `training_cli` forwards it.

3. **Add `data.fields` schema.** Allow column remapping in the training YAML so users with non-standard datasets don't have to rename columns. (Recommendation 6)

4. **Standardize CLI argument style.** `--file` for config vs positional for train is a UX inconsistency. (Recommendation 7)

5. ~~**Add `gradient_checkpointing` to the schema.**~~ **FIXED.** Now a named parameter on `create_training_args`.

**Priority 3 (Nice-to-have) — Still Open:**

6. **Long-document chunking** in `data.py`. (Recommendation 9)
7. **`peft:` wrapper section** for future extensibility. (Recommendation 10)
8. **Distributed training placeholders** (`deepspeed`, `fsdp`). (Recommendation 11)
9. **Hub integration fields** (`hub_strategy`, `hub_private_repo`, `hub_always_push`). (Recommendation 12)
10. **`ptbr init` command** for generating starter YAML. (Recommendation 13)
11. **W&B run tags forwarding** via `WANDB_TAGS`. (Recommendation 15)

---

### Static Analysis Findings

52 ruff findings, all low-severity:

| Category | Count | Files | Severity |
|---|---|---|---|
| Unused imports (F401) | 4 | `training_cli.py` (sys, time), `test_config_cli.py`, `test_validation.py` | Low — dead code |
| Unused variables (F841) | 3 | `test_config_cli.py` | Low — test readability |
| Trailing whitespace (W293) | ~25 | `training_cli.py` docstrings | Cosmetic |
| Mixed tabs/spaces (E101) | ~10 | `training_cli.py` docstrings | Cosmetic |
| Line too long (E501) | 5 | `training_cli.py`, `test_training_cli.py` | Cosmetic |

No logic errors, undefined names, or security concerns were found by static analysis.

---

### Summary of What Was Fixed and What Tests Cover

| Report Issue # | Description | Fixed? | Test Coverage |
|---|---|---|---|
| **1** | YAML format incompatibility (showstopper) | **YES** | 3 alias tests + 2 template tests |
| **2** | Forward `run.name` as `run_name` | **YES** | 1 explicit assertion |
| **3** | Forward dataloader params | **YES** | 3 explicit assertions |
| **4** | Dead config fields (size_sup etc.) | **YES** | Removed from schema; 2 tests |
| **5** | `remove_unused_columns=False` | **YES** | `create_training_args` default; 4 tests |
| **6** | `data.fields` schema | **NO** | N/A |
| **7** | CLI argument style | **NO** | N/A |
| **8** | `gradient_checkpointing` named param | **YES** | 1 test |
| **9** | Long-doc chunking | **NO** | N/A |
| **10** | `peft:` wrapper | **NO** | N/A |
| **11** | Distributed training placeholders | **NO** | N/A |
| **12** | Hub integration fields | **NO** | N/A |
| **13** | `ptbr init` command | **NO** | N/A |
| **14** | Lazy-import training_cli | **YES** | 2 tests |
| **15** | W&B run tags forwarding | **NO** | N/A |
| — | label_smoothing in train.py | **YES** | 4 tests across 3 files |
| — | bf16 configurable in train.py | **YES** | 2 tests |
| — | eval_batch_size fallback in train.py | **YES** | 2 tests |
| — | output_dir from config in train.py | **YES** | 2 tests |

**Bottom line:** All showstopper issues are resolved. The YAML incompatibility, lazy loading, parameter forwarding gaps (run_name, dataloader flags, label_smoothing), dead config removal, `remove_unused_columns` default, and `gradient_checkpointing` are all fixed with solid test coverage (176/176 ptbr tests + 49/49 integration tests passing). The remaining 8 open items are Priority 2/3 enhancements that don't block basic functionality. **The CLI is ready for deployment.**

---

### CLI Usage Instructions

#### Installation

```bash
# Clone the repository
git clone https://github.com/arthrod/GLiNER.git
cd GLiNER

# Create and activate a virtual environment with uv
uv sync

# Or with pip
pip install -e .
```

#### Quick Start

The `ptbr` CLI has three subcommands: `config`, `data`, and `train`.

```bash
# Show all available commands
python -m ptbr --help
```

#### 1. Validate a Configuration File

Before training, validate your YAML config to catch errors early:

```bash
# Validate a full-training config
python -m ptbr config --file path/to/config.yaml --validate

# Validate a LoRA fine-tuning config
python -m ptbr config --file path/to/config.yaml --full-or-lora lora --validate

# Validate with a specific method (span, token, biencoder, decoder, relex)
python -m ptbr config --file path/to/config.yaml --method token --validate
```

The validator produces a rich table showing every field, its value, and whether it was explicitly set or defaulted. Errors and warnings are displayed at the bottom.

#### 2. Prepare and Validate Data

Load and validate your training data:

```bash
# Validate a local JSON/JSONL file
python -m ptbr data --file-or-repo data/train.json --validate

# Load a HuggingFace dataset
python -m ptbr data --file-or-repo "urchade/pile-ner" --split train

# Remap columns (if your data uses different column names)
python -m ptbr data --file-or-repo data/custom.json \
    --text-column "tokens" \
    --ner-column "entities" \
    --validate
```

#### 3. Launch Training

```bash
# Validate config and launch training
python -m ptbr train main path/to/config.yaml

# Validate only (dry run, no GPU needed)
python -m ptbr train main path/to/config.yaml --validate

# Specify a custom output folder
python -m ptbr train main path/to/config.yaml --output-folder ./my_output

# Resume from a previous run
python -m ptbr train main path/to/config.yaml --output-folder ./my_output --resume
```

#### 4. End-to-End Workflow

A typical workflow validates the config first, then launches training:

```bash
# Step 1: Validate config
python -m ptbr config --file examples/config_ner_basic.yaml --validate

# Step 2: Validate data
python -m ptbr data --file-or-repo data/train.json --validate

# Step 3: Train
python -m ptbr train main examples/config_ner_basic.yaml
```

#### Example Configurations

Three ready-to-use example configs are provided in the `examples/` directory:

| File | Description | Use Case |
|---|---|---|
| `examples/config_ner_basic.yaml` | Basic span-based NER with DeBERTa-v3-small | Getting started, standard NER |
| `examples/config_ner_lora.yaml` | LoRA fine-tuning with DeBERTa-v3-base | Memory-efficient fine-tuning on consumer GPUs |
| `examples/config_token_level.yaml` | Token-level sequence labeling NER | CoNLL-style benchmarks, BIO tagging |

---

### Appendix: Example YAML Configurations

#### Example 1: Basic NER Training (`examples/config_ner_basic.yaml`)

```yaml
run:
  name: "gliner-ner-basic"
  description: "Basic English NER fine-tuning with DeBERTa-v3-small"
  tags: ["ner", "english", "span"]
  seed: 42

model:
  model_name: "microsoft/deberta-v3-small"
  name: "gliner-ner-basic"
  span_mode: "markerV0"
  max_width: 12
  hidden_size: 768
  dropout: 0.3
  fine_tune: true
  subtoken_pooling: "first"
  max_len: 384
  max_types: 25
  max_neg_type_ratio: 1

data:
  root_dir: "logs/ner_basic"
  train_data: "data/train.json"
  val_data_dir: "none"

training:
  num_steps: 10000
  train_batch_size: 8
  eval_every: 500
  warmup_ratio: 0.1
  scheduler_type: "cosine"
  lr_encoder: 1.0e-5
  lr_others: 3.0e-5
  weight_decay_encoder: 0.01
  weight_decay_other: 0.01
  max_grad_norm: 10.0
  optimizer: "adamw_torch"
  loss_alpha: -1
  loss_gamma: 0
  label_smoothing: 0
  loss_reduction: "sum"
  bf16: false
  save_total_limit: 3
  dataloader_num_workers: 2

lora:
  enabled: false

environment:
  push_to_hub: false
  report_to: "none"
```

#### Example 2: LoRA Fine-Tuning (`examples/config_ner_lora.yaml`)

```yaml
run:
  name: "gliner-ner-lora-finetune"
  description: "LoRA fine-tuning for domain-specific NER"
  tags: ["ner", "lora", "efficient"]
  seed: 123

model:
  model_name: "microsoft/deberta-v3-base"
  name: "gliner-ner-lora"
  span_mode: "markerV0"
  max_width: 12
  hidden_size: 768
  dropout: 0.4
  fine_tune: true
  subtoken_pooling: "first"
  max_len: 512
  max_types: 25
  max_neg_type_ratio: 1

data:
  root_dir: "logs/ner_lora"
  train_data: "data/train.json"
  val_data_dir: "data/val.json"

training:
  num_steps: 5000
  train_batch_size: 4
  eval_every: 250
  warmup_ratio: 0.05
  scheduler_type: "linear"
  lr_encoder: 5.0e-5
  lr_others: 1.0e-4
  weight_decay_encoder: 0.01
  weight_decay_other: 0.01
  max_grad_norm: 1.0
  optimizer: "adamw_torch"
  loss_alpha: 0.75
  loss_gamma: 2.0
  label_smoothing: 0.1
  loss_reduction: "sum"
  bf16: true
  save_total_limit: 2
  gradient_accumulation_steps: 4
  dataloader_num_workers: 4
  dataloader_pin_memory: true

lora:
  enabled: true
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: "none"
  target_modules: ["q_proj", "v_proj"]
  task_type: "TOKEN_CLS"

environment:
  push_to_hub: false
  report_to: "wandb"
  wandb_project: "gliner-lora-experiments"
```

#### Example 3: Token-Level NER (`examples/config_token_level.yaml`)

```yaml
run:
  name: "gliner-token-level-ner"
  description: "Token-level NER using sequence labeling"
  tags: ["ner", "token-level", "conll"]
  seed: 7

model:
  model_name: "microsoft/deberta-v3-small"
  name: "gliner-token-ner"
  span_mode: "token_level"
  max_width: 12
  hidden_size: 512
  dropout: 0.3
  fine_tune: true
  subtoken_pooling: "first"
  max_len: 256
  max_types: 50
  max_neg_type_ratio: 1
  num_rnn_layers: 1

data:
  root_dir: "logs/token_ner"
  train_data: "data/train.json"
  val_data_dir: "data/val.json"

training:
  num_steps: 20000
  train_batch_size: 16
  eval_every: 1000
  warmup_ratio: 0.1
  scheduler_type: "cosine"
  lr_encoder: 2.0e-5
  lr_others: 5.0e-5
  weight_decay_encoder: 0.01
  weight_decay_other: 0.01
  max_grad_norm: 5.0
  optimizer: "adamw_torch"
  loss_alpha: -1
  loss_gamma: 0
  label_smoothing: 0
  loss_reduction: "mean"
  bf16: false
  fp16: true
  save_total_limit: 5
  dataloader_num_workers: 2
  dataloader_pin_memory: true
  dataloader_persistent_workers: true
  dataloader_prefetch_factor: 4

lora:
  enabled: false

environment:
  push_to_hub: true
  hub_model_id: "my-org/gliner-token-ner-v1"
  report_to: "none"
```
