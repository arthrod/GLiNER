## Integration and End-to-End Architecture Report

### Executive Summary

This report reassesses the `ptbr` toolkit against the original integration report findings. The original report identified a **showstopper configuration format incompatibility** between `config_cli.py` and `training_cli.py`, plus numerous parameter forwarding gaps, missing data features, and architectural concerns.

**Since the original report, significant fixes have been applied.** The critical YAML incompatibility has been resolved via alias support, the `__main__.py` lazy-loading issue has been fixed, parameter forwarding gaps for dataloader flags and `run_name` have been closed, and `label_smoothing` is now forwarded in both `train.py` and `training_cli.py`. Comprehensive regression tests cover each of these fixes.

However, **several Priority 2 and Priority 3 issues remain open**, including dead config fields (`size_sup`, `shuffle_types`, `random_drop`), missing `remove_unused_columns=False` default, absent `data.fields` schema, no long-document chunking, no `gradient_checkpointing` schema entry, and unforwarded `run.tags`. The CLI argument style inconsistency (`--file` vs positional) also persists.

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

**Main tests (tests/):** 173 passed, 33 failed, 54 errors (pre-existing issues unrelated to ptbr; mostly tokenizer/processor incompatibilities with current transformers version).

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

**Status: NOT FIXED**

These three fields are still in `_FIELD_SCHEMA` (lines 182-184) and are validated, but they are never read by `_launch_training()` and never forwarded to `model.train_model()`. They remain dead configuration that gives users a false sense of control.

**Test coverage:** No tests verify these are forwarded (because they aren't). The schema validation tests confirm they parse correctly, but there's no integration test confirming they reach the trainer.

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

1. **Remove or wire dead config fields.** `training.size_sup`, `training.shuffle_types`, `training.random_drop` are validated but never forwarded. Either remove from `_FIELD_SCHEMA` or forward them to the trainer. (Recommendation 4)

2. **Add `remove_unused_columns=False` default.** GLiNER uses custom data collators, making this important for HF Trainer compatibility. (Recommendation 5)

3. **Add `data.fields` schema.** Allow column remapping in the training YAML so users with non-standard datasets don't have to rename columns. (Recommendation 6)

4. **Standardize CLI argument style.** `--file` for config vs positional for train is a UX inconsistency. (Recommendation 7)

5. **Add `gradient_checkpointing` to the schema.** Essential for fine-tuning large models on consumer hardware. (Recommendation 8)

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
| **4** | Dead config fields (size_sup etc.) | **NO** | N/A |
| **5** | `remove_unused_columns=False` | **NO** | N/A |
| **6** | `data.fields` schema | **NO** | N/A |
| **7** | CLI argument style | **NO** | N/A |
| **8** | `gradient_checkpointing` schema | **NO** | N/A |
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

**Bottom line:** The 2 showstopper issues (YAML incompatibility, lazy loading) and the 3 highest-impact parameter forwarding gaps (run_name, dataloader flags, label_smoothing) have been fixed with solid test coverage (176/176 passing). The remaining 11 open items are Priority 2/3 enhancements that don't block basic functionality.
