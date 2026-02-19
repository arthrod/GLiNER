# Integration Report Assessment: Current Code vs. Original Report

## Methodology

This assessment compares the original `integration_report.md` (written before code changes) against the current state of the codebase. The analysis was performed by:

1. Reading all source files in `ptbr/` (training_cli.py, config_cli.py, data.py, __main__.py, template.yaml)
2. Reading all test files in `ptbr/tests/` and `tests/` related to integration issues
3. Running the full test suite (306 tests across 4 suites)
4. Running static analysis (flake8, ruff, py_compile, ast.parse)

---

## Test Suite Summary

| Suite | Passed | Failed | xfail | Errors | Total |
|-------|--------|--------|-------|--------|-------|
| `ptbr/tests/` | **176** | 0 | 0 | 0 | 176 |
| `tests/test_validator_integration.py` | 32 | 15 | 0 | 0 | 47 |
| `tests/test_training_validation.py` | 20 | 0 | 22 | 0 | 42 |
| `tests/test_config_forwarding.py` | 7 | 14 | 0 | 20 | 41 |
| **Total** | **235** | **29** | **22** | **20** | **306** |

The ptbr test suite (176 tests) is **fully green**. Failures in `test_validator_integration.py` are mostly **inverted assertions** -- tests that documented bugs now fail because those bugs have been fixed. The 20 errors in `test_config_forwarding.py` are all import failures (`from gliner import GLiNER` unavailable in the test environment), not code bugs.

---

## Focus Area 1: Long-Doc Chunking

### Report Recommendation (Priority 3, Item 9)
The report recommended implementing `data.preprocessing.long_doc_chunking` with `max_words`, `stride`, and `drop_empty_chunks` parameters in `data.py`.

### Current Implementation Status: **NOT IMPLEMENTED**

- `ptbr/data.py` has no chunking, stride, or preprocessing logic
- No `preprocessing` sub-section exists in `template.yaml`
- No `data.fields` or `data.preprocessing` entries in `_FIELD_SCHEMA` (`training_cli.py`)
- grep for `chunk|stride|max_words|drop_empty` across `ptbr/` returns zero matches

### Test Coverage: **NO TESTS**

- No tests exist for long-doc chunking since the feature isn't implemented
- `tests/test_infer_packing.py` tests inference request packing (a different concern)

### Verdict: **STILL OUTSTANDING**

This Priority 3 recommendation has not been addressed. Documents exceeding `model.max_len` will be silently truncated by the tokenizer with no overlapping-window strategy available.

---

## Focus Area 2: peft: Wrapper

### Report Recommendation (Priority 3, Item 10)
The report recommended wrapping the flat `lora:` section in a `peft:` parent section with `method` and `adapter_name` fields for future extensibility (QLoRA, prefix-tuning, IA3).

### Current Implementation Status: **NOT IMPLEMENTED**

- `template.yaml` still uses a flat `lora:` section (lines 458-492)
- `_FIELD_SCHEMA` in `training_cli.py` defines `lora.*` fields (lines 187-194), not `peft.lora.*`
- `_apply_lora()` in `training_cli.py` (lines 1148-1207) only supports standard LoRA
- No `method`, `adapter_name`, `init_lora_weights`, or `use_rslora` fields in `training_cli.py`'s schema
- `config_cli.py` does define `init_lora_weights` and `use_rslora` in its `_LORA_RULES` (lines 114-115) but these are in the separate config validation path, not the training path

### Test Coverage: **PARTIAL (documents the divergence)**

- `test_validator_integration.py::TestLoRASectionNaming::test_lora_field_sets_differ_between_clis` (line 228) -- **PASSES** -- confirms that config_cli has `use_rslora`, `init_lora_weights`, `fan_in_fan_out` that training_cli lacks, and training_cli has `enabled` that config_cli lacks
- `ptbr/tests/test_training_cli.py` tests the existing `_apply_lora` function (LoRA application path works)

### Verdict: **STILL OUTSTANDING**

The flat `lora:` structure remains. No `peft:` wrapper, no `method`/`adapter_name` extensibility, and the LoRA field sets remain divergent between `config_cli.py` and `training_cli.py`.

---

## Focus Area 3: Distributed Training Placeholders

### Report Recommendation (Priority 3, Item 11)
Add `deepspeed` and `fsdp` fields to `_FIELD_SCHEMA`, even if not yet wired, to prevent users from having to modify the schema later.

### Current Implementation Status: **NOT IMPLEMENTED**

- grep for `deepspeed|fsdp` across `ptbr/` returns zero matches
- `_FIELD_SCHEMA` has no distributed training entries
- `template.yaml` has no distributed training section
- The underlying `gliner/training/trainer.py` does reference `self.deepspeed` for gradient accumulation compatibility, but ptbr has no way to configure it

### Test Coverage: **NO TESTS**

No tests exist for distributed training placeholders.

### Verdict: **STILL OUTSTANDING**

No distributed training placeholders have been added. Users needing DeepSpeed or FSDP must manually extend the schema.

---

## Focus Area 4: Hub Integration Fields

### Report Recommendation (Priority 3, Item 12)
Add `hub_strategy`, `hub_private_repo`, `hub_always_push` to the environment section.

### Current Implementation Status: **NOT IMPLEMENTED**

- grep for `hub_strategy|hub_private_repo|hub_always_push` across `ptbr/` returns zero matches
- `_FIELD_SCHEMA` environment section (lines 197-204) has only: `push_to_hub`, `hub_model_id`, `hf_token`, `report_to`, `wandb_project`, `wandb_entity`, `wandb_api_key`, `cuda_visible_devices`
- `template.yaml` environment section matches the schema exactly -- no additional hub fields

### Test Coverage: **PARTIAL (documents the gap)**

- `test_config_forwarding.py::TestCreateTrainingArgsSignatureGaps::test_push_to_hub_is_named_parameter` -- **FAILS** -- confirms `push_to_hub` is not a named parameter in `create_training_args`
- `test_config_forwarding.py::TestCreateTrainingArgsSignatureGaps::test_hub_model_id_is_named_parameter` -- **FAILS** -- confirms `hub_model_id` is not a named parameter
- `test_training_validation.py::TestCreateTrainingArgsSignature::test_push_to_hub_is_explicit` -- **XFAIL** -- documents this gap

### Verdict: **STILL OUTSTANDING**

The basic hub fields (`push_to_hub`, `hub_model_id`) exist in the ptbr schema and are forwarded to the environment, but the additional `hub_strategy`, `hub_private_repo`, and `hub_always_push` fields recommended by the report have not been added.

---

## Focus Area 5: ptbr init Command

### Report Recommendation (Priority 3, Item 13)
Add a `ptbr init` command that generates a starter YAML from the template, optionally with a model name pre-filled.

### Current Implementation Status: **NOT IMPLEMENTED**

- `__main__.py` (lines 1-136) registers three subcommands: `data`, `config`, `train`
- No `init` subcommand exists
- grep for `init.*command|ptbr init|def init` in `__main__.py` returns zero matches
- Users must manually find and copy `template.yaml`

### Test Coverage: **NO TESTS**

No tests exist for an init command.

### Verdict: **STILL OUTSTANDING**

No `ptbr init` command has been implemented. Users cannot generate a starter YAML from the CLI.

---

## Focus Area 6: W&B Tags Forwarding

### Report Recommendation (Priority 3, Item 15)
Forward `run.tags` to W&B via the `WANDB_TAGS` environment variable or `TrainingArguments`.

### Current Implementation Status: **NOT IMPLEMENTED**

- `run.tags` is defined in `_FIELD_SCHEMA` (line 99) and validated
- `_launch_training()` sets `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` (lines 1035-1043) but does **NOT** set `WANDB_TAGS`
- `run.tags` is never read or forwarded in `_launch_training()` -- it's validated and stored in the config dict but never consumed
- `run_name` IS forwarded (line 1140: `run_name=cfg["run"]["name"]`), but `run.tags` is not

### Test Coverage: **YES -- tests correctly document the gap**

- `test_validator_integration.py::TestParameterForwardingGaps::test_run_tags_not_forwarded` (line 371) -- **PASSES** -- confirms `run_tags` is NOT in the `model.train_model()` call
- The test asserts the bug exists (absence of forwarding), which is correct

### Verdict: **STILL OUTSTANDING**

Tags are validated but silently discarded. Neither `WANDB_TAGS` env var nor `run_tags` kwarg is set. The test at `test_validator_integration.py:371` correctly documents this gap.

---

## Issues FIXED Since the Report (Not in Focus Areas)

The report identified several other issues that HAVE been addressed:

### 1. YAML Schema Incompatibility (Report Priority 1, Item 1) -- **FIXED**

`config_cli.py` now accepts `model:` as an alias for `gliner_config:` (lines 418-432) and `lora:` as an alias for `lora_config:` (lines 470-486). The `test_config_cli_aliases.py` tests (3 tests, all passing) confirm this works correctly.

Evidence: `test_validator_integration.py` tests that expected template.yaml to fail config_cli now **fail** (inverted assertions), proving the fix.

### 2. run_name Forwarding (Report Priority 1, Item 2) -- **FIXED**

`_launch_training()` now passes `run_name=cfg["run"]["name"]` to `model.train_model()` (line 1140). W&B runs will be named.

Evidence: `test_validator_integration.py::TestParameterForwardingGaps::test_run_name_not_forwarded` now **fails** (expected it to be absent, but it's present).

### 3. Dataloader Parameters Forwarding (Report Priority 2, Item 3) -- **FIXED**

`_launch_training()` now forwards all three dataloader parameters (lines 1135-1137):
- `dataloader_pin_memory`
- `dataloader_persistent_workers`
- `dataloader_prefetch_factor`

Evidence: `test_validator_integration.py` tests for these three fields now **fail** (they asserted the parameters were NOT forwarded).

### 4. Lazy Import of training_cli (Report Priority 3, Item 14) -- **FIXED**

`__main__.py` uses `_attach_train_subcommand()` (lines 118-126) to lazy-load training_cli only when the `train` subcommand is invoked.

Evidence: `test_main_cli.py::test_importing_main_does_not_import_training_cli` -- **PASSES**.

---

## Issues Partially Fixed

### eval_strategy / eval_steps Forwarding

The report noted these were missing. `_launch_training()` now conditionally passes `eval_strategy: "steps"` and `eval_steps` when an eval dataset is present (lines 1127-1128). However, this uses dict unpacking (`**{...}`) which is fragile and not visible in AST-based test inspection.

### label_smoothing Forwarding (via ptbr training_cli)

`_launch_training()` now forwards `label_smoothing` (line 1117). However, upstream `train.py` still does not forward it, and `create_training_args` does not have it as a named parameter (relies on `**kwargs`).

---

## Issues Still Documented by Passing Tests

These tests pass and correctly document remaining gaps:

| Test | What It Confirms |
|------|-----------------|
| `test_validator_integration.py::test_size_sup_not_forwarded` | `size_sup` is dead config |
| `test_validator_integration.py::test_shuffle_types_not_forwarded` | `shuffle_types` is dead config |
| `test_validator_integration.py::test_random_drop_not_forwarded` | `random_drop` is dead config |
| `test_validator_integration.py::test_run_tags_not_forwarded` | W&B tags not forwarded |
| `test_validator_integration.py::test_run_description_not_forwarded` | Description unused |
| `test_validator_integration.py::TestRemoveUnusedColumns` (4 tests) | remove_unused_columns not set to False |
| `test_validator_integration.py::TestCreateTrainingArgsGaps` (3 tests) | label_smoothing, gradient_checkpointing, run_name not named params |
| `test_validator_integration.py::TestLoRASectionNaming` (3 tests) | LoRA field sets diverge between CLIs |
| `test_validator_integration.py::TestCLIArgumentInconsistency` (3 tests) | --file vs positional argument style |
| `test_training_validation.py` (22 xfail tests) | Various create_training_args gaps |

---

## Static Analysis Results

**All source files compile cleanly** (py_compile + ast.parse: 4/4 OK).

Flake8 found 63 issues (mostly cosmetic):
- 26 whitespace-in-blank-lines (W293)
- 10 lines over 120 chars (E501)
- 16 tab/space mixing (E101/W191) in training_cli.py -- worth fixing
- 5 unused imports: `sys`, `time` in training_cli.py; `textwrap`, `GLiNERConfigResult` in tests
- 3 unused local variables in test_config_cli.py
- No functional bugs detected via static analysis

---

## Inverted Tests (Need Fixing)

The following 15 tests in `test_validator_integration.py` now **fail** because they asserted bugs that have since been fixed. These tests should be updated to assert the correct (fixed) behavior:

1. `TestYAMLSchemaIncompatibility::test_template_yaml_fails_config_cli_validation` -- template now passes via alias
2. `TestYAMLSchemaIncompatibility::test_no_single_yaml_satisfies_both_clis` -- both CLIs now accept `model:` format
3. `TestParameterForwardingGaps::test_dataloader_pin_memory_not_forwarded` -- now forwarded
4. `TestParameterForwardingGaps::test_dataloader_persistent_workers_not_forwarded` -- now forwarded
5. `TestParameterForwardingGaps::test_dataloader_prefetch_factor_not_forwarded` -- now forwarded
6. `TestParameterForwardingGaps::test_run_name_not_forwarded` -- now forwarded
7. `TestTrainPyHardcodedValues::test_output_dir_hardcoded` -- test relies on `train.py` AST parsing
8. `TestTrainPyHardcodedValues::test_bf16_hardcoded_to_true` -- test relies on `train.py` AST parsing
9. `TestTrainPyHardcodedValues::test_eval_batch_size_reuses_train_batch_size` -- test relies on `train.py`
10. `TestTrainPyHardcodedValues::test_label_smoothing_not_forwarded_by_train_py` -- test relies on `train.py`
11. `TestConfigFieldsReachTraining::test_all_training_fields_in_config_yaml_are_forwarded` -- more fields now forwarded
12. `TestSchemaVsForwarding::test_validated_training_fields_not_all_forwarded` -- gaps reduced
13. `TestMainImportSideEffects::test_training_cli_imported_at_module_level` -- now lazy-loaded
14. `TestTemplateValidation::test_template_fails_config_cli` -- template now passes via alias
15. `TestEndToEndWorkflow::test_validate_then_train_is_impossible_with_single_yaml` -- now possible

---

## Summary Table: Six Focus Areas

| Focus Area | Report Priority | Implemented? | Tests Exist? | Tests Correct? |
|-----------|----------------|-------------|-------------|---------------|
| Long-doc chunking | P3 #9 | **No** | No | N/A |
| peft: wrapper | P3 #10 | **No** | Partial (documents divergence) | Yes |
| Distributed training placeholders | P3 #11 | **No** | No | N/A |
| Hub integration fields | P3 #12 | **No** | Partial (documents gap) | Yes |
| ptbr init command | P3 #13 | **No** | No | N/A |
| W&B tags forwarding | P3 #15 | **No** | Yes | **Yes** (correctly documents gap) |

**Key finding:** All six focus areas from the report remain unimplemented. The tests that exist for these areas correctly document the gaps (they don't falsely claim fixes). The major fixes that have been made address Priority 1 and Priority 2 items (YAML compatibility, parameter forwarding, lazy imports) but none of the six Priority 3 items in scope.
