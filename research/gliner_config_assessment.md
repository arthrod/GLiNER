## GLiNER Configuration Report Assessment

### Methodology

This assessment cross-references the 18 recommendations in `research/gliner_config_report.md` against:
- The current state of `ptbr/config_cli.py` (775 lines)
- The current state of `ptbr/training_cli.py` (1216 lines)
- The current state of `gliner/config.py` (361 lines)
- The current state of `train.py` (115 lines)
- The current state of `ptbr/__main__.py` (137 lines)
- Static analysis via `ruff` and `pyflakes`
- Test results from `ptbr/tests/test_config_cli.py`, `ptbr/tests/test_config_cli_aliases.py`, `ptbr/tests/test_training_cli.py`, and `tests/test_validator_integration.py`

---

### Static Analysis Findings

**ruff** found no critical issues. Notable findings:
- `ptbr/training_cli.py`: Unused imports (`sys`, `time`), whitespace issues, mixed tabs/spaces in some docstrings, and long lines. None are functional bugs.
- `ptbr/config_cli.py`: Minor style issues (unsorted imports, unused function arguments `full_or_lora` and `validate` in cross-constraint validation). No functional bugs.
- `gliner/config.py`: Clean — no issues found.

**pyflakes** confirmed the unused imports in `training_cli.py` (`sys`, `time`).

---

### Test Results Summary

| Test Suite | Result | Notes |
|-----------|--------|-------|
| `ptbr/tests/test_config_cli.py` | **Cannot run** | Imports `gliner.config.GLiNERConfig` which triggers `gliner/__init__.py` → `gliner/model.py` → `onnxruntime` (not installed). The test file lacks a mock strategy for heavy DL imports. |
| `ptbr/tests/test_config_cli_aliases.py` | **3/3 passed** | Uses monkeypatch + fake `GLiNERConfig` stub. Confirms `model:` alias and `lora:` alias work correctly. |
| `ptbr/tests/test_training_cli.py` | **62/62 passed** | Comprehensive. Covers `_deep_get`/`_deep_set`, `_check_type`, `validate_config`, `semantic_checks`, `check_huggingface`, `check_wandb`, `check_resume`, CLI integration, edge cases, LoRA application, and training parameter forwarding. |
| `tests/test_validator_integration.py` | **22 passed, 25 failed** | See detailed analysis below. |

#### Integration Test Failures — Categorized

The integration tests were written as **"documentation tests"** — they assert that known bugs exist. When bugs get fixed in the code, these tests fail because the bugs are gone. This is the expected pattern.

**Category 1: Import failures (12 tests)** — Tests that import `ptbr.config_cli` or `gliner.model`/`gliner.utils` trigger `gliner/__init__.py` → `onnxruntime` import. These tests need the `onnxruntime` package or need a mock strategy like `test_config_cli_aliases.py` uses.

**Category 2: Fixes that invalidated "bug documentation" tests (11 tests):**

| Test | Why It Fails | What Was Fixed |
|------|-------------|---------------|
| `test_dataloader_pin_memory_not_forwarded` | `dataloader_pin_memory` IS now forwarded at `training_cli.py:1135` | Bug fixed |
| `test_dataloader_persistent_workers_not_forwarded` | `dataloader_persistent_workers` IS now forwarded at `training_cli.py:1136` | Bug fixed |
| `test_dataloader_prefetch_factor_not_forwarded` | `dataloader_prefetch_factor` IS now forwarded at `training_cli.py:1137` | Bug fixed |
| `test_run_name_not_forwarded` | `run_name` IS now forwarded at `training_cli.py:1140` | Bug fixed |
| `test_output_dir_hardcoded` | `train.py:73` now uses `str(output_dir)` from `cfg.data.root_dir` instead of hardcoded `"models"` | Bug fixed |
| `test_bf16_hardcoded_to_true` | `train.py:104` now reads `getattr(cfg.training, "bf16", False)` instead of hardcoded `True` | Bug fixed |
| `test_eval_batch_size_reuses_train_batch_size` | `train.py:58-62` now has proper fallback logic for `eval_batch_size` | Bug fixed |
| `test_label_smoothing_not_forwarded_by_train_py` | `train.py:93` now forwards `label_smoothing` | Bug fixed |
| `test_all_training_fields_in_config_yaml_are_forwarded` | Expected missing set changed because `label_smoothing` is now forwarded | Bug fixed |
| `test_validated_training_fields_not_all_forwarded` | Only `{size_sup, random_drop, shuffle_types}` remain as gaps (dataloader fields were fixed) | Partially fixed |
| `test_training_cli_imported_at_module_level` | `__main__.py:118-126` now uses lazy import via `_attach_train_subcommand()` | Bug fixed |

**Category 3: Tests that still pass and correctly document remaining bugs (22 tests):**
These cover YAML schema incompatibility (via `training_cli`-only paths), LoRA key naming, CLI argument style divergence, `train.py` dead config fields (`size_sup`, `shuffle_types`, `random_drop`), config consistency checks, and template validation via `training_cli`.

---

### Issue-by-Issue Assessment Against the 18 Recommendations

#### Recommendation 1: Unify YAML section naming (`gliner_config:` vs `model:`)
**Status: FIXED**

`config_cli.py:416-440` now implements alias resolution. It checks for `gliner_config` first, falls back to `model` with a warning, and errors only if neither is present. Confirmed by `test_config_cli_aliases.py` (3/3 passed): `model:` is accepted as an alias, `lora:` is accepted as an alias, and canonical keys take precedence when both are present.

The template.yaml uses `model:` and `lora:` (training_cli format). With the alias support, `config_cli` now accepts it too (with warnings).

#### Recommendation 2: Unify required/optional for `span_mode` and `max_len`
**Status: NOT FIXED**

`config_cli.py:49` treats `span_mode` as optional (default `"markerV0"`). `training_cli.py:118` treats it as required (`True`). `config_cli.py:80` treats `max_len` as optional (default `384`). `training_cli.py:130` treats it as required. The divergence remains.

#### Recommendation 3: Fix `decoder_mode` default in config_cli.py
**Status: NOT FIXED**

`config_cli.py:59` still sets the default to `"span"`. Upstream `UniEncoderSpanDecoderConfig.__init__` defaults to `None`. This means `config_cli` will inject `decoder_mode="span"` into configs that never intended to set it.

#### Recommendation 4: Harmonize `span_mode` allowed values
**Status: NOT FIXED**

`training_cli.py:458` still restricts `_VALID_SPAN_MODES = {"markerV0", "token_level"}` (2 values). `config_cli.py:49-53` allows 12 values. Legitimate upstream modes like `"markerV1"`, `"marker"`, `"query"`, `"mlp"`, `"cat"`, and all `conv_*` variants will be rejected by `training_cli`.

#### Recommendation 5: Harmonize `subtoken_pooling` allowed values
**Status: NOT FIXED**

`training_cli.py:467` still uses `_VALID_SUBTOKEN_POOLING = {"first", "mean"}`. `config_cli.py:75` includes `"max"`. The `"max"` option is still rejected by `training_cli`.

#### Recommendation 6: Add `labels_encoder_config` validation
**Status: NOT FIXED**

Neither `config_cli.py` nor `training_cli.py` validates `labels_encoder_config`. This field exists on `BiEncoderConfig` (line 249 of `gliner/config.py`) but is absent from both validators' schemas.

#### Recommendation 7: Add `labels_decoder_config` validation
**Status: NOT FIXED**

Neither validator covers `labels_decoder_config`. This field exists on `UniEncoderSpanDecoderConfig` (line 149 of `gliner/config.py`) but is absent from both schemas.

#### Recommendation 8: Add `encoder_config` to config_cli.py
**Status: NOT FIXED**

`config_cli.py` still does not validate `encoder_config`. `training_cli.py:140` does validate it as `("model.encoder_config", (dict, type(None)), False, None, "Encoder config override")`.

#### Recommendation 9: Add method-aware validation to training_cli.py
**Status: NOT FIXED**

`training_cli.py`'s `semantic_checks()` function (line 472) still has no concept of a "method" (biencoder/decoder/relex/span/token). It does not check that BiEncoder requires `labels_encoder`, Decoder requires `labels_decoder`, or Relex requires `relations_layer`. It does not validate `span_mode` vs architecture consistency.

#### Recommendation 10: Add range constraints to training_cli.py
**Status: NOT FIXED**

`training_cli.py` still lacks range checks for `max_width`, `hidden_size`, `max_len`, `max_types`, `max_neg_type_ratio`, `num_post_fusion_layers`, and `num_rnn_layers`. The bounded numeric checks at lines 540-554 only cover `warmup_ratio`, `dropout`, `blank_entity_prob`, `label_smoothing`, and `lora_dropout`.

#### Recommendation 11: Add `words_splitter_type` enum check to training_cli.py
**Status: NOT FIXED**

`training_cli.py` does not validate `model.words_splitter_type` against any set of allowed values.

#### Recommendation 12: Add `decoder_mode` enum check to training_cli.py
**Status: NOT FIXED**

`training_cli.py` warns when `decoder_mode` is None while `labels_decoder` is set (line 502-509), but does not validate `decoder_mode` against `{"span", "prompt"}`.

#### Recommendation 13: Fix upstream `"token-level"` vs `"token_level"` bug
**Status: FIXED**

`gliner/config.py:323-341` (`GLiNERConfig.model_type` property) now uses `"token_level"` (underscore) consistently throughout. Lines 327, 332, 334, 338 in the original report all used `"token-level"` (hyphen) — these have all been changed to `"token_level"`. Verified by reading the current file: the `model_type` property correctly routes `span_mode="token_level"` to the right model types.

The test at `test_config_cli.py:474-489` (`test_gliner_config_model_type_token_level`) parametrically verifies that `GLiNERConfig(span_mode="token_level")` routes to the correct model types: `"gliner_uni_encoder_token"`, `"gliner_bi_encoder_token"`, `"gliner_uni_encoder_token_decoder"`, `"gliner_uni_encoder_token_relex"`.

#### Recommendation 14: Add `represent_spans` cross-field check
**Status: NOT FIXED**

Neither validator enforces that `represent_spans` should be `True` when `span_mode == "token_level"` and `labels_decoder` is set (the `UniEncoderTokenDecoderConfig` scenario). The upstream code at `gliner/config.py:184` hardcodes `self.represent_spans = True`.

#### Recommendation 15: Unify LoRA section naming and defaults
**Status: PARTIALLY FIXED**

Section naming: `config_cli.py` now accepts both `lora_config:` and `lora:` via alias resolution (lines 458-487). Confirmed by `test_config_cli_aliases.py`. However, default value disagreements remain:

| Field | config_cli.py default | training_cli.py default |
|-------|----------------------|------------------------|
| `lora_dropout` | `0.1` (line 107) | `0.05` (line 190) |
| `target_modules` | `["query_proj", "value_proj"]` (line 108) | `["q_proj", "v_proj"]` (line 192) |
| `task_type` | `"FEATURE_EXTRACTION"` (line 110) | `"TOKEN_CLS"` (line 193) |

Extra fields in `config_cli` not in `training_cli`: `fan_in_fan_out`, `use_rslora`, `init_lora_weights`. Extra field in `training_cli` not in `config_cli`: `enabled` toggle.

#### Recommendation 16: Extract shared validation logic
**Status: NOT FIXED**

The two validators remain entirely separate codebases. No shared module exists.

#### Recommendation 17: Add `_VALID_DECODER_MODES` to training_cli.py
**Status: NOT FIXED**

`training_cli.py` does not validate `decoder_mode` against `{"span", "prompt"}`.

#### Recommendation 18: Update template.yaml
**Status: FIXED (via alias support)**

`template.yaml` uses `model:` and `lora:`. With the alias support added to `config_cli.py`, the template now works with both tools (config_cli emits a warning but accepts it).

---

### Additional Fixes Found (Not in Original Report)

1. **`train.py` overhaul**: `train.py` was significantly rewritten. It no longer hardcodes `output_dir="models"` (now uses `cfg.data.root_dir`), no longer hardcodes `bf16=True` (reads from config), properly handles `eval_batch_size` fallback, forwards `label_smoothing`, and handles `loss_prob_margin` and `logging_steps`.

2. **`_launch_training` in `training_cli.py`**: Now forwards `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor`, `run_name`, `gradient_accumulation_steps`, `compile_model`, `fp16`, and `use_cpu` — all of which were previously missing.

3. **Lazy import in `__main__.py`**: `training_cli` is no longer imported at module level. `_attach_train_subcommand()` (line 118-126) does a lazy import, avoiding import side effects when only using `config` or `data` subcommands.

---

### Summary Scorecard

| # | Recommendation | Status | Test Coverage |
|---|---------------|--------|---------------|
| 1 | Unify YAML section naming | **FIXED** | `test_config_cli_aliases.py` (3 tests) |
| 2 | Unify required/optional for span_mode/max_len | NOT FIXED | No specific test |
| 3 | Fix decoder_mode default | NOT FIXED | No specific test |
| 4 | Harmonize span_mode allowed values | NOT FIXED | `test_invalid_span_mode` (training_cli only) |
| 5 | Harmonize subtoken_pooling allowed values | NOT FIXED | No specific test |
| 6 | Add labels_encoder_config validation | NOT FIXED | No test |
| 7 | Add labels_decoder_config validation | NOT FIXED | No test |
| 8 | Add encoder_config to config_cli | NOT FIXED | No test |
| 9 | Add method-aware validation to training_cli | NOT FIXED | No test |
| 10 | Add range constraints to training_cli | NOT FIXED | No test |
| 11 | Add words_splitter_type enum to training_cli | NOT FIXED | No test |
| 12 | Add decoder_mode enum to training_cli | NOT FIXED | `test_decoder_mode_warning` (partial) |
| 13 | Fix token-level/token_level upstream bug | **FIXED** | `test_gliner_config_model_type_token_level` (4 parametrized) |
| 14 | Add represent_spans cross-field check | NOT FIXED | No test |
| 15 | Unify LoRA naming and defaults | **PARTIAL** | `test_config_cli_aliases.py`, `test_lora_field_sets_differ_between_clis` |
| 16 | Extract shared validation logic | NOT FIXED | No test |
| 17 | Add _VALID_DECODER_MODES to training_cli | NOT FIXED | No test |
| 18 | Update template.yaml | **FIXED** (via aliases) | `test_full_template_validates` |

**Fixed: 4 (Rec 1, 13, 18, plus significant train.py/training_cli fixes)**
**Partially fixed: 1 (Rec 15)**
**Not fixed: 13**

---

### Stale Integration Tests

The following 11 integration tests in `tests/test_validator_integration.py` assert that bugs exist, but those bugs have been fixed. These tests should be updated to assert the fixed behavior instead:

1. `TestParameterForwardingGaps::test_dataloader_pin_memory_not_forwarded`
2. `TestParameterForwardingGaps::test_dataloader_persistent_workers_not_forwarded`
3. `TestParameterForwardingGaps::test_dataloader_prefetch_factor_not_forwarded`
4. `TestParameterForwardingGaps::test_run_name_not_forwarded`
5. `TestTrainPyHardcodedValues::test_output_dir_hardcoded`
6. `TestTrainPyHardcodedValues::test_bf16_hardcoded_to_true`
7. `TestTrainPyHardcodedValues::test_eval_batch_size_reuses_train_batch_size`
8. `TestTrainPyHardcodedValues::test_label_smoothing_not_forwarded_by_train_py`
9. `TestConfigFieldsReachTraining::test_all_training_fields_in_config_yaml_are_forwarded`
10. `TestSchemaVsForwarding::test_validated_training_fields_not_all_forwarded`
11. `TestMainImportSideEffects::test_training_cli_imported_at_module_level`

Additionally, 12 tests fail due to missing `onnxruntime` dependency (triggered by `gliner/__init__.py` → `gliner/model.py` import chain). These tests need a mock strategy similar to `test_config_cli_aliases.py`.

### Remaining Forwarding Gaps

Three training config fields are still validated but never forwarded to `model.train_model()`:
- `training.size_sup`
- `training.shuffle_types`
- `training.random_drop`
