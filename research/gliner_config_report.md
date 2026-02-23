## GLiNER Model Configuration Report

### Executive Summary

The toolkit has two independent validation paths for model fields -- `config_cli.py` (expects `gliner_config:` YAML section) and `training_cli.py` (expects `model:` YAML section) -- that are structurally incompatible, use different validation engines, and diverge on field coverage, required-vs-optional designations, literal allowed-value sets, and cross-field semantic checks. Upstream `gliner/config.py` exposes fields across five config subclasses that are only partially covered: `labels_encoder_config`, `labels_decoder_config`, and several decoder/relex constructor parameters are missing from one or both validators. A critical upstream bug exists where `GLiNERConfig.model_type` checks for `"token-level"` (hyphen) while the typed config subclasses and our validators all use `"token_level"` (underscore), creating a silent architecture-routing mismatch.

---

### config_cli.py vs training_cli.py Consistency

These two files validate the **same conceptual model fields** but disagree on YAML structure, field schemas, required/optional status, allowed values, and cross-field logic. A config YAML valid for one will **fail** for the other.

#### 1. YAML Section Name Mismatch

| Aspect | config_cli.py | training_cli.py |
|--------|--------------|-----------------|
| Section key | `gliner_config:` | `model:` |
| Error prefix | `gliner_config.field_name` | `model.field_name` |

A user writing a YAML with `model:` (as the template.yaml does) will get an immediate `"Missing 'gliner_config' section"` error from `config_cli.py`. Conversely, a YAML with `gliner_config:` will cause `training_cli.py` to emit DEFAULT warnings for every model field and then fail on the required ones (`model.model_name`, `model.span_mode`, `model.max_len`).

#### 2. Required Field Disagreements

| Field | config_cli.py | training_cli.py |
|-------|--------------|-----------------|
| `model_name` | required | required |
| `span_mode` | **optional** (default `"markerV0"`) | **required** (no default) |
| `max_len` | **optional** (default `384`) | **required** (no default) |

`config_cli.py` treats `span_mode` and `max_len` as optional with defaults. `training_cli.py` marks both as `True` (required), meaning they must be explicitly provided. This means a minimal config that works with `config_cli.py` will fail validation in `training_cli.py`.

#### 3. Allowed Literal Sets for `span_mode`

| Validator | Allowed values |
|-----------|---------------|
| `config_cli.py` line 49-53 | `{"markerV0", "markerV1", "marker", "query", "mlp", "cat", "conv_conv", "conv_max", "conv_mean", "conv_sum", "conv_share", "token_level"}` (12 values) |
| `training_cli.py` line 455 | `{"markerV0", "token_level"}` (2 values) |

`training_cli.py` will **reject** legitimate upstream span modes like `"markerV1"`, `"marker"`, `"query"`, `"mlp"`, `"cat"`, and all `conv_*` variants. This is a significant over-restriction.

#### 4. Allowed Literal Sets for `subtoken_pooling`

| Validator | Allowed values |
|-----------|---------------|
| `config_cli.py` line 75 | `{"first", "mean", "max"}` |
| `training_cli.py` line 464 | `{"first", "mean"}` |

`config_cli.py` allows `"max"` but `training_cli.py` does not.

#### 5. Missing Fields in config_cli.py

`config_cli.py` does **not** validate:
- `encoder_config` (present in `training_cli.py` at line 137)

#### 6. Missing Fields in training_cli.py

`training_cli.py` does **not** have explicit range constraints on model fields. It only does type checks and enum checks. For example:
- No range check on `max_width` (config_cli checks `[1, 128]`)
- No range check on `hidden_size` (config_cli checks `[64, 4096]`)
- No range check on `max_len` (config_cli checks `[32, 8192]`)
- No range check on `max_types` (config_cli checks `[1, 1000]`)
- No range check on `max_neg_type_ratio` (config_cli checks `[0, 100]`)
- No range check on `num_post_fusion_layers` (config_cli checks `[1, 12]`)
- No range check on `num_rnn_layers` (config_cli checks `[0, 4]`)
- No range check on `rel_token_index` (config_cli checks `[-1, 100000]`)
- No range check on `class_token_index` (config_cli checks `[-1, 100000]`)
- No range check on `vocab_size` (config_cli checks `[-1, 1000000]`)

The `training_cli.py` does have bounded numeric checks for `model.dropout` and `model.blank_entity_prob` in `semantic_checks()` (line 537-548), but these are a small subset.

#### 7. Cross-Field Validation Differences

| Check | config_cli.py | training_cli.py |
|-------|--------------|-----------------|
| BiEncoder requires `labels_encoder` | Yes (line 303-307) | No |
| Decoder requires `labels_decoder` | Yes (line 309-313) | No |
| Relex requires `relations_layer` | Yes (line 315-319) | No |
| `span_mode` vs method consistency | Yes (line 323-335) | No (no method concept) |
| Decoder fields without `labels_decoder` | Yes (line 338-351) | Partial (warns if `decoder_mode` missing when `labels_decoder` set, line 500-506) |
| Relex fields without `relations_layer` | Yes (line 353-369) | No |
| `bf16`/`fp16` mutual exclusion | No | Yes (line 553-558) |
| WandB project required when `report_to=wandb` | No | Yes (line 508-515) |
| Hub model id required when `push_to_hub` | No | Yes (line 517-523) |

`config_cli.py` has method-aware cross-field checks (it takes a `method` parameter). `training_cli.py` has no concept of a "method" at all and cannot perform these checks. However, `training_cli.py` has training-specific cross-field checks that `config_cli.py` lacks.

#### 8. Validation Engine Architecture

- `config_cli.py`: Rule-based tuples with `(key, type, required, default, constraint)`. Validates a flat dict. Produces `ValidationReport` with structured issues.
- `training_cli.py`: Schema-based tuples with `(dotted_key, type, required, default, description)`. Validates a nested dict via `_deep_get`/`_deep_set`. Produces `ValidationResult` with string-based issues.

These are entirely separate codebases with no shared logic.

#### 9. Default Value Disagreements

| Field | config_cli.py default | training_cli.py default |
|-------|----------------------|------------------------|
| `decoder_mode` | `"span"` | `None` |
| `post_fusion_schema` | `""` | `""` |

For `decoder_mode`: `config_cli.py` (line 59) defaults to `"span"`, while `training_cli.py` (line 104) defaults to `None`. This means if a user omits `decoder_mode`, `config_cli.py` will inject `"span"` into the config, but `training_cli.py` will leave it as `None`. This changes downstream behavior.

#### 10. LoRA Section Differences

| Aspect | config_cli.py | training_cli.py |
|--------|--------------|-----------------|
| Section key | `lora_config:` | `lora:` |
| Activation | `--full-or-lora lora` CLI flag | `lora.enabled: true` field |
| Default `lora_dropout` | `0.1` | `0.05` |
| Default `target_modules` | `["query_proj", "value_proj"]` | `["q_proj", "v_proj"]` |
| Default `task_type` | `"FEATURE_EXTRACTION"` | `"TOKEN_CLS"` |
| Extra fields | `fan_in_fan_out`, `use_rslora`, `init_lora_weights` | None of these |
| Missing fields | No `enabled` toggle | Has `enabled` toggle |

---

### Coverage vs GLiNER Config Classes

#### BaseGLiNERConfig (28 explicit parameters)

| Parameter | config_cli.py | training_cli.py | Notes |
|-----------|:---:|:---:|-------|
| `model_name` | Yes | Yes | |
| `name` | Yes | Yes | |
| `max_width` | Yes | Yes | |
| `hidden_size` | Yes | Yes | |
| `dropout` | Yes | Yes | |
| `fine_tune` | Yes | Yes | |
| `subtoken_pooling` | Yes | Yes | |
| `span_mode` | Yes | Yes | |
| `post_fusion_schema` | Yes | Yes | |
| `num_post_fusion_layers` | Yes | Yes | |
| `vocab_size` | Yes | Yes | |
| `max_neg_type_ratio` | Yes | Yes | |
| `max_types` | Yes | Yes | |
| `max_len` | Yes | Yes | |
| `words_splitter_type` | Yes | Yes | |
| `num_rnn_layers` | Yes | Yes | |
| `fuse_layers` | Yes | Yes | |
| `embed_ent_token` | Yes | Yes | |
| `class_token_index` | Yes | Yes | |
| `encoder_config` | **NO** | Yes | config_cli.py completely ignores this |
| `ent_token` | Yes | Yes | |
| `sep_token` | Yes | Yes | |
| `_attn_implementation` | Yes | Yes | |
| `token_loss_coef` | Yes | Yes | |
| `span_loss_coef` | Yes | Yes | |
| `represent_spans` | Yes | Yes | |
| `neg_spans_ratio` | Yes | Yes | |

Coverage: config_cli.py covers 27/28; training_cli.py covers 28/28.

#### BiEncoderConfig (adds 2 parameters)

| Parameter | config_cli.py | training_cli.py | Notes |
|-----------|:---:|:---:|-------|
| `labels_encoder` | Yes | Yes | |
| `labels_encoder_config` | **NO** | **NO** | Missing from both validators |

Coverage: Both cover 1/2 BiEncoder-specific parameters.

#### UniEncoderSpanDecoderConfig (adds 6 parameters)

| Parameter | config_cli.py | training_cli.py | Notes |
|-----------|:---:|:---:|-------|
| `labels_decoder` | Yes | Yes | |
| `decoder_mode` | Yes | Yes | |
| `full_decoder_context` | Yes | Yes | |
| `blank_entity_prob` | Yes | Yes | |
| `labels_decoder_config` | **NO** | **NO** | Missing from both validators |
| `decoder_loss_coef` | Yes | Yes | |

Coverage: Both cover 5/6 Decoder-specific parameters.

#### UniEncoderRelexConfig (adds 7 parameters)

| Parameter | config_cli.py | training_cli.py | Notes |
|-----------|:---:|:---:|-------|
| `relations_layer` | Yes | Yes | |
| `triples_layer` | Yes | Yes | |
| `embed_rel_token` | Yes | Yes | |
| `rel_token_index` | Yes | Yes | |
| `rel_token` | Yes | Yes | |
| `adjacency_loss_coef` | Yes | Yes | |
| `relation_loss_coef` | Yes | Yes | |

Coverage: Both cover 7/7 Relex-specific parameters.

#### GLiNERConfig (legacy auto-detect class, adds 3 explicit parameters)

| Parameter | config_cli.py | training_cli.py | Notes |
|-----------|:---:|:---:|-------|
| `labels_encoder` | Yes | Yes | |
| `labels_decoder` | Yes | Yes | |
| `relations_layer` | Yes | Yes | |

Coverage: Both cover 3/3 GLiNERConfig-specific parameters.

**Key gap**: `labels_encoder_config` and `labels_decoder_config` are accepted by the upstream config classes but validated by neither of our tools.

---

### Validation Rules Accuracy

#### Defaults vs Upstream

All defaults in both validators were compared against `BaseGLiNERConfig.__init__` defaults (lines 13-42 of `gliner/config.py`):

| Field | Upstream default | config_cli.py | training_cli.py | Match? |
|-------|-----------------|--------------|-----------------|--------|
| `model_name` | `"microsoft/deberta-v3-small"` | required (no default) | required (no default) | OK (required is stricter) |
| `name` | `"gliner"` | `"gliner"` | `"gliner"` | Yes |
| `max_width` | `12` | `12` | `12` | Yes |
| `hidden_size` | `512` | `512` | `512` | Yes |
| `dropout` | `0.4` | `0.4` | `0.4` | Yes |
| `fine_tune` | `True` | `True` | `True` | Yes |
| `subtoken_pooling` | `"first"` | `"first"` | `"first"` | Yes |
| `span_mode` | `"markerV0"` | `"markerV0"` | required | config_cli matches; training_cli deviates |
| `post_fusion_schema` | `""` | `""` | `""` | Yes |
| `num_post_fusion_layers` | `1` | `1` | `1` | Yes |
| `vocab_size` | `-1` | `-1` | `-1` | Yes |
| `max_neg_type_ratio` | `1` | `1` | `1` | Yes |
| `max_types` | `25` | `25` | `25` | Yes |
| `max_len` | `384` | `384` | required | config_cli matches; training_cli deviates |
| `words_splitter_type` | `"whitespace"` | `"whitespace"` | `"whitespace"` | Yes |
| `num_rnn_layers` | `1` | `1` | `1` | Yes |
| `fuse_layers` | `False` | `False` | `False` | Yes |
| `embed_ent_token` | `True` | `True` | `True` | Yes |
| `class_token_index` | `-1` | `-1` | `-1` | Yes |
| `encoder_config` | `None` | not validated | `None` | N/A / Yes |
| `ent_token` | `"<<ENT>>"` | `"<<ENT>>"` | `"<<ENT>>"` | Yes |
| `sep_token` | `"<<SEP>>"` | `"<<SEP>>"` | `"<<SEP>>"` | Yes |
| `_attn_implementation` | `None` | `None` | `None` | Yes |
| `token_loss_coef` | `1.0` | `1.0` | `1.0` | Yes |
| `span_loss_coef` | `1.0` | `1.0` | `1.0` | Yes |
| `represent_spans` | `False` | `False` | `False` | Yes |
| `neg_spans_ratio` | `1.0` | `1.0` | `1.0` | Yes |
| `blank_entity_prob` | `0.1` | `0.1` | `0.1` | Yes |
| `decoder_loss_coef` | `0.5` | `0.5` | `0.5` | Yes |
| `decoder_mode` | `None` | **`"span"`** | `None` | **config_cli.py WRONG** -- upstream default is `None` |
| `full_decoder_context` | `True` | `True` | `True` | Yes |
| `adjacency_loss_coef` | `1.0` | `1.0` | `1.0` | Yes |
| `relation_loss_coef` | `1.0` | `1.0` | `1.0` | Yes |

**Bug found**: `config_cli.py` line 59 sets `decoder_mode` default to `"span"`, but upstream `UniEncoderSpanDecoderConfig.__init__` (line 146) defaults to `None`. This means `config_cli.py` will inject `decoder_mode="span"` into configs that never intended to set it, which could silently alter model behavior if `labels_decoder` is also set.

#### Range Constraints Analysis

`config_cli.py` defines range constraints. Are they reasonable?

| Field | config_cli range | Assessment |
|-------|-----------------|------------|
| `max_width` | `[1, 128]` | Reasonable. Upstream default is 12. |
| `hidden_size` | `[64, 4096]` | Reasonable for projection sizes. |
| `dropout` | `[0.0, 0.9]` | OK. training_cli.py uses `[0.0, 1.0]` -- the stricter 0.9 upper bound is arguably better. |
| `max_len` | `[32, 8192]` | Good, accommodates modern long-context models. |
| `max_types` | `[1, 1000]` | Reasonable. |
| `max_neg_type_ratio` | `[0, 100]` | Reasonable. |
| `num_post_fusion_layers` | `[1, 12]` | Reasonable. |
| `num_rnn_layers` | `[0, 4]` | Reasonable. |
| `blank_entity_prob` | `[0.0, 1.0]` | Correct (probability). |
| `decoder_loss_coef` | `[0.0, 10.0]` | Reasonable. |
| `adjacency_loss_coef` | `[0.0, 10.0]` | Reasonable. |
| `relation_loss_coef` | `[0.0, 10.0]` | Reasonable. |
| `token_loss_coef` | `[0.0, 10.0]` | Reasonable. |
| `span_loss_coef` | `[0.0, 10.0]` | Reasonable. |
| `neg_spans_ratio` | `[0.0, 10.0]` | Reasonable. |

---

### Cross-Field Validation Completeness

#### What config_cli.py checks (lines 289-369)

1. **BiEncoder requires `labels_encoder`**: Yes (line 303-307).
2. **Decoder requires `labels_decoder`**: Yes (line 309-313).
3. **Relex requires `relations_layer`**: Yes (line 315-319).
4. **`span_mode` vs method consistency**: Yes (lines 323-335). Forces `token_level` for token method; warns for other methods using `token_level`.
5. **Decoder fields when method is not decoder**: Yes (lines 338-351). Warns about orphaned decoder fields.
6. **Relex fields when method is not relex**: Yes (lines 353-369). Warns about orphaned relex fields.

#### What config_cli.py MISSES

1. **`represent_spans` must be `True` for `UniEncoderTokenDecoderConfig`**: Upstream `UniEncoderTokenDecoderConfig.__init__` (line 187) hardcodes `self.represent_spans = True`. Our validator does not enforce this constraint.
2. **`span_mode` incompatibility with span configs**: `UniEncoderSpanConfig` (line 125-126), `UniEncoderSpanRelexConfig` (line 236-237), and `BiEncoderSpanConfig` (line 276-277) all raise `ValueError` if `span_mode == "token_level"`. Our cross-field validation warns but does not error.
3. **`triples_layer` depends on `relations_layer`**: Partially checked (lines 361-369 warn if `triples_layer` set without `relations_layer`), but only when method is not relex.
4. **No check that `labels_decoder` is not None when decoder-specific fields are used**: The check at lines 338-351 only warns; it should arguably error.
5. **No check for `labels_encoder_config` consistency with `labels_encoder`**: If `labels_encoder` is set, `labels_encoder_config` might be needed.

#### What training_cli.py checks (lines 469-560)

1. **Enum validation for model fields**: `span_mode`, `subtoken_pooling`, `_attn_implementation` (lines 485-497).
2. **Decoder fields require `labels_decoder`**: Partial (lines 500-506). Only checks that `decoder_mode` should be set when `labels_decoder` is used.
3. **WandB/Hub integration checks**: Yes (lines 508-523).
4. **Positive numeric checks**: Yes (lines 526-535).
5. **Bounded numeric checks**: Yes (lines 537-548).
6. **bf16/fp16 mutual exclusion**: Yes (lines 553-558).

#### What training_cli.py MISSES

1. **No method-aware validation at all**: No concept of biencoder/decoder/relex/span/token methods.
2. **No check that BiEncoder requires `labels_encoder`**.
3. **No check that Relex requires `relations_layer`**.
4. **No `span_mode` vs architecture consistency checks**.
5. **No orphaned decoder/relex field warnings**.
6. **No check for `decoder_mode` allowed values** (it does not have `_VALID_DECODER_MODES`).
7. **No check for `words_splitter_type` allowed values** (config_cli has 7 allowed splitters; training_cli has none).

---

### Missing Fields

#### Fields in upstream `gliner/config.py` NOT validated by either tool

| Field | Source class | Why it matters |
|-------|-------------|---------------|
| `labels_encoder_config` | `BiEncoderConfig` (line 252) | Snapshot of the labels encoder's PretrainedConfig. Needed for reproducibility and Hub serialization. |
| `labels_decoder_config` | `UniEncoderSpanDecoderConfig` (line 149) | Snapshot of the decoder's PretrainedConfig. Needed for reproducibility and Hub serialization. |

#### Fields recommended by the research report NOT validated by either tool

| Field | Report section | Status |
|-------|---------------|--------|
| `labels_encoder_config` | "Optional snapshots" | Missing from both |
| `labels_decoder_config` | "Optional snapshots" | Missing from both |
| `data_seed` | Research report YAML | Not in either validator (training_cli has `run.seed` only) |

#### Fields in config_cli.py NOT in training_cli.py

None -- `training_cli.py` covers all fields that `config_cli.py` covers, plus `encoder_config`.

#### Fields in training_cli.py NOT in config_cli.py

| Field | Notes |
|-------|-------|
| `encoder_config` | training_cli line 137 validates this; config_cli does not |

---

### Bug: span_mode "token-level" vs "token_level"

This is a critical inconsistency in the upstream codebase itself, and our validators inherit it partially.

#### The Upstream Bug

In `gliner/config.py`, there are **two different string conventions** for token-level span mode:

1. **Typed config subclasses use underscore `"token_level"`**:
   - `UniEncoderSpanConfig.__init__` (line 125): `if self.span_mode == "token_level": raise ValueError`
   - `UniEncoderTokenConfig.__init__` (line 136): `self.span_mode = "token_level"`
   - `UniEncoderTokenDecoderConfig.__init__` (line 185): `self.span_mode = "token_level"`
   - `UniEncoderSpanRelexConfig.__init__` (line 236): `if self.span_mode == "token_level": raise ValueError`
   - `UniEncoderTokenRelexConfig.__init__` (line 246): `self.span_mode = "token_level"`
   - `BiEncoderSpanConfig.__init__` (line 276): `if self.span_mode == "token_level": raise ValueError`
   - `BiEncoderTokenConfig.__init__` (line 286): `self.span_mode = "token_level"`

2. **The legacy `GLiNERConfig.model_type` property uses hyphen `"token-level"`**:
   - Line 327: `if self.span_mode == "token-level":`
   - Line 332: `if self.span_mode != "token-level"`
   - Line 334: `if self.span_mode == "token-level":`
   - Line 338: `elif self.span_mode == "token-level":`

#### The Consequence

If a user instantiates `GLiNERConfig(span_mode="token_level")`, the `model_type` property (lines 323-341) will **never match the `"token-level"` checks** and will fall through to return `"gliner_uni_encoder_span"` instead of `"gliner_uni_encoder_token"`. This means:

- `GLiNERConfig(span_mode="token_level")` -> model_type = `"gliner_uni_encoder_span"` (**WRONG**: should be `"gliner_uni_encoder_token"`)
- `GLiNERConfig(span_mode="token-level")` -> model_type = `"gliner_uni_encoder_token"` (**CORRECT** per the property, but inconsistent with all typed subclasses)

This is a **silent architecture misrouting bug** in the upstream code. The typed subclasses enforce `"token_level"` (underscore), but the legacy auto-detect class requires `"token-level"` (hyphen) to route correctly.

#### Our Validators

Both `config_cli.py` and `training_cli.py` standardize on `"token_level"` (underscore), which is consistent with the typed config subclasses and the research report recommendation. However:

- `config_cli.py` builds a `GLiNERConfig` object (line 376: `GLiNERConfig(**gliner_data)`), which uses the **legacy auto-detect** `model_type` property. If the user sets `span_mode="token_level"`, the resulting `GLiNERConfig` object will have the wrong `model_type`.
- `training_cli.py` passes the model config dict to `GLiNER.from_config(model_cfg)` (line 1052), which internally may use `GLiNERConfig` or a typed subclass depending on the `from_config` implementation.

**Neither validator warns the user about this upstream inconsistency.**

---

### Concrete Recommendations

1. **Unify YAML section naming** (`config_cli.py` line 417, `training_cli.py` line 99-139): Choose one section name. The template.yaml uses `model:`, so `config_cli.py` should be updated to look for `model:` instead of `gliner_config:`. Alternatively, support both with a deprecation warning.

2. **Unify required/optional status for `span_mode` and `max_len`** (`config_cli.py` lines 49, 80 vs `training_cli.py` lines 115, 127): Either both should be required or both should have defaults. Recommendation: make them optional with the upstream defaults (`"markerV0"` and `384`), as the upstream `BaseGLiNERConfig` has defaults for both.

3. **Fix `decoder_mode` default in `config_cli.py`** (line 59): Change from `"span"` to `None` to match upstream `UniEncoderSpanDecoderConfig.__init__` (line 146 of `gliner/config.py`).

4. **Harmonize `span_mode` allowed values** (`training_cli.py` line 455): Expand `_VALID_SPAN_MODES` to include all values from `config_cli.py` (line 49-53): `{"markerV0", "markerV1", "marker", "query", "mlp", "cat", "conv_conv", "conv_max", "conv_mean", "conv_sum", "conv_share", "token_level"}`.

5. **Harmonize `subtoken_pooling` allowed values** (`training_cli.py` line 464): Add `"max"` to `_VALID_SUBTOKEN_POOLING` to match `config_cli.py` (line 75).

6. **Add `labels_encoder_config` validation to both files**: Add a rule in `config_cli.py` after line 56 and a schema entry in `training_cli.py` after line 102: `("model.labels_encoder_config", (dict, type(None)), False, None, "Labels encoder config snapshot")`.

7. **Add `labels_decoder_config` validation to both files**: Add a rule in `config_cli.py` after line 61 and a schema entry in `training_cli.py` after line 103: `("model.labels_decoder_config", (dict, type(None)), False, None, "Labels decoder config snapshot")`.

8. **Add `encoder_config` to `config_cli.py`** (missing entirely): Add a rule like `("encoder_config", dict, False, None, None)` to `_GLINER_RULES` around line 100, accepting `dict` or `None`.

9. **Add method-aware validation to `training_cli.py`**: The `semantic_checks()` function (line 469) should accept a method parameter or infer method from the config (e.g., presence of `labels_encoder`, `labels_decoder`, `relations_layer`) and perform the same cross-field checks as `config_cli.py` lines 289-369.

10. **Add range constraints to `training_cli.py`**: Port the range checks from `config_cli.py` into `training_cli.py`'s `semantic_checks()` function for at least: `max_width`, `hidden_size`, `max_len`, `max_types`, `max_neg_type_ratio`, `num_post_fusion_layers`, `num_rnn_layers`.

11. **Add `words_splitter_type` enum check to `training_cli.py`**: Add to `semantic_checks()` a `_check_enum` call for `model.words_splitter_type` with the set `{"whitespace", "spacy", "stanza", "mecab", "jieba", "janome", "camel"}`.

12. **Add `decoder_mode` enum check to `training_cli.py`**: When `model.labels_decoder` is not None, validate `model.decoder_mode` against `{"span", "prompt"}`.

13. **Fix the upstream `"token-level"` vs `"token_level"` bug**: In `gliner/config.py` lines 327, 332, 334, 338: replace all occurrences of `"token-level"` with `"token_level"`. This is a bug fix in the upstream code. Until this is fixed, `config_cli.py` should NOT use `GLiNERConfig(...)` directly (line 376) for token-level configs; it should instead use the typed subclass (e.g., `UniEncoderTokenConfig`).

14. **Add a `represent_spans` cross-field check**: When `span_mode == "token_level"` and `labels_decoder` is set (i.e., `UniEncoderTokenDecoderConfig` scenario), enforce or warn that `represent_spans` should be `True`, per upstream line 187.

15. **Unify LoRA section naming and defaults** (`config_cli.py` line 103-116 vs `training_cli.py` line 184-191):
    - Section: `lora_config:` vs `lora:` -- pick one.
    - `lora_dropout`: `0.1` vs `0.05` -- align to one value (the research report suggests `0.05`).
    - `target_modules`: `["query_proj", "value_proj"]` vs `["q_proj", "v_proj"]` -- these are model-dependent; document this clearly and consider making it required rather than defaulted.
    - `task_type`: `"FEATURE_EXTRACTION"` vs `"TOKEN_CLS"` -- the training_cli uses `TOKEN_CLS` which is more appropriate for token-level NER; update config_cli.

16. **Extract shared validation logic**: Both validators duplicate substantial model-field logic. Extract a shared module (e.g., `ptbr/model_schema.py`) that defines the canonical field schema, type constraints, range constraints, allowed literals, and cross-field rules. Both `config_cli.py` and `training_cli.py` should import from this shared module.

17. **Add `_VALID_DECODER_MODES` to `training_cli.py`** semantic checks: After line 497, add validation that when `model.decoder_mode` is not None, it must be one of `{"span", "prompt"}`.

18. **Update template.yaml**: The template at `/Users/arthrod/temp/T/GLiNER_testing/GLiNER/ptbr/template.yaml` uses `model:` (compatible with `training_cli.py`), but is not compatible with `config_cli.py` which expects `gliner_config:`. After fixing recommendation 1, ensure the template works with both tools.
