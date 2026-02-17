# Research Topic: GLiNER Model Configuration

## Source: deep-research-report.md (sections on GLiNER config, architecture routing)

### Key Findings from Report

**GLiNER config fields (belong under `model:`):**
- Backbone: `model_name`, `fine_tune`
- Architecture routing: `span_mode`, `labels_encoder`, `labels_decoder`, `relations_layer`, `triples_layer`
- Prompt/special tokens: `ent_token`, `sep_token`, `rel_token`, embedding toggles, token indices
- Sequence/spans: `max_len`, `max_width`, `represent_spans`, `neg_spans_ratio`
- Open label-set: `max_types`, `max_neg_type_ratio`
- Representation: `subtoken_pooling`, `words_splitter_type`, head sizes, dropout, fusion knobs
- Multitask coefficients: `token_loss_coef`, `span_loss_coef`, decoder/adjacency/relation loss coefs
- Optional snapshots: `encoder_config`, `labels_encoder_config`, `labels_decoder_config`

**Core principle: GLiNER config affects architecture, inference, preprocessing**
- These get serialized as `gliner_config.json` via `save_pretrained`
- They should be set BEFORE training

**Validation rules the report recommends:**
- If parameter changes how inputs are tokenized/spans enumerated/prompts constructed/label spaces sampled -> GLiNER config
- span_mode standardize on `token_level` (not `token-level`)
- Backbone config snapshot is read-only unless building new backbone

### Analysis Task

Critically compare our `config_cli.py` `_GLINER_RULES` and `training_cli.py` `_FIELD_SCHEMA` model section against the report. Document:
1. Coverage of all recommended GLiNER config fields
2. Whether our validation rules (ranges, literals) match the report's recommendations
3. Whether the config_cli and training_cli model sections are consistent with each other
4. Whether cross-field validation (method-specific requirements) is complete
5. Any fields in the report that we're missing or handling incorrectly
