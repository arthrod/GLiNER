# Research Topic: Integration and End-to-End Architecture

## Source: deep-research-report.md (sections on canonical YAML, routing, implementation conventions)

### Key Findings from Report

**Canonical YAML structure recommended:**
```
experiment:    -> script-level metadata
model:         -> GLiNER config (serialized to gliner_config.json)
peft:          -> LoRA config (training strategy)
data:          -> script-level dataset paths + schema + preprocessing
training:      -> HF TrainingArguments + GLiNER extensions
logging:       -> W&B / extra logging controls
```

**Our actual YAML structure (template.yaml):**
```
run:           -> metadata
model:         -> GLiNER config
data:          -> dataset paths
training:      -> training parameters
lora:          -> LoRA config
environment:   -> Hub, W&B, hardware
```

**Key integration points:**
- config_cli.py expects `gliner_config:` and `lora_config:` sections
- training_cli.py expects `run:`, `model:`, `data:`, `training:`, `lora:`, `environment:` sections
- __main__.py routes to both but they have DIFFERENT YAML structures

**Report recommendations on what train.py should forward:**
- Hub upload args not currently forwarded
- W&B run naming not forwarded
- `label_smoothing` not forwarded
- `remove_unused_columns: false` often essential

**Data section report recommends but we don't have:**
- Document schema fields (tokens, ner, relations)
- Long document chunking configuration
- preprocessing section

### Analysis Task

Critically compare the end-to-end integration of our system against the report. Document:
1. Whether config_cli.py and training_cli.py use COMPATIBLE YAML structures (they currently don't!)
2. Whether the canonical YAML structure from the report is achievable with our current code
3. Whether __main__.py correctly routes between the two CLIs
4. Missing data preprocessing features (chunking, schema validation)
5. Whether the parameter flow diagram from the report matches our implementation
6. Whether `train.py` (upstream GLiNER) forwards all the parameters our CLI validates
7. Concrete recommendations to unify the config format between config_cli and training_cli
