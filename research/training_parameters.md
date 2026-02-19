# Research Topic: Training Parameters and Loss Configuration

## Source: deep-research-report.md (sections on Trainer extensions, LoRA, loss knobs)

### Key Findings from Report

**GLiNER-specific Trainer extensions (training-only, not standard HF):**
- `others_lr`, `others_weight_decay`
- `focal_loss_alpha`, `focal_loss_gamma`, `focal_loss_prob_margin`
- `label_smoothing`, `loss_reduction`
- `negatives`, `masking`

**LoRA/PEFT recommendations:**
- LoRA belongs in "training config" not "model config" (optimization strategy)
- Start with `r in {8, 16, 32}`, `lora_alpha ~ 2r`, `lora_dropout in [0.05, 0.1]`
- Apply to attention projections (`target_modules` model-dependent)
- Keep `bias="none"` unless needed
- Key PEFT fields: `r`, `lora_alpha`, `lora_dropout`, `bias`, `target_modules`, `modules_to_save`, `init_lora_weights`, `use_rslora`

**Parameter flow (from Mermaid diagram):**
- YAML -> model (GLiNER.from_config) -> GLiNER model + processors
- YAML -> training (GLiNER.train_model) -> HF Trainer + custom GLiNER Trainer
- YAML -> peft (Script applies PEFT to backbone) -> model
- YAML -> data (Dataset loader + preprocessing) -> Trainer

**Key decision rule:**
- If parameter changes how gradients flow / optimizer / evaluate / save / log -> trainer config
- freeze_components is listed under training but could be GLiNER-specific

### Analysis Task

Critically compare our `training_cli.py` training and lora schema sections against the report. Document:
1. Whether GLiNER extension fields are correctly identified and separated from standard HF
2. Whether LoRA configuration matches PEFT best practices from the report
3. Whether our `_launch_training` correctly maps config fields to `model.train_model()` kwargs
4. Missing training parameters the report recommends
5. Whether loss function parameters are correctly wired (focal loss, label smoothing)
6. Whether the parameter flow matches the report's recommended architecture
