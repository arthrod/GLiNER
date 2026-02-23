# Research Topic: Standard HuggingFace Configuration

## Source: deep-research-report.md (sections on TrainingArguments, Hub integration, W&B)

### Key Findings from Report

**TrainingArguments fields that belong under `training:`:**
- Optimization: `learning_rate`, `weight_decay`, `optim`, `adam_*`, schedulers, warmup
- Batch and steps: `per_device_*_batch_size`, `gradient_accumulation_steps`, `max_steps`, `num_train_epochs`
- Precision/perf: `bf16`, `fp16`, `tf32`, `gradient_checkpointing`, `torch_compile`
- Logging/saving: `logging_steps`, `save_steps`, `evaluation_strategy`, `eval_steps`, `save_total_limit`
- Integrations: `report_to`, `run_name`, `push_to_hub` and hub settings
- Distributed: `deepspeed`, `fsdp`, ddp knobs

**Hub sync fields (standard TrainingArguments):**
- `push_to_hub`, `hub_model_id`, `hub_strategy`, `hub_token`, `hub_private_repo`, `hub_always_push`

**W&B tracking:**
- `report_to="wandb"`, `run_name`, `logging_steps`
- Extra script-level: `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_API_KEY`, `WANDB_LOG_MODEL`

**Red flags from W&B dump:**
- `warmup_steps` shown as 0.05 (float) - should be `warmup_ratio: 0.05` with `warmup_steps: 0`

**Professional extras to include even if not wired yet:**
- Resume: `resume_from_checkpoint`
- Determinism: `full_determinism`
- Model selection: `load_best_model_at_end`, `metric_for_best_model`, `greater_is_better`
- Distributed: `deepspeed`, `fsdp` config keys

### Analysis Task

Critically compare our `training_cli.py` and `template.yaml` against the report's recommendations for standard HuggingFace TrainingArguments. Document:
1. Which recommended fields are present and correctly placed
2. Which recommended fields are missing
3. Which fields deviate from the report's recommendations
4. Specific gaps in Hub integration and W&B support
5. Whether `warmup_ratio` vs `warmup_steps` is handled correctly
