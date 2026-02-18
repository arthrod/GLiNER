# Scratchpad

## 2026-02-18T00:00:00Z
- Objective pivoted to PR14 (`arthrod-GLiNER-pr14-comments.md`) after PR13 staging integration is complete.
- Primary requirement is to prove config propagation end-to-end:
  - YAML values are parsed/validated by PTBR.
  - Values are forwarded into `gliner.model.BaseGLiNER.train_model(...)`.
  - Forwarded kwargs reach Transformers `TrainingArguments`.
  - Torch-facing behavior can be asserted deterministically without GPU (FakeTensor/meta pathways).
- Constraint: one atomic task per iteration. This iteration will establish and commit an actionable execution plan plus task queue.

Plan narrative:
1. Fetch PR14 branch and inspect current diff against `origin/dev`.
2. Introduce focused tests for propagation of representative fields:
   - PTBR forwarding: `num_steps`, `train_batch_size`, `eval_batch_size`, `bf16`, `output_dir`, dataloader flags, run metadata.
   - Transformers capture: patch `TrainingArguments.__init__` / constructor boundary to assert exact values.
   - Torch capture: use CPU-safe paths and FakeTensor/meta checks to verify dtype/device-related config influence where applicable.
3. Implement minimal code fixes in `ptbr/training_cli.py`, `train.py`, and/or `gliner/model.py` only for failing propagation tests.
4. Run targeted verification (pytest subset or static fallback where deps are missing), merge to `staging`, and push.

Decision confidence:
- 88/100 to prioritize test-first propagation assertions before broad refactors.
- Reason: comments show most failures are forwarding mismatches and hardcoded overrides; deterministic tests constrain fixes.

Implementation notes after source inspection:
- `ptbr.training_cli._launch_training` already forwards many training kwargs but currently does not pass:
  - `dataloader_pin_memory`
  - `dataloader_persistent_workers`
  - `dataloader_prefetch_factor`
  - run metadata (`run_name`, optional tags/description bridge)
- `train.py` still hardcodes:
  - `output_dir="models"` instead of `cfg.data.root_dir`
  - `per_device_eval_batch_size=cfg.training.train_batch_size` instead of optional eval override
  - `bf16=True` instead of config-driven precision

Concrete test matrix to implement (PR14):
1. PTBR forwarding test (`ptbr/tests/test_training_cli.py` or new focused test module):
   - Patch/stub `GLiNER.from_config` and capture `model.train_model(...)` kwargs.
   - Feed minimal validated config and assert exact forwarding for:
     - scheduler/batch/optimizer/loss fields
     - dataloader pin/persistent/prefetch fields
     - precision (`bf16`, `fp16`, `use_cpu`)
     - run metadata handoff (`run.name` -> `run_name` if supported)
2. GLiNER-to-Transformers boundary test (`tests/`):
   - Patch `gliner.model.TrainingArguments` constructor (or wrapper boundary) and assert kwargs from `BaseGLiNER.create_training_args`.
   - Include fields often lost in translation: `label_smoothing`, `gradient_accumulation_steps`, `report_to`, `save_steps`, `eval_strategy`.
3. Torch-safe behavior test (`tests/`):
   - Use CPU-safe assertions with torch available in test env.
   - Validate no CUDA requirement.
   - For deterministic signal, use `FakeTensorMode`/meta tensor checks to ensure model move/precision path follows config values without executing heavy kernels.
4. Regression test for `train.py`:
   - Stub `build_model` return object and assert `train_model` receives `output_dir`, eval batch size fallback, and precision values from config, not hardcoded constants.
