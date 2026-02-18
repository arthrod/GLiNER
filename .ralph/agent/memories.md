# Memories

## Patterns

### mem-1771372862-749d
> GLiNER training/eval uses custom Trainer.get_train_dataloader/get_eval_dataloader that directly feed dataset rows to the collator and bypass HF _remove_unused_columns; create_training_args not forcing remove_unused_columns=False is intentional in this path.
<!-- tags: training, trainer, collator | created: 2026-02-18 -->

### mem-1771372427-9555
> GLiNER config routing convention: span_mode values are markerV0/token_level (underscore), and legacy GLiNERConfig.model_type detection must use token_level checks for decoder, bi-encoder, relex, and base token variants.
<!-- tags: config, model-routing, token_level | created: 2026-02-17 -->

## Decisions

## Fixes

### mem-1771372910-e0c5
> failure: cmd=python3 -m pytest -q ptbr/tests/test_config_cli.py -k model_type, exit=1, error='No module named pytest', next=use source-level static checks and git-history-backed verification when pytest is unavailable
<!-- tags: testing, tooling, error-handling | created: 2026-02-18 -->

### mem-1771372662-315f
> failure: cmd=python3 -m pip --version, exit=1, error='No module named pip', next=use uvx to execute pytest without relying on system pip
<!-- tags: tooling, error-handling, python, testing | created: 2026-02-17 -->

### mem-1771372536-fb9b
> failure: cmd=uv run --python 3.11 --group dev pytest tests/test_trainer_column_pruning.py -q, exit=1, error='uv still resolves unsupported python_full_version==3.9 split due jieba3 marker conflict', next=use source-level tests and run via pip-installed pytest without project dependency resolution
<!-- tags: tooling, error-handling, testing, uv | created: 2026-02-17 -->

### mem-1771372508-dbfd
> failure: cmd=uv run --group dev pytest tests/test_trainer_column_pruning.py -q, exit=1, error='dependency resolution failed due jieba3/Python marker conflict from requires-python>=3.8', next=run uv with explicit Python 3.11 for this test invocation
<!-- tags: tooling, error-handling, testing, uv | created: 2026-02-17 -->

### mem-1771372400-7cf2
> failure: cmd=python3 importlib load of gliner/config.py assertions, exit=1, error=ModuleNotFoundError: transformers, next=limit verification to static checks in dependency-constrained environment
<!-- tags: testing, tooling, error-handling, python | created: 2026-02-17 -->

### mem-1771372388-b05d
> failure: cmd=python3 inline assertions importing gliner.config, exit=1, error=ModuleNotFoundError: torch (triggered via gliner.__init__ import chain), next=load gliner/config.py directly with importlib to avoid package __init__
<!-- tags: testing, tooling, error-handling, python | created: 2026-02-17 -->

### mem-1771372373-797f
> failure: cmd=uv run pytest -q ptbr/tests/test_config_cli.py -k model_type, exit=1, error=uv dependency resolution unsatisfiable due jieba3/tokenizers across supported python range, next=run direct python3 assertions that exercise GLiNERConfig.model_type branches
<!-- tags: testing, tooling, error-handling, uv | created: 2026-02-17 -->

### mem-1771372363-0827
> failure: cmd=rg -n 'remove_unused_columns|words_mask|span_idx|text_lengths|adj_matrix' arthrod-GLiNER-pr4-comments.md arthrod-GLiNER-pr5-comments.md report.md agents.md -S, exit=1, error='no matches', next=inspect targeted commits and code directly for intent and behavior
<!-- tags: tooling, error-handling, search | created: 2026-02-17 -->

### mem-1771372345-5184
> failure: cmd=python3 -m pytest -q ptbr/tests/test_config_cli.py -k model_type, exit=1, error=No module named pytest, next=use local assertion script for behavioral verification or install pytest in environment
<!-- tags: testing, tooling, error-handling | created: 2026-02-17 -->

### mem-1771372341-382d
> failure: cmd=rg -n 'train_model|create_training_args|Trainer|remove_unused_columns|data_collator|get_train_dataloader|get_eval_dataloader' tests -S, exit=1, error='no matches in tests', next=add focused regression test for custom Trainer dataloaders preserving collator input keys
<!-- tags: tooling, error-handling, testing | created: 2026-02-17 -->

### mem-1771372335-089a
> failure: cmd=python -m pytest -q ptbr/tests/test_config_cli.py -k model_type, exit=127, error=command not found: python, next=run via python3 -m pytest
<!-- tags: testing, tooling, error-handling | created: 2026-02-17 -->

### mem-1771372327-c555
> failure: cmd=pytest -q ptbr/tests/test_config_cli.py -k model_type, exit=127, error=command not found: pytest, next=run via python -m pytest
<!-- tags: testing, tooling, error-handling | created: 2026-02-17 -->

### mem-1771372294-83e6
> failure: cmd=python3 inspect transformers Trainer, exit=1, error='ModuleNotFoundError: No module named transformers', next=use repository source and git history for behavior analysis; run tests only if env dependencies are available
<!-- tags: tooling, error-handling, dependencies | created: 2026-02-17 -->

### mem-1771372278-3715
> failure: cmd=python - <<'PY' ..., exit=127, error='zsh: command not found: python', next=use python3 for local source inspection commands
<!-- tags: tooling, error-handling, python | created: 2026-02-17 -->

### mem-1771372248-5121
> failure: cmd=append scratchpad with echo lines containing backticks, exit=1, error=zsh parse error near (), next=use here-doc with single-quoted delimiter to avoid command substitution
<!-- tags: tooling, error-handling | created: 2026-02-17 -->

### mem-1771372220-89af
> failure: cmd=cat .ralph/agent/scratchpad.md, exit=1, error='No such file or directory', next=initialize required scratchpad file at .ralph/agent/scratchpad.md before reading/appending
<!-- tags: tooling, error-handling, ralph | created: 2026-02-17 -->

## Context

### mem-1771372933-386a
> GLiNERConfig.model_type token_level routing bug is already fixed on dev by commit 6972469 (2026-02-18) with regression coverage in ptbr/tests/test_config_cli.py; only stale token-level docstring references may remain.
<!-- tags: config, model-routing, review | created: 2026-02-18 -->
