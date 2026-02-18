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

### mem-1771378913-db23
> failure: cmd=python3 inline load_data runtime check with validate kwarg, exit=1, error='TypeError: load_data() got an unexpected keyword argument validate', next=inspect ptbr.data.load_data signature and call without unsupported validate kwarg
<!-- tags: testing, error-handling, ptbr, data | created: 2026-02-18 -->

### mem-1771378888-80d1
> failure: cmd=rm -rf /tmp/gliner-staging-integration && git worktree add /tmp/gliner-staging-integration origin/dev, exit=policy_blocked, error='blocked by policy', next=use git worktree add --force without rm -rf in this environment
<!-- tags: tooling, error-handling, git | created: 2026-02-18 -->

### mem-1771378741-02e5
> PR13: ptbr/data.py local file load_data now validates requested custom text_column/ner_column against available local keys and raises ValueError before remapping; added regression check in ptbr/tests/test_validation.py.
<!-- tags: ptbr, data, error-handling, testing | created: 2026-02-18 -->

### mem-1771378588-60cf
> PR13: ptbr/tests/test_validation.py now uses subprocess.run([sys.executable, ...], stdout/stderr DEVNULL) instead of os.system in CLI smoke tests; also renamed unused unpacked errs to _errs to avoid Ruff RUF059.
<!-- tags: ptbr, testing, cli, error-handling | created: 2026-02-18 -->

### mem-1771378572-cd08
> failure: cmd=python3 -m ptbr --file-or-repo examples/sample_data.json --validate, exit=1, error='ModuleNotFoundError: No module named typer', next=install ptbr CLI dependencies (typer) or verify via static checks in dependency-constrained environment
<!-- tags: tooling, error-handling, dependencies, ptbr | created: 2026-02-18 -->

### mem-1771378444-0712
> PR13: ptbr/training_cli.py should keep a single import copy and must close existing _file_handler before replacing it in _attach_file_handler to avoid descriptor leaks.
<!-- tags: ptbr, cli, error-handling | created: 2026-02-18 -->

### mem-1771378335-f27e
> PR13: ptbr/__main__.py config_cmd must pass Path(file) to print_and_log_result; passing str risks runtime .parent errors.
<!-- tags: ptbr, cli, error-handling | created: 2026-02-18 -->

### mem-1771378243-830f
> PR13: ptbr/tests/test_training_cli.py had duplicate TestCLI methods (output_folder_empty_ok and output_folder_allows_validation_artifacts); removed earlier duplicate definitions to prevent shadowing and preserve launch-call assertions.
<!-- tags: testing, ptbr, cli | created: 2026-02-18 -->

### mem-1771378157-e6d6
> PR13: generate_noisy_jsonl Step 3 parser block had extra leading spaces causing invalid syntax; fixed indentation and added explicit num_corrupt>=len(ALL_NOISE) guard before guaranteed assignment
<!-- tags: testing, ptbr, error-handling | created: 2026-02-18 -->

### mem-1771378131-ea72
> failure: cmd=ruff check ptbr/tests/generate_noisy_jsonl.py, exit=127, error='command not found: ruff', next=use python3 -m py_compile or install ruff/uvx for lint verification
<!-- tags: tooling, error-handling, testing, lint | created: 2026-02-18 -->

### mem-1771378012-b94c
> failure: cmd=cat .ralph/agent/scratchpad.md, exit=1, error='No such file or directory', next=initialize required scratchpad file at .ralph/agent/scratchpad.md before reading/appending
<!-- tags: tooling, error-handling, ralph | created: 2026-02-18 -->

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

### mem-1771378943-f889
> PR13 integration: created staging from origin/dev, merged dev via merge commit 56efbfb, verified via py_compile + targeted ptbr.data runtime check, and pushed origin/staging.
<!-- tags: git, release, ptbr, staging | created: 2026-02-18 -->

### mem-1771372933-386a
> GLiNERConfig.model_type token_level routing bug is already fixed on dev by commit 6972469 (2026-02-18) with regression coverage in ptbr/tests/test_config_cli.py; only stale token-level docstring references may remain.
<!-- tags: config, model-routing, review | created: 2026-02-18 -->
