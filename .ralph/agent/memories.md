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

### mem-1771379248-8e0d
> failure: cmd=git switch staging && git merge --no-ff ralph/prime-iris -m 'Merge PR15 review fixes into staging', exit=1, error='merge conflicts in ptbr/data.py and ptbr/tests/test_validation.py', next=resolve conflicts by keeping stricter per-row custom-column checks plus subprocess CLI helper updates, then commit merge
<!-- tags: git, merge, error-handling, ptbr | created: 2026-02-18 -->

### mem-1771379187-4792
> failure: cmd=git rev-parse --short ralph/prime-iris staging, exit=128, error='fatal: Needed a single revision', next=resolve each ref separately and confirm staging branch existence via git show-ref
<!-- tags: tooling, error-handling, git | created: 2026-02-18 -->

### mem-1771379091-1025
> failure: cmd=rg -n 'output_dir="/tmp|/tmp/gliner_test_output|sig = inspect.signature\(_create_training_args_via_classmethod\)' tests/test_training_validation.py -S, exit=1, error='no matches (expected after cleanup)', next=use 'rg ... || true' when absence is the expected assertion
<!-- tags: tooling, error-handling, search, testing | created: 2026-02-18 -->

### mem-1771378885-0f8f
> failure: cmd=ls -l /home/arthrod/workspace/gliner_review/GliNER/.worktrees/peppy-willow/tests/test_training_validation.py, exit=2, error='No such file or directory', next=locate PR15 test file path via repo-wide search in source branch
<!-- tags: tooling, error-handling, search, testing | created: 2026-02-18 -->

### mem-1771378873-18d6
> failure: cmd=rg -n 'create_training_args_masking_matches|sig\s*=\s*inspect\.signature|inspect\.signature\(' ptbr/tests/test_training_cli.py ptbr/tests/test_validation.py -S, exit=1, error='no matches', next=inspect nearby test modules and map PR comments to current file names before applying edits
<!-- tags: tooling, error-handling, search, testing | created: 2026-02-18 -->

### mem-1771378866-417c
> failure: cmd=rg --files | rg -n 'training_validation\.py|training.*validation|validation.*training' -S, exit=1, error='no matches', next=search under scoped test directories (tests, ptbr/tests) and use actual file names present
<!-- tags: tooling, error-handling, search | created: 2026-02-18 -->

### mem-1771378857-8171
> failure: cmd=rg -n '\bsig\s*=\s*inspect\.signature\(' tests/test_training_validation.py -S && sed -n '380,440p' tests/test_training_validation.py, exit=2, error='No such file or directory', next=locate renamed/moved validation test file via rg --files and apply fix at actual path
<!-- tags: tooling, error-handling, search, testing | created: 2026-02-18 -->

### mem-1771378786-31d2
> PR13: local load_data path now raises ValueError for missing remapped text_column/ner_column and reports available local columns; regression covered by test_column_remapping_missing_custom_column in ptbr/tests/test_validation.py.
<!-- tags: ptbr, data-loading, validation, testing | created: 2026-02-18 -->

### mem-1771378701-d5b5
> failure: cmd=rg -n 'custom-column|custom column|text_column|ner_column|ptbr/data.py|Missing required' /home/arthrod/workspace/gliner_review/GliNER/.worktrees/peppy-willow/arthrod-GLiNER-pr15-comments.md -S, exit=1, error='no matches', next=open the review artifact directly and search broader terms
<!-- tags: tooling, error-handling, search | created: 2026-02-18 -->

### mem-1771378629-3ae8
> PR13: ptbr/tests/test_validation.py CLI smoke checks should invoke  via subprocess.run and skip with a pass-detail when  is unavailable in minimal environments.
<!-- tags: ptbr, testing, cli, error-handling | created: 2026-02-18 -->

### mem-1771378594-1640
> failure: cmd=python3 ptbr/tests/test_validation.py, exit=1, error='CLI --validate exits 0 on valid data failed because python3 -m ptbr raises ModuleNotFoundError: typer', next=handle missing CLI dependency in test helper or run under interpreter/environment with typer installed
<!-- tags: testing, tooling, error-handling, ptbr | created: 2026-02-18 -->

### mem-1771378594-0019
> failure: cmd=rg -n 'os.system|python -m ptbr --file-or-repo' ptbr/tests/test_validation.py -S, exit=1, error='no matches', next=treat no-match grep as expected after migration and use explicit status checks when command may legitimately return 1
<!-- tags: tooling, error-handling, search | created: 2026-02-18 -->

### mem-1771378536-7015
> failure: cmd=rg -n 'ptbr/tests/test_validation.py|os.system|subprocess.run|custom-column|task-1771378101|task-1771378103' /home/arthrod/workspace/gliner_review/GliNER/.worktrees/peppy-willow/arthrod-GLiNER-pr15-comments.md -S, exit=1, error='no matches', next=use task descriptions and inspect target files directly when review export lacks references
<!-- tags: tooling, error-handling, search | created: 2026-02-18 -->

### mem-1771378495-785b
> failure: cmd=tail -n 80 .ralph/agent/scratchpad.md, exit=1, error='No such file or directory', next=initialize required scratchpad file at .ralph/agent/scratchpad.md before reading/appending
<!-- tags: tooling, error-handling, ralph | created: 2026-02-18 -->

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

### mem-1771379328-98e9
> PR15 final integration: staging merge commit 60dddae resolved ptbr/data.py and ptbr/tests/test_validation.py conflicts by keeping per-row custom-column missing checks plus typer-guarded subprocess CLI smoke tests; verification used py_compile + standalone validation suite (45/45).
<!-- tags: pr15, staging, merge, testing, ptbr | created: 2026-02-18 -->

### mem-1771379117-6cb5
> PR15: tests/test_training_validation.py now requires explicit output_dir in helper constructors (wired via pytest tmp_path), removes dead sig signature line, and parses config.yaml training fields with yaml.safe_load via pytest.importorskip('yaml').
<!-- tags: testing, pr15, training | created: 2026-02-18 -->

### mem-1771378965-62da
> PR15 baseline validation suite from origin/claude/add-validation-tests-eIXm5 (commit 32770dc) is now imported on ralph/prime-iris as commit d388444 via tests/test_training_validation.py; next iterations should apply review fixes on top of this file.
<!-- tags: testing, pr15, training | created: 2026-02-18 -->

### mem-1771372933-386a
> GLiNERConfig.model_type token_level routing bug is already fixed on dev by commit 6972469 (2026-02-18) with regression coverage in ptbr/tests/test_config_cli.py; only stale token-level docstring references may remain.
<!-- tags: config, model-routing, review | created: 2026-02-18 -->
