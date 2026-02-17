# PR Comment Implementation Report

## Summary

- **Total comments**: 16
- **Implemented**: 11
- **Skipped**: 5

## Implemented Changes

### Comment 1: Add negative relation index validation
- **Reviewer**: coderabbitai[bot]
- **File**: `ptbr/__init__.py`
- **What was done**: Added validation to reject negative `head`/`tail` relation indices with a dedicated error message before upper-bound checks.

### Comment 4: Avoid hardcoded CLI output filenames
- **Reviewer**: gemini-code-assist[bot]
- **File**: `ptbr/__main__.py`
- **What was done**: Added `--output-embeddings-path` and `--output-labels-path` options and used them instead of hardcoded `label_embeddings.pt` and `labels.json`.

### Comment 5: Use explicit label index in span extraction
- **Reviewer**: gemini-code-assist[bot]
- **File**: `ptbr/__init__.py`
- **What was done**: Updated label extraction to use `span[2]` explicitly and added structural/type guards so malformed spans are skipped safely.

### Comment 6: Add CLI split option
- **Reviewer**: gemini-code-assist[bot]
- **File**: `ptbr/__main__.py`
- **What was done**: Added `--split` option and forwarded it to `load_data(..., split=split)`.

### Comment 7: Add typer to runtime dependencies
- **Reviewer**: chatgpt-codex-connector[bot]
- **File**: `pyproject.toml`
- **What was done**: Added `typer` to `project.dependencies` so `python -m ptbr` works in standard installs.

### Comment 8: Include ptbr package in distribution
- **Reviewer**: chatgpt-codex-connector[bot]
- **File**: `pyproject.toml`
- **What was done**: Updated setuptools package discovery includes to ship `ptbr` and `ptbr.*` in built artifacts.

### Comment 9: Validate before label extraction in prepare()
- **Reviewer**: chatgpt-codex-connector[bot]
- **File**: `ptbr/__init__.py`
- **What was done**: Reordered `prepare()` to run `validate_data()` before `extract_labels()` when `validate=True`, preventing validation from being bypassed by extraction-time failures.

### Comment 10: Tighten validate_data return type
- **Reviewer**: sourcery-ai[bot]
- **File**: `ptbr/__init__.py`
- **What was done**: Changed return annotation from bare `tuple` to `Tuple[bool, List[str]]` and imported `Tuple` from `typing`.

### Comment 11: Catch negative relation indices
- **Reviewer**: sourcery-ai[bot]
- **File**: `ptbr/__init__.py`
- **What was done**: Addressed by the same relation-index validation change as Comment 1 (negative `head`/`tail` now rejected explicitly).

### Comment 12: Improve missing dataset-column errors
- **Reviewer**: sourcery-ai[bot]
- **File**: `ptbr/__init__.py`
- **What was done**: Added explicit HuggingFace split column validation and raised a clear `ValueError` listing missing and available columns.

### Comment 13: Add CLI split passthrough
- **Reviewer**: sourcery-ai[bot]
- **File**: `ptbr/__main__.py`
- **What was done**: Addressed by the same CLI `--split` addition as Comment 6.

## Skipped Comments

### Comment 2: Add trust_remote_code control in prepare()
- **Reviewer**: gemini-code-assist[bot]
- **File**: `ptbr/__init__.py`
- **Reason**: SKIP — risky — In this codebase, core GLiNER internals hardcode `trust_remote_code=True`; adding a `prepare()` flag only in `ptbr` would either be ineffective or break behavior without coordinated core-library changes.

### Comment 3: Add --trust-remote-code CLI flag
- **Reviewer**: gemini-code-assist[bot]
- **File**: `ptbr/__main__.py`
- **Reason**: SKIP — risky — The same core-library constraint applies; a CLI flag here cannot safely enforce the requested behavior without deeper changes outside the scoped comment targets.

### Comment 14: Walkthrough/pre-merge summary
- **Reviewer**: coderabbitai[bot]
- **File**: `N/A`
- **Reason**: SKIP — trivial — This is a generated overview/checklist comment, not a concrete code issue with a specific fix target.

### Comment 15: PR summary message
- **Reviewer**: gemini-code-assist[bot]
- **File**: `N/A`
- **Reason**: SKIP — trivial — This section summarizes the PR and does not request a direct code change.

### Comment 16: Reviewer guide
- **Reviewer**: sourcery-ai[bot]
- **File**: `N/A`
- **Reason**: SKIP — trivial — This is informational review metadata, not an actionable defect report.
