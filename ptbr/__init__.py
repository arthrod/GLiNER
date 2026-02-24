"""ptbr - GLiNER fine-tuning toolkit for pt-BR and general use.

Submodules:
    ptbr.data          - Data loading, validation, and preparation
    ptbr.config_cli    - YAML configuration validation
    ptbr.training_cli  - Training launch and validation

CLI usage:
    python -m ptbr data      --file-or-repo data.json --validate
    python -m ptbr config    --file config.yaml --validate
    python -m ptbr train     config.yaml --output-folder ./runs
"""

from ptbr.data import (
    GLiNERData,
    prepare,
    load_data,
    validate_data,
    extract_labels,
)

__all__ = [
    "GLiNERData",
    "extract_labels",
    "load_data",
    "prepare",
    "validate_data",
]
