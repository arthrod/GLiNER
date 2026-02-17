"""GLiNER Training CLI.

Typer-based command-line interface for validating configuration and launching
GLiNER training runs.  Every config field is validated before training begins.
Fields that require a value error out; fields using a default emit a warning.
A rich summary is printed and persisted to a log file.

Usage:
    python -m ptbr.training_cli --validate config.yaml
    python -m ptbr.training_cli --output-folder ./runs config.yaml
    python -m ptbr.training_cli --output-folder ./runs --resume config.yaml
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------
app = typer.Typer(
    name="gliner-train",
    help="Validate configuration and launch GLiNER training runs.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Rich console (stderr so stdout stays clean for piping)
# ---------------------------------------------------------------------------
console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Module-level logger  --  prints to terminal via Rich AND writes to a file
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(message)s"
logger = logging.getLogger("ptbr.training_cli")
logger.setLevel(logging.DEBUG)

# Rich handler (terminal)
_rich_handler = RichHandler(
    console=console,
    show_time=True,
    show_path=False,
    markup=True,
)
_rich_handler.setLevel(logging.DEBUG)
logger.addHandler(_rich_handler)

# File handler is attached later once we know the output folder.
_file_handler: Optional[logging.FileHandler] = None


def _attach_file_handler(log_path: Path) -> None:
    """Create and attach a file handler to the module logger."""
    global _file_handler
    if _file_handler is not None:
        logger.removeHandler(_file_handler)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    )
    logger.addHandler(_file_handler)


# ======================================================================== #
#  SCHEMA  --  declarative description of every config field               #
# ======================================================================== #

# Each entry:  (dotted_key, python_type, required?, default_value, description)
# For required fields default_value is ignored (use None as placeholder).

_FIELD_SCHEMA: list[tuple[str, type | tuple[type, ...], bool, Any, str]] = [
    # -- run --
    ("run.name",        str,    True,   None,       "Run name"),
    ("run.description", str,    False,  "",         "Run description"),
    ("run.tags",        list,   False,  [],         "Run tags"),
    ("run.seed",        int,    False,  42,         "Random seed"),

    # -- model --
    ("model.model_name",            str,    True,   None,           "Backbone model name/path"),
    ("model.name",                  str,    False,  "gliner",       "Model display name"),
    ("model.labels_encoder",        (str, type(None)),  False,  None,   "Bi-encoder labels model"),
    ("model.labels_decoder",        (str, type(None)),  False,  None,   "Decoder model for generative labels"),
    ("model.decoder_mode",          (str, type(None)),  False,  None,   "Decoder mode (span/prompt)"),
    ("model.full_decoder_context",  bool,   False,  True,           "Full context in decoder"),
    ("model.blank_entity_prob",     float,  False,  0.1,            "Blank entity probability"),
    ("model.decoder_loss_coef",     float,  False,  0.5,            "Decoder loss coefficient"),
    ("model.relations_layer",       (str, type(None)),  False,  None,   "Relation extraction layer"),
    ("model.triples_layer",         (str, type(None)),  False,  None,   "Triples layer"),
    ("model.embed_rel_token",       bool,   False,  True,           "Embed relation token"),
    ("model.rel_token_index",       int,    False,  -1,             "Relation token index"),
    ("model.rel_token",             str,    False,  "<<REL>>",      "Relation marker token"),
    ("model.adjacency_loss_coef",   float,  False,  1.0,            "Adjacency loss coefficient"),
    ("model.relation_loss_coef",    float,  False,  1.0,            "Relation loss coefficient"),
    ("model.span_mode",             str,    True,   None,           "Span mode (markerV0 / token_level)"),
    ("model.max_width",             int,    False,  12,             "Max entity span width"),
    ("model.represent_spans",       bool,   False,  False,          "Explicit span representations"),
    ("model.neg_spans_ratio",       float,  False,  1.0,            "Negative spans ratio"),
    ("model.hidden_size",           int,    False,  512,            "Projection hidden size"),
    ("model.dropout",               float,  False,  0.4,            "Dropout probability"),
    ("model.fine_tune",             bool,   False,  True,           "Fine-tune encoder"),
    ("model.subtoken_pooling",      str,    False,  "first",        "Sub-token pooling strategy"),
    ("model.fuse_layers",           bool,   False,  False,          "Fuse layers"),
    ("model.post_fusion_schema",    (str, type(None)),  False,  "",  "Post-fusion schema"),
    ("model.num_post_fusion_layers", int,   False,  1,              "Post-fusion layer count"),
    ("model.num_rnn_layers",        int,    False,  1,              "Bi-LSTM layers"),
    ("model.max_len",               int,    True,   None,           "Max sequence length"),
    ("model.max_types",             int,    False,  25,             "Max entity types per example"),
    ("model.max_neg_type_ratio",    int,    False,  1,              "Max neg/pos type ratio"),
    ("model.words_splitter_type",   str,    False,  "whitespace",   "Word splitter"),
    ("model.embed_ent_token",       bool,   False,  True,           "Embed entity token"),
    ("model.class_token_index",     int,    False,  -1,             "Class token index"),
    ("model.ent_token",             str,    False,  "<<ENT>>",      "Entity marker token"),
    ("model.sep_token",             str,    False,  "<<SEP>>",      "Separator token"),
    ("model.token_loss_coef",       float,  False,  1.0,            "Token loss coefficient"),
    ("model.span_loss_coef",        float,  False,  1.0,            "Span loss coefficient"),
    ("model.encoder_config",        (dict, type(None)),   False,  None,   "Encoder config override"),
    ("model._attn_implementation",  (str, type(None)),    False,  None,   "Attention implementation"),
    ("model.vocab_size",            int,    False,  -1,             "Vocabulary size override"),

    # -- data --
    ("data.root_dir",       str,    True,   None,       "Root log directory"),
    ("data.train_data",     str,    True,   None,       "Training data path"),
    ("data.val_data_dir",   str,    False,  "none",     "Validation data path"),

    # -- training --
    ("training.prev_path",                  (str, type(None)),  False,  None,   "Pretrained checkpoint"),
    ("training.num_steps",                  int,    True,   None,       "Total training steps"),
    ("training.scheduler_type",             str,    False,  "cosine",   "LR scheduler type"),
    ("training.warmup_ratio",               float,  False,  0.1,        "Warmup ratio"),
    ("training.train_batch_size",           int,    True,   None,       "Per-device train batch size"),
    ("training.eval_batch_size",            (int, type(None)),  False,  None,   "Per-device eval batch size"),
    ("training.gradient_accumulation_steps", int,   False,  1,          "Gradient accumulation steps"),
    ("training.max_grad_norm",              float,  False,  1.0,        "Max gradient norm"),
    ("training.optimizer",                  str,    False,  "adamw_torch", "Optimizer"),
    ("training.lr_encoder",                 float,  True,   None,       "Encoder learning rate"),
    ("training.lr_others",                  float,  True,   None,       "Others learning rate"),
    ("training.weight_decay_encoder",       float,  False,  0.01,       "Encoder weight decay"),
    ("training.weight_decay_other",         float,  False,  0.01,       "Others weight decay"),
    ("training.loss_alpha",                 (float, int),  False,  -1,  "Focal loss alpha"),
    ("training.loss_gamma",                 (float, int),  False,  0,   "Focal loss gamma"),
    ("training.loss_prob_margin",           (float, int),  False,  0,   "Focal loss prob margin"),
    ("training.label_smoothing",            (float, int),  False,  0,   "Label smoothing"),
    ("training.loss_reduction",             str,    False,  "sum",      "Loss reduction"),
    ("training.negatives",                  float,  False,  1.0,        "Negative sampling ratio"),
    ("training.masking",                    str,    False,  "none",     "Masking strategy"),
    ("training.eval_every",                 int,    True,   None,       "Eval/save interval (steps)"),
    ("training.save_total_limit",           int,    False,  3,          "Max checkpoints kept"),
    ("training.logging_steps",              (int, type(None)),  False,  None,   "Logging interval"),
    ("training.bf16",                       bool,   False,  False,      "bfloat16 precision"),
    ("training.fp16",                       bool,   False,  False,      "float16 precision"),
    ("training.use_cpu",                    bool,   False,  False,      "Force CPU"),
    ("training.dataloader_num_workers",     int,    False,  2,          "Dataloader workers"),
    ("training.dataloader_pin_memory",      bool,   False,  True,       "Pin memory"),
    ("training.dataloader_persistent_workers", bool, False, False,      "Persistent workers"),
    ("training.dataloader_prefetch_factor",  int,   False,  2,          "Prefetch factor"),
    ("training.freeze_components",          (list, type(None)),  False,  None,  "Components to freeze"),
    ("training.compile_model",              bool,   False,  False,      "torch.compile"),
    ("training.size_sup",                   int,    False,  -1,         "Max supervised examples"),
    ("training.shuffle_types",              bool,   False,  True,       "Shuffle entity types"),
    ("training.random_drop",                bool,   False,  True,       "Random drop types"),

    # -- lora --
    ("lora.enabled",        bool,   False,  False,              "Enable LoRA"),
    ("lora.r",              int,    False,  8,                  "LoRA rank"),
    ("lora.lora_alpha",     int,    False,  16,                 "LoRA alpha"),
    ("lora.lora_dropout",   float,  False,  0.05,               "LoRA dropout"),
    ("lora.bias",           str,    False,  "none",             "LoRA bias mode"),
    ("lora.target_modules",  list,  False,  ["q_proj", "v_proj"], "LoRA target modules"),
    ("lora.task_type",      str,    False,  "TOKEN_CLS",        "PEFT task type"),
    ("lora.modules_to_save", (list, type(None)), False, None,   "Modules to save"),

    # -- environment --
    ("environment.push_to_hub",         bool,   False,  False,  "Push to HF Hub"),
    ("environment.hub_model_id",        (str, type(None)),  False,  None,   "HF Hub model id"),
    ("environment.hf_token",            (str, type(None)),  False,  None,   "HF token override"),
    ("environment.report_to",           str,    False,  "none", "Reporting backend"),
    ("environment.wandb_project",       (str, type(None)),  False,  None,   "WandB project"),
    ("environment.wandb_entity",        (str, type(None)),  False,  None,   "WandB entity"),
    ("environment.wandb_api_key",       (str, type(None)),  False,  None,   "WandB API key override"),
    ("environment.cuda_visible_devices", (str, type(None)), False, None,    "CUDA visible devices"),
]


# ======================================================================== #
#  HELPERS                                                                  #
# ======================================================================== #

def _deep_get(d: dict, dotted_key: str) -> tuple[bool, Any]:
    """Retrieve a value from a nested dict using a dotted key.

    Returns (found: bool, value).
    """
    keys = dotted_key.split(".")
    current = d
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return False, None
        current = current[k]
    return True, current


def _deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key."""
    keys = dotted_key.split(".")
    current = d
    for k in keys[:-1]:
        current = current.setdefault(k, {})
    current[keys[-1]] = value


def _type_name(t: type | tuple) -> str:
    if isinstance(t, tuple):
        return " | ".join(x.__name__ for x in t)
    return t.__name__


def _check_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    """Loose type check that treats int as float when float is expected."""
    if isinstance(expected, tuple):
        types = expected
    else:
        types = (expected,)

    if isinstance(value, bool) and bool not in types:
        return False

    # Allow int where float is expected
    if isinstance(value, int) and not isinstance(value, bool) and float in types:
        return True

    return isinstance(value, types)


# ======================================================================== #
#  VALIDATION ENGINE                                                        #
# ======================================================================== #

class ValidationResult:
    """Accumulates validation outcomes for every config field."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[tuple[str, str, Any]] = []   # (key, status, value)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_config(cfg: dict) -> ValidationResult:
    """Validate *every* field declared in ``_FIELD_SCHEMA``.

    Rules:
    - REQUIRED field missing or None  -> ERROR
    - Optional field missing          -> WARNING (default applied in-place)
    - Wrong type                      -> ERROR
    - Extra keys not in schema        -> WARNING (ignored but noted)
    """
    result = ValidationResult()

    known_keys: set[str] = set()

    for dotted_key, expected_type, required, default, description in _FIELD_SCHEMA:
        known_keys.add(dotted_key)
        found, value = _deep_get(cfg, dotted_key)

        if not found or (value is None and required):
            if required:
                msg = f"[REQUIRED] '{dotted_key}' is missing and has no default ({description})"
                result.errors.append(msg)
                result.info.append((dotted_key, "ERROR", "MISSING"))
                logger.error(msg)
            else:
                _deep_set(cfg, dotted_key, default)
                msg = f"[DEFAULT]  '{dotted_key}' not set -- using default: {default!r}"
                result.warnings.append(msg)
                result.info.append((dotted_key, "DEFAULT", default))
                logger.warning(msg)
            continue

        # Type check
        if not _check_type(value, expected_type):
            msg = (
                f"[TYPE]     '{dotted_key}' has type {type(value).__name__} "
                f"but expected {_type_name(expected_type)} ({description})"
            )
            result.errors.append(msg)
            result.info.append((dotted_key, "ERROR", value))
            logger.error(msg)
            continue

        # Valid
        result.info.append((dotted_key, "OK", value))
        logger.info(f"[OK]       '{dotted_key}' = {value!r}")

    # Detect extra keys
    _check_extra_keys(cfg, known_keys, result)

    return result


def _check_extra_keys(
    cfg: dict,
    known: set[str],
    result: ValidationResult,
    prefix: str = "",
) -> None:
    """Warn about keys present in the config but absent from the schema."""
    for key, value in cfg.items():
        full = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            _check_extra_keys(value, known, result, full)
        else:
            if full not in known:
                msg = f"[EXTRA]    '{full}' is not in the schema (will be ignored)"
                result.warnings.append(msg)
                logger.warning(msg)


# ======================================================================== #
#  CROSS-FIELD SEMANTIC CHECKS                                              #
# ======================================================================== #

_VALID_SPAN_MODES = {"markerV0", "token_level"}
_VALID_SCHEDULERS = {
    "linear", "cosine", "constant", "constant_with_warmup",
    "polynomial", "inverse_sqrt",
}
_VALID_OPTIMIZERS = {"adamw_torch", "adamw_hf", "adafactor", "sgd"}
_VALID_LOSS_REDUCTIONS = {"sum", "mean", "none"}
_VALID_MASKING = {"none", "global"}
_VALID_REPORT_TO = {"none", "wandb", "tensorboard", "all"}
_VALID_SUBTOKEN_POOLING = {"first", "mean"}
_VALID_LORA_BIAS = {"none", "all", "lora_only"}
_VALID_ATTN_IMPL = {"eager", "sdpa", "flash_attention_2"}


def semantic_checks(cfg: dict, result: ValidationResult) -> None:
    """Run cross-field and enum-value validations."""
    _check_enum(cfg, result, "model.span_mode", _VALID_SPAN_MODES)
    _check_enum(cfg, result, "training.scheduler_type", _VALID_SCHEDULERS)
    _check_enum(cfg, result, "training.optimizer", _VALID_OPTIMIZERS)
    _check_enum(cfg, result, "training.loss_reduction", _VALID_LOSS_REDUCTIONS)
    _check_enum(cfg, result, "training.masking", _VALID_MASKING)
    _check_enum(cfg, result, "environment.report_to", _VALID_REPORT_TO)
    _check_enum(cfg, result, "model.subtoken_pooling", _VALID_SUBTOKEN_POOLING)
    _check_enum(cfg, result, "lora.bias", _VALID_LORA_BIAS)

    # Optional enums (only check if non-null)
    _, attn = _deep_get(cfg, "model._attn_implementation")
    if attn is not None:
        _check_enum(cfg, result, "model._attn_implementation", _VALID_ATTN_IMPL)

    # Decoder fields require labels_decoder
    _, dec = _deep_get(cfg, "model.labels_decoder")
    if dec is not None:
        _, dm = _deep_get(cfg, "model.decoder_mode")
        if dm is None:
            msg = "'model.decoder_mode' should be set when 'model.labels_decoder' is used"
            result.warnings.append(msg)
            logger.warning(msg)

    # WandB requires project
    _, report = _deep_get(cfg, "environment.report_to")
    if report in ("wandb", "all"):
        _, proj = _deep_get(cfg, "environment.wandb_project")
        if not proj:
            msg = "'environment.wandb_project' is required when report_to includes wandb"
            result.errors.append(msg)
            logger.error(msg)

    # HF Hub requires model id
    _, push = _deep_get(cfg, "environment.push_to_hub")
    if push:
        _, hid = _deep_get(cfg, "environment.hub_model_id")
        if not hid:
            msg = "'environment.hub_model_id' is required when push_to_hub is true"
            result.errors.append(msg)
            logger.error(msg)

    # Positive numeric checks
    for key in (
        "training.num_steps", "training.train_batch_size", "training.eval_every",
        "training.lr_encoder", "training.lr_others",
    ):
        _, val = _deep_get(cfg, key)
        if val is not None and val <= 0:
            msg = f"'{key}' must be > 0, got {val}"
            result.errors.append(msg)
            logger.error(msg)

    # bf16 and fp16 mutual exclusivity
    _, bf16 = _deep_get(cfg, "training.bf16")
    _, fp16 = _deep_get(cfg, "training.fp16")
    if bf16 and fp16:
        msg = "Cannot enable both 'training.bf16' and 'training.fp16'"
        result.errors.append(msg)
        logger.error(msg)


def _check_enum(
    cfg: dict, result: ValidationResult, key: str, valid: set[str]
) -> None:
    _, val = _deep_get(cfg, key)
    if val is not None and val not in valid:
        msg = f"'{key}' = {val!r} is not one of {sorted(valid)}"
        result.errors.append(msg)
        logger.error(msg)


# ======================================================================== #
#  API CONNECTIVITY CHECKS                                                  #
# ======================================================================== #

def check_huggingface(cfg: dict, result: ValidationResult) -> None:
    """Make a real API call to HuggingFace to verify credentials."""
    _, push = _deep_get(cfg, "environment.push_to_hub")
    if not push:
        logger.info("[HF]       push_to_hub is false -- skipping HF API check")
        return

    _, token_override = _deep_get(cfg, "environment.hf_token")
    token = token_override or os.environ.get("HF_TOKEN", "")
    if not token:
        msg = "HuggingFace: push_to_hub is true but no token found (set HF_TOKEN or environment.hf_token)"
        result.errors.append(msg)
        logger.error(msg)
        return

    try:
        import requests
        resp = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            username = data.get("name", "unknown")
            logger.info(f"[HF]       Authenticated as '{username}'")
        else:
            msg = f"HuggingFace API returned status {resp.status_code}: {resp.text[:200]}"
            result.errors.append(msg)
            logger.error(msg)
    except Exception as exc:
        msg = f"HuggingFace API call failed: {exc}"
        result.errors.append(msg)
        logger.error(msg)


def check_wandb(cfg: dict, result: ValidationResult) -> None:
    """Make a real API call to WandB to verify credentials."""
    _, report = _deep_get(cfg, "environment.report_to")
    if report not in ("wandb", "all"):
        logger.info("[WANDB]    report_to does not include wandb -- skipping WandB API check")
        return

    _, key_override = _deep_get(cfg, "environment.wandb_api_key")
    key = key_override or os.environ.get("WANDB_API_KEY", "")
    if not key:
        msg = "WandB: report_to includes wandb but no API key found (set WANDB_API_KEY or environment.wandb_api_key)"
        result.errors.append(msg)
        logger.error(msg)
        return

    try:
        import requests
        resp = requests.get(
            "https://api.wandb.ai/graphql",
            headers={"Authorization": f"Bearer {key}"},
            json={"query": "{ viewer { username } }"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            username = (
                data.get("data", {}).get("viewer", {}).get("username", "unknown")
            )
            logger.info(f"[WANDB]    Authenticated as '{username}'")
        else:
            msg = f"WandB API returned status {resp.status_code}: {resp.text[:200]}"
            result.errors.append(msg)
            logger.error(msg)
    except Exception as exc:
        msg = f"WandB API call failed: {exc}"
        result.errors.append(msg)
        logger.error(msg)


# ======================================================================== #
#  RESUME CHECK                                                             #
# ======================================================================== #

def check_resume(cfg: dict, output_folder: Path, result: ValidationResult) -> None:
    """When --resume is passed, verify output_folder has a compatible checkpoint."""
    _, run_name = _deep_get(cfg, "run.name")
    if not run_name:
        msg = "Cannot resume: 'run.name' is missing"
        result.errors.append(msg)
        logger.error(msg)
        return

    # Look for checkpoint dirs inside output_folder
    checkpoint_dirs = sorted(output_folder.glob("checkpoint-*"))
    if not checkpoint_dirs:
        msg = f"Cannot resume: no checkpoint-* directories found in {output_folder}"
        result.errors.append(msg)
        logger.error(msg)
        return

    # Check for config.yaml in output_folder to verify run name matches
    saved_cfg_path = output_folder / "config.yaml"
    if not saved_cfg_path.exists():
        msg = f"Cannot resume: {saved_cfg_path} not found"
        result.errors.append(msg)
        logger.error(msg)
        return

    with open(saved_cfg_path) as f:
        saved_cfg = yaml.safe_load(f) or {}

    saved_name = saved_cfg.get("run", {}).get("name", "")
    if saved_name != run_name:
        msg = (
            f"Cannot resume: run.name mismatch -- "
            f"config says '{run_name}' but checkpoint has '{saved_name}'"
        )
        result.errors.append(msg)
        logger.error(msg)
        return

    latest = checkpoint_dirs[-1]
    logger.info(f"[RESUME]   Found compatible checkpoint: {latest}")


# ======================================================================== #
#  RICH SUMMARY TABLE                                                       #
# ======================================================================== #

def print_summary(result: ValidationResult) -> str:
    """Print a Rich table of all validated fields and return the plain-text version."""
    table = Table(
        title="Configuration Validation Summary",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Key", style="dim", no_wrap=True, ratio=3)
    table.add_column("Status", justify="center", ratio=1)
    table.add_column("Value", ratio=3)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Configuration Validation Summary")
    lines.append("=" * 80)

    for key, status, value in result.info:
        val_str = repr(value) if not isinstance(value, str) else value
        if status == "OK":
            style = "green"
        elif status == "DEFAULT":
            style = "yellow"
        else:
            style = "red"
        table.add_row(key, Text(status, style=style), str(val_str))
        lines.append(f"  {key:50s}  {status:10s}  {val_str}")

    console.print()
    console.print(table)

    if result.warnings:
        console.print(
            Panel(
                "\n".join(result.warnings),
                title="Warnings",
                border_style="yellow",
            )
        )
        lines.append("")
        lines.append("WARNINGS:")
        lines.extend(f"  {w}" for w in result.warnings)

    if result.errors:
        console.print(
            Panel(
                "\n".join(result.errors),
                title="Errors",
                border_style="red",
            )
        )
        lines.append("")
        lines.append("ERRORS:")
        lines.extend(f"  {e}" for e in result.errors)

    if result.ok:
        console.print(
            Panel(
                f"[bold green]Validation passed[/] -- "
                f"{len(result.info)} fields checked, "
                f"{len(result.warnings)} warnings, 0 errors",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]Validation FAILED[/] -- "
                f"{len(result.errors)} error(s)",
                border_style="red",
            )
        )

    lines.append("")
    lines.append(
        f"Total: {len(result.info)} fields | "
        f"{len(result.warnings)} warnings | {len(result.errors)} errors"
    )
    return "\n".join(lines)


# ======================================================================== #
#  MAIN COMMAND                                                             #
# ======================================================================== #

@app.command()
def main(
    config: Path = typer.Argument(
        ...,
        help="Path to the YAML configuration file.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Validate the config and exit without training.",
    ),
    output_folder: Optional[Path] = typer.Option(
        None,
        "--output-folder",
        help=(
            "Root output folder for checkpoints and logs. "
            "Must be empty unless --resume is passed."
        ),
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help=(
            "Resume from a previous run.  The output-folder must contain "
            "checkpoints whose config.yaml run.name matches the current config."
        ),
    ),
) -> None:
    """Validate configuration and optionally launch a GLiNER training run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ---- Load YAML ----
    logger.info(f"Loading config from {config}")
    with open(config) as f:
        cfg: dict = yaml.safe_load(f) or {}

    # ---- Determine log path ----
    if output_folder is not None:
        log_path = output_folder / f"validation_{timestamp}.log"
    else:
        log_path = config.parent / f"validation_{timestamp}.log"
    _attach_file_handler(log_path)

    logger.info(f"Log file: {log_path}")
    logger.info(f"Timestamp: {timestamp}")

    # ---- Schema validation ----
    result = validate_config(cfg)

    # ---- Semantic checks ----
    semantic_checks(cfg, result)

    # ---- API connectivity ----
    check_huggingface(cfg, result)
    check_wandb(cfg, result)

    # ---- Output folder checks ----
    if output_folder is not None and not validate:
        output_folder.mkdir(parents=True, exist_ok=True)

        if resume:
            check_resume(cfg, output_folder, result)
        else:
            # Must be empty
            existing = list(output_folder.iterdir())
            # Allow the log file we just created
            non_log = [
                p for p in existing
                if not p.name.startswith("validation_")
            ]
            if non_log:
                msg = (
                    f"--output-folder '{output_folder}' is not empty "
                    f"(contains {len(non_log)} item(s)).  "
                    f"Use --resume if you want to continue a previous run."
                )
                result.errors.append(msg)
                logger.error(msg)

    # ---- Print & save summary ----
    summary_text = print_summary(result)
    logger.info("Summary written to log file")

    # Also write a dedicated summary file
    if output_folder is not None:
        summary_path = output_folder / f"summary_{timestamp}.txt"
    else:
        summary_path = config.parent / f"summary_{timestamp}.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info(f"Summary saved to {summary_path}")

    # ---- Bail on errors or validate-only ----
    if not result.ok:
        logger.error(f"Validation failed with {len(result.errors)} error(s). Aborting.")
        raise typer.Exit(code=1)

    if validate:
        logger.info("--validate flag set.  Exiting without training.")
        raise typer.Exit(code=0)

    # ---- Require --output-folder for training ----
    if output_folder is None:
        logger.error("--output-folder is required to start training (or use --validate).")
        raise typer.Exit(code=1)

    # ---- Save resolved config ----
    resolved_path = output_folder / "config.yaml"
    with open(resolved_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Resolved config saved to {resolved_path}")

    # ---- Launch training ----
    _launch_training(cfg, output_folder, resume=resume)


# ======================================================================== #
#  TRAINING LAUNCHER                                                        #
# ======================================================================== #

def _launch_training(cfg: dict, output_folder: Path, resume: bool) -> None:
    """Build the GLiNER model and start training."""
    logger.info("Preparing training run ...")

    # Lazy-import heavy dependencies so validation stays fast
    import json
    import torch
    from gliner import GLiNER
    from gliner.training import TrainingArguments, Trainer

    # -- Seed --
    seed = cfg["run"]["seed"]
    torch.manual_seed(seed)

    # -- CUDA --
    cuda_devs = cfg["environment"].get("cuda_visible_devices")
    if cuda_devs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devs)

    # -- WandB env vars --
    report_to = cfg["environment"]["report_to"]
    if report_to in ("wandb", "all"):
        key = cfg["environment"].get("wandb_api_key") or os.environ.get("WANDB_API_KEY", "")
        if key:
            os.environ["WANDB_API_KEY"] = key
        proj = cfg["environment"].get("wandb_project")
        if proj:
            os.environ["WANDB_PROJECT"] = proj
        entity = cfg["environment"].get("wandb_entity")
        if entity:
            os.environ["WANDB_ENTITY"] = entity

    # -- Build model --
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    prev_path = train_cfg.get("prev_path")
    if prev_path and str(prev_path).lower() not in ("none", "null", ""):
        logger.info(f"Loading pretrained model from: {prev_path}")
        model = GLiNER.from_pretrained(prev_path)
    else:
        logger.info("Initialising model from config ...")
        model = GLiNER.from_config(model_cfg)

    model = model.to(dtype=torch.float32)
    logger.info(f"Model class: {model.__class__.__name__}")

    # -- LoRA --
    if cfg.get("lora", {}).get("enabled", False):
        _apply_lora(model, cfg["lora"])

    # -- Load data --
    train_data_path = cfg["data"]["train_data"]
    logger.info(f"Loading training data from {train_data_path}")
    with open(train_data_path) as f:
        train_dataset = json.load(f)
    logger.info(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    val_path = cfg["data"].get("val_data_dir", "none")
    if val_path and val_path.lower() not in ("none", "null", ""):
        logger.info(f"Loading validation data from {val_path}")
        with open(val_path) as f:
            eval_dataset = json.load(f)
        logger.info(f"Validation samples: {len(eval_dataset)}")

    # -- Freeze components --
    freeze = train_cfg.get("freeze_components")
    if freeze:
        logger.info(f"Freezing: {freeze}")

    # -- Eval batch size fallback --
    eval_bs = train_cfg.get("eval_batch_size") or train_cfg["train_batch_size"]

    # -- Logging steps fallback --
    log_steps = train_cfg.get("logging_steps") or train_cfg["eval_every"]

    # -- Train --
    logger.info("Starting training ...")
    model.train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_folder),
        freeze_components=freeze,
        compile_model=train_cfg.get("compile_model", False),
        # Schedule
        max_steps=train_cfg["num_steps"],
        lr_scheduler_type=train_cfg["scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        # Batch
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=eval_bs,
        # Optimisation
        learning_rate=float(train_cfg["lr_encoder"]),
        others_lr=float(train_cfg["lr_others"]),
        weight_decay=float(train_cfg["weight_decay_encoder"]),
        others_weight_decay=float(train_cfg["weight_decay_other"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        optim=train_cfg.get("optimizer", "adamw_torch"),
        # Loss
        focal_loss_alpha=float(train_cfg["loss_alpha"]),
        focal_loss_gamma=float(train_cfg["loss_gamma"]),
        focal_loss_prob_margin=float(train_cfg.get("loss_prob_margin", 0)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0)),
        loss_reduction=train_cfg["loss_reduction"],
        negatives=float(train_cfg["negatives"]),
        masking=train_cfg["masking"],
        # Logging & checkpoints
        save_steps=train_cfg["eval_every"],
        logging_steps=log_steps,
        save_total_limit=train_cfg["save_total_limit"],
        # Precision
        bf16=train_cfg.get("bf16", False),
        # Hardware
        use_cpu=train_cfg.get("use_cpu", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        # Reporting
        report_to=report_to,
        # Gradient accumulation
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
    )

    logger.info(f"Training complete.  Checkpoints in {output_folder}")


def _apply_lora(model: Any, lora_cfg: dict) -> None:
    """Wrap the model backbone with a LoRA adapter via peft."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.error("peft is not installed.  Install with: pip install peft")
        raise typer.Exit(code=1)

    task_map = {
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    }
    task_type = task_map.get(lora_cfg.get("task_type", "TOKEN_CLS"), TaskType.TOKEN_CLS)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type=task_type,
        modules_to_save=lora_cfg.get("modules_to_save"),
    )

    # Apply to the backbone encoder inside the model
    if hasattr(model, "model") and hasattr(model.model, "token_rep_layer"):
        model.model.token_rep_layer = get_peft_model(
            model.model.token_rep_layer, peft_config
        )
        logger.info(
            f"[LORA]     Applied LoRA (r={lora_cfg['r']}, "
            f"alpha={lora_cfg['lora_alpha']}) to token_rep_layer"
        )
    else:
        logger.warning("[LORA]     Could not locate token_rep_layer; LoRA not applied")


# ======================================================================== #
#  ENTRYPOINT                                                               #
# ======================================================================== #

if __name__ == "__main__":
    app()
