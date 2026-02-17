"""GLiNER fine-tuning configuration validator and CLI.

Usage as CLI (via Typer):
    python -m ptbr.config_cli --file config.yaml --validate \\
        --full-or-lora full --method span

Usage as module:
    from ptbr.config_cli import load_and_validate_config
    result = load_and_validate_config("config.yaml", full_or_lora="full", method="span")
    gliner_cfg = result.gliner_config   # ready for GLiNER.from_config(...)
    lora_cfg   = result.lora_config     # dict or None
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gliner.config import (
    GLiNERConfig,
    BiEncoderSpanConfig,
    BiEncoderTokenConfig,
    UniEncoderSpanConfig,
    UniEncoderTokenConfig,
    UniEncoderSpanDecoderConfig,
    UniEncoderTokenDecoderConfig,
    UniEncoderSpanRelexConfig,
    UniEncoderTokenRelexConfig,
)

logger = logging.getLogger(__name__)
console = Console()

# ============================================================================
# Validation Rules
# ============================================================================
# Each rule is a tuple: (key, type, required, default, constraint)
#   constraint is one of:
#     - ("range", min, max)
#     - ("literals", {set of allowed values})
#     - None  (no extra constraint beyond type)
# ============================================================================

_GLINER_RULES: List[Tuple[str, type, bool, Any, Any]] = [
    # 1.1 Backbone / Encoder
    ("model_name",          str,   True,  None,          None),
    ("name",                str,   False, "gliner",      None),
    ("fine_tune",           bool,  False, True,          ("literals", {True, False})),
    # 1.2 Architecture
    ("span_mode",           str,   False, "markerV0",    ("literals", {
        "markerV0", "markerV1", "marker", "query", "mlp", "cat",
        "conv_conv", "conv_max", "conv_mean", "conv_sum", "conv_share",
        "token_level",
    })),
    ("max_width",           int,   False, 12,            ("range", 1, 128)),
    # 1.3 BiEncoder
    ("labels_encoder",      str,   False, None,          None),
    # 1.4 Decoder
    ("labels_decoder",      str,   False, None,          None),
    ("decoder_mode",        str,   False, "span",        ("literals", {"span", "prompt"})),
    ("full_decoder_context", bool, False, True,          ("literals", {True, False})),
    ("blank_entity_prob",   float, False, 0.1,           ("range", 0.0, 1.0)),
    ("decoder_loss_coef",   float, False, 0.5,           ("range", 0.0, 10.0)),
    # 1.5 Relex
    ("relations_layer",     str,   False, None,          None),
    ("triples_layer",       str,   False, None,          None),
    ("embed_rel_token",     bool,  False, True,          ("literals", {True, False})),
    ("rel_token_index",     int,   False, -1,            ("range", -1, 100000)),
    ("rel_token",           str,   False, "<<REL>>",     None),
    ("adjacency_loss_coef", float, False, 1.0,           ("range", 0.0, 10.0)),
    ("relation_loss_coef",  float, False, 1.0,           ("range", 0.0, 10.0)),
    # 1.6 Hidden dims
    ("hidden_size",         int,   False, 512,           ("range", 64, 4096)),
    ("dropout",             float, False, 0.4,           ("range", 0.0, 0.9)),
    # 1.7 Subtoken
    ("subtoken_pooling",    str,   False, "first",       ("literals", {"first", "mean", "max"})),
    ("words_splitter_type", str,   False, "whitespace",  ("literals", {
        "whitespace", "spacy", "stanza", "mecab", "jieba", "janome", "camel",
    })),
    # 1.8 Sequence limits
    ("max_len",             int,   False, 384,           ("range", 32, 8192)),
    ("max_types",           int,   False, 25,            ("range", 1, 1000)),
    ("max_neg_type_ratio",  int,   False, 1,             ("range", 0, 100)),
    # 1.9 Post-fusion & layers
    ("post_fusion_schema",  str,   False, "",            None),
    ("num_post_fusion_layers", int, False, 1,            ("range", 1, 12)),
    ("fuse_layers",         bool,  False, False,         ("literals", {True, False})),
    ("num_rnn_layers",      int,   False, 1,             ("range", 0, 4)),
    # 1.10 Special tokens
    ("embed_ent_token",     bool,  False, True,          ("literals", {True, False})),
    ("class_token_index",   int,   False, -1,            ("range", -1, 100000)),
    ("vocab_size",          int,   False, -1,            ("range", -1, 1000000)),
    ("ent_token",           str,   False, "<<ENT>>",     None),
    ("sep_token",           str,   False, "<<SEP>>",     None),
    # 1.11 Loss coefficients
    ("token_loss_coef",     float, False, 1.0,           ("range", 0.0, 10.0)),
    ("span_loss_coef",      float, False, 1.0,           ("range", 0.0, 10.0)),
    ("represent_spans",     bool,  False, False,         ("literals", {True, False})),
    ("neg_spans_ratio",     float, False, 1.0,           ("range", 0.0, 10.0)),
    # 1.12 Attention
    ("_attn_implementation", str,  False, None,          ("literals", {None, "eager", "sdpa", "flash_attention_2"})),
]

_LORA_RULES: List[Tuple[str, type, bool, Any, Any]] = [
    ("r",                  int,   False, 8,                ("range", 1, 256)),
    ("lora_alpha",         int,   False, 16,               ("range", 1, 512)),
    ("lora_dropout",       float, False, 0.1,              ("range", 0.0, 0.9)),
    ("target_modules",     list,  False, ["query_proj", "value_proj"], None),
    ("bias",               str,   False, "none",           ("literals", {"none", "all", "lora_only"})),
    ("task_type",          str,   False, "FEATURE_EXTRACTION", ("literals", {
        "FEATURE_EXTRACTION", "TOKEN_CLS", "SEQ_CLS", "CAUSAL_LM", "SEQ_2_SEQ_LM",
    })),
    ("modules_to_save",    list,  False, None,             None),
    ("fan_in_fan_out",     bool,  False, False,            ("literals", {True, False})),
    ("use_rslora",         bool,  False, False,            ("literals", {True, False})),
    ("init_lora_weights",  bool,  False, True,             ("literals", {True, False})),
]


# Method -> config class mapping
_METHOD_CONFIG_CLASS = {
    ("span",      False, False): UniEncoderSpanConfig,
    ("token",     False, False): UniEncoderTokenConfig,
    ("biencoder", True,  False): BiEncoderSpanConfig,     # span variant
    ("biencoder_token", True, False): BiEncoderTokenConfig,
    ("decoder",   False, True):  UniEncoderSpanDecoderConfig,
    ("decoder_token", False, True): UniEncoderTokenDecoderConfig,
    ("relex",     False, False): UniEncoderSpanRelexConfig,
    ("relex_token", False, False): UniEncoderTokenRelexConfig,
}


# ============================================================================
# Validation Issue Tracking
# ============================================================================

@dataclass
class ValidationIssue:
    """A single validation finding."""
    level: str          # "ERROR" or "WARNING"
    field: str          # dotted path: "gliner_config.model_name"
    message: str


@dataclass
class ValidationReport:
    """Collection of all issues found during validation."""
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "WARNING"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, field: str, message: str) -> None:
        self.issues.append(ValidationIssue("ERROR", field, message))

    def add_warning(self, field: str, message: str) -> None:
        self.issues.append(ValidationIssue("WARNING", field, message))


# ============================================================================
# Result Object
# ============================================================================

@dataclass
class GLiNERConfigResult:
    """Validated configuration result returned when used as a module.

    Attributes:
        gliner_config: A GLiNERConfig (or subclass) instance ready for the trainer.
        lora_config: A dict of LoRA parameters (or None when full fine-tuning).
        raw_yaml: The original parsed YAML dictionary.
        report: The ValidationReport with all warnings/errors.
        full_or_lora: Whether this is a "full" or "lora" configuration.
        method: The resolved method string.
    """
    gliner_config: GLiNERConfig
    lora_config: Optional[Dict[str, Any]]
    raw_yaml: Dict[str, Any]
    report: ValidationReport
    full_or_lora: str
    method: str


# ============================================================================
# Core Validation Logic
# ============================================================================

def _coerce_type(value: Any, expected_type: type) -> Any:
    """Try to coerce a value to the expected type."""
    if value is None:
        return None
    if expected_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ("true", "yes", "1"):
                return True
            if value.lower() in ("false", "no", "0"):
                return False
        raise TypeError(f"Cannot coerce {value!r} to bool")
    if expected_type is int:
        if isinstance(value, bool):
            raise TypeError(f"Cannot coerce bool {value!r} to int")
        return int(value)
    if expected_type is float:
        if isinstance(value, bool):
            raise TypeError(f"Cannot coerce bool {value!r} to float")
        return float(value)
    if expected_type is str:
        return str(value)
    if expected_type is list:
        if isinstance(value, list):
            return value
        raise TypeError(f"Expected list, got {type(value).__name__}")
    return value


def _validate_section(
    section_data: Dict[str, Any],
    rules: List[Tuple[str, type, bool, Any, Any]],
    section_prefix: str,
    report: ValidationReport,
) -> Dict[str, Any]:
    """Validate a section of the config against its rules.

    Returns a cleaned dict with validated/defaulted values.
    """
    result: Dict[str, Any] = {}

    for key, expected_type, required, default, constraint in rules:
        field_path = f"{section_prefix}.{key}"
        raw_value = section_data.get(key)

        # --- Missing / None handling ---
        if raw_value is None:
            if required:
                report.add_error(field_path, f"REQUIRED field is missing or null.")
                continue
            else:
                if default is not None:
                    report.add_warning(
                        field_path,
                        f"Not set; using default: {default!r}",
                    )
                result[key] = default
                continue

        # --- Type coercion ---
        try:
            value = _coerce_type(raw_value, expected_type)
        except (TypeError, ValueError) as exc:
            report.add_error(
                field_path,
                f"Type error: expected {expected_type.__name__}, got {type(raw_value).__name__} "
                f"({raw_value!r}). {exc}",
            )
            continue

        # --- Constraint checking ---
        if constraint is not None:
            kind = constraint[0]
            if kind == "range":
                _, lo, hi = constraint
                if not (lo <= value <= hi):
                    report.add_error(
                        field_path,
                        f"Value {value!r} outside allowed range [{lo}, {hi}].",
                    )
                    continue
            elif kind == "literals":
                allowed = constraint[1]
                if value not in allowed:
                    report.add_error(
                        field_path,
                        f"Value {value!r} not in allowed set: {sorted(str(v) for v in allowed)}.",
                    )
                    continue

        result[key] = value

    # --- Warn about unknown keys ---
    known_keys = {r[0] for r in rules}
    for key in section_data:
        if key not in known_keys:
            report.add_warning(
                f"{section_prefix}.{key}",
                f"Unknown field (will be ignored): {key!r}",
            )

    return result


def _validate_cross_constraints(
    gliner_data: Dict[str, Any],
    method: str,
    full_or_lora: str,
    report: ValidationReport,
) -> None:
    """Check cross-field constraints that depend on method and mode."""

    # --- Method-specific requirements ---
    if method == "biencoder":
        if not gliner_data.get("labels_encoder"):
            report.add_error(
                "gliner_config.labels_encoder",
                "BiEncoder method requires labels_encoder to be set to a model name.",
            )
    elif method == "decoder":
        if not gliner_data.get("labels_decoder"):
            report.add_error(
                "gliner_config.labels_decoder",
                "Decoder method requires labels_decoder to be set to a model name.",
            )
    elif method == "relex":
        if not gliner_data.get("relations_layer"):
            report.add_error(
                "gliner_config.relations_layer",
                "Relex method requires relations_layer to be set.",
            )

    # --- span_mode vs method consistency ---
    span_mode = gliner_data.get("span_mode", "markerV0")
    if method == "token" and span_mode != "token_level":
        report.add_warning(
            "gliner_config.span_mode",
            f"Method is 'token' but span_mode is {span_mode!r}; forcing to 'token_level'.",
        )
        gliner_data["span_mode"] = "token_level"
    elif method in ("span", "biencoder", "decoder", "relex") and span_mode == "token_level":
        report.add_warning(
            "gliner_config.span_mode",
            f"Method is {method!r} but span_mode is 'token_level'; this may be intentional "
            f"for a token-level variant. If not, change span_mode.",
        )

    # --- decoder fields when no decoder ---
    if method != "decoder":
        for fld in ("decoder_mode", "full_decoder_context", "blank_entity_prob", "decoder_loss_coef"):
            if gliner_data.get(fld) is not None and fld in ("decoder_mode",) and gliner_data.get("labels_decoder") is None:
                pass  # defaults are fine

    # --- relex fields when no relex ---
    if method != "relex":
        for fld in ("relations_layer", "triples_layer"):
            if gliner_data.get(fld) is not None:
                report.add_warning(
                    f"gliner_config.{fld}",
                    f"Field {fld!r} is set but method is not 'relex'; it will be ignored.",
                )


def _build_gliner_config(
    gliner_data: Dict[str, Any],
    method: str,
) -> GLiNERConfig:
    """Build the appropriate GLiNERConfig subclass from validated data."""
    # Filter out keys not accepted by BaseGLiNERConfig or the subclass
    # Use GLiNERConfig which auto-detects model_type
    cfg_kwargs = {}
    for key, value in gliner_data.items():
        if value is not None or key in (
            "labels_encoder", "labels_decoder", "relations_layer",
            "triples_layer", "_attn_implementation", "post_fusion_schema",
            "modules_to_save",
        ):
            cfg_kwargs[key] = value

    # Remove relex-specific fields from non-relex builds to avoid __init__ errors
    base_fields = {
        "model_name", "name", "max_width", "hidden_size", "dropout", "fine_tune",
        "subtoken_pooling", "span_mode", "post_fusion_schema", "num_post_fusion_layers",
        "vocab_size", "max_neg_type_ratio", "max_types", "max_len",
        "words_splitter_type", "num_rnn_layers", "fuse_layers", "embed_ent_token",
        "class_token_index", "encoder_config", "ent_token", "sep_token",
        "_attn_implementation", "token_loss_coef", "span_loss_coef",
        "represent_spans", "neg_spans_ratio",
    }

    biencoder_fields = base_fields | {"labels_encoder", "labels_encoder_config"}

    decoder_fields = base_fields | {
        "labels_decoder", "decoder_mode", "full_decoder_context",
        "blank_entity_prob", "labels_decoder_config", "decoder_loss_coef",
    }

    relex_fields = base_fields | {
        "relations_layer", "triples_layer", "embed_rel_token",
        "rel_token_index", "rel_token", "adjacency_loss_coef", "relation_loss_coef",
    }

    # Legacy GLiNERConfig accepts a broad set
    gliner_legacy_fields = base_fields | {
        "labels_encoder", "labels_decoder", "relations_layer",
    }

    # Use the legacy GLiNERConfig which auto-detects the right model_type
    filtered = {k: v for k, v in cfg_kwargs.items() if k in gliner_legacy_fields}
    return GLiNERConfig(**filtered)


# ============================================================================
# Public API: load_and_validate_config
# ============================================================================

def load_and_validate_config(
    file: str | Path,
    full_or_lora: Literal["full", "lora"] = "full",
    method: Literal["biencoder", "decoder", "relex", "span", "token"] = "span",
    validate: bool = True,
) -> GLiNERConfigResult:
    """Load a YAML config file, validate every field, and return a config object.

    Args:
        file: Path to the YAML configuration file.
        full_or_lora: Whether to do full fine-tuning or LoRA.
        method: Architecture method to use.
        validate: Whether to perform validation (always True in practice).

    Returns:
        GLiNERConfigResult with the validated config, lora dict, and report.

    Raises:
        SystemExit: When used as CLI and validation fails.
        ValueError: When used as module and validation fails.
    """
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"Config file not found: {file}")

    with open(file) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

    report = ValidationReport()

    # --- Validate gliner_config section ---
    gliner_section = raw.get("gliner_config", {})
    if not gliner_section:
        report.add_error("gliner_config", "Missing 'gliner_config' section in YAML file.")
        gliner_section = {}

    validated_gliner = _validate_section(
        gliner_section, _GLINER_RULES, "gliner_config", report,
    )

    # --- Cross-field validation ---
    _validate_cross_constraints(validated_gliner, method, full_or_lora, report)

    # --- Validate lora_config section (only for lora mode) ---
    validated_lora: Optional[Dict[str, Any]] = None
    if full_or_lora == "lora":
        lora_section = raw.get("lora_config", {})
        if not lora_section:
            report.add_warning("lora_config", "LoRA mode selected but 'lora_config' section is missing; using all defaults.")
            lora_section = {}
        validated_lora = _validate_section(
            lora_section, _LORA_RULES, "lora_config", report,
        )

    # --- Build GLiNERConfig object ---
    gliner_config = None
    if report.is_valid:
        gliner_config = _build_gliner_config(validated_gliner, method)

    result = GLiNERConfigResult(
        gliner_config=gliner_config,
        lora_config=validated_lora,
        raw_yaml=raw,
        report=report,
        full_or_lora=full_or_lora,
        method=method,
    )

    return result


# ============================================================================
# Rich Output: Summary Printing
# ============================================================================

def _make_gliner_table(validated: Dict[str, Any], report: ValidationReport) -> Table:
    """Build a rich table showing each gliner_config field and its status."""
    table = Table(title="GLiNER Configuration", show_lines=True)
    table.add_column("Field", style="cyan", min_width=25)
    table.add_column("Value", style="white", min_width=30)
    table.add_column("Status", style="white", min_width=20)

    warning_fields = {i.field.split(".")[-1] for i in report.warnings}
    error_fields = {i.field.split(".")[-1] for i in report.errors}

    for key, _type, required, default, _constraint in _GLINER_RULES:
        value = validated.get(key, "[missing]")
        if key in error_fields:
            status = "[bold red]ERROR[/]"
        elif key in warning_fields:
            status = "[yellow]DEFAULT[/]"
        elif required:
            status = "[green]SET (required)[/]"
        else:
            status = "[green]SET[/]"
        table.add_row(key, repr(value), status)

    return table


def _make_lora_table(validated: Dict[str, Any], report: ValidationReport) -> Table:
    """Build a rich table showing each lora_config field and its status."""
    table = Table(title="LoRA Configuration", show_lines=True)
    table.add_column("Field", style="cyan", min_width=25)
    table.add_column("Value", style="white", min_width=30)
    table.add_column("Status", style="white", min_width=20)

    warning_fields = {i.field.split(".")[-1] for i in report.warnings}
    error_fields = {i.field.split(".")[-1] for i in report.errors}

    for key, _type, required, default, _constraint in _LORA_RULES:
        value = validated.get(key, "[missing]")
        if key in error_fields:
            status = "[bold red]ERROR[/]"
        elif key in warning_fields:
            status = "[yellow]DEFAULT[/]"
        else:
            status = "[green]SET[/]"
        table.add_row(key, repr(value), status)

    return table


def _print_issues(report: ValidationReport) -> None:
    """Print errors and warnings using rich."""
    if report.errors:
        console.print(Panel("[bold red]VALIDATION ERRORS[/]", style="red"))
        for issue in report.errors:
            console.print(f"  [bold red]ERROR[/]  {issue.field}: {issue.message}")
        console.print()

    if report.warnings:
        console.print(Panel("[bold yellow]WARNINGS (defaults applied)[/]", style="yellow"))
        for issue in report.warnings:
            console.print(f"  [yellow]WARN[/]   {issue.field}: {issue.message}")
        console.print()


def _save_validation_log(
    result: GLiNERConfigResult,
    log_path: Path,
) -> None:
    """Save a structured JSON log of the validation run."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "file": str(result.raw_yaml.get("_source_file", "unknown")),
        "full_or_lora": result.full_or_lora,
        "method": result.method,
        "valid": result.report.is_valid,
        "error_count": len(result.report.errors),
        "warning_count": len(result.report.warnings),
        "issues": [
            {"level": i.level, "field": i.field, "message": i.message}
            for i in result.report.issues
        ],
    }
    # Add resolved config if valid
    if result.gliner_config is not None:
        log_data["resolved_gliner_config"] = result.gliner_config.to_dict()
    if result.lora_config is not None:
        log_data["resolved_lora_config"] = result.lora_config

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)


def print_and_log_result(
    result: GLiNERConfigResult,
    file_path: Path,
    log_dir: Optional[Path] = None,
) -> Path:
    """Print the rich summary to the terminal and save a log file.

    Args:
        result: The validated config result.
        file_path: The original config file path (for display).
        log_dir: Directory for the log file. Defaults to file_path.parent.

    Returns:
        Path to the saved log file.
    """
    console.print()
    console.print(Panel(
        f"[bold]Config Validation Report[/]\n"
        f"File: {file_path}\n"
        f"Mode: [cyan]{result.full_or_lora}[/]  |  Method: [cyan]{result.method}[/]",
        title="GLiNER ptbr Config",
        style="blue",
    ))
    console.print()

    # Re-validate to get section data for tables
    raw = result.raw_yaml
    gliner_section = raw.get("gliner_config", {})
    validated_gliner = {}
    for key, _type, _req, _def, _con in _GLINER_RULES:
        val = gliner_section.get(key)
        if val is None:
            val = _def
        validated_gliner[key] = val

    console.print(_make_gliner_table(validated_gliner, result.report))
    console.print()

    if result.full_or_lora == "lora" and result.lora_config is not None:
        console.print(_make_lora_table(result.lora_config, result.report))
        console.print()

    _print_issues(result.report)

    if result.report.is_valid:
        console.print("[bold green]Validation PASSED[/]")
    else:
        console.print("[bold red]Validation FAILED[/]")

    # Save log
    if log_dir is None:
        log_dir = file_path.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"config_validation_{timestamp}.json"
    result.raw_yaml["_source_file"] = str(file_path)
    _save_validation_log(result, log_path)
    console.print(f"\nLog saved to: [cyan]{log_path}[/]")
    console.print()

    return log_path


# ============================================================================
# Typer CLI Application
# ============================================================================

def _build_app():
    """Build the Typer app. Imported lazily so the module works without typer
    when used purely as a library (typer is only needed for CLI execution)."""
    import typer

    app = typer.Typer(
        name="ptbr-config",
        help="Validate and inspect GLiNER fine-tuning YAML configurations.",
        add_completion=False,
    )

    @app.command()
    def main(
        file: Path = typer.Option(
            ...,
            "--file",
            "-f",
            help="Path to the YAML configuration file.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
        validate: bool = typer.Option(
            False,
            "--validate",
            "-v",
            help="Run validation and print a rich summary.",
        ),
        full_or_lora: str = typer.Option(
            "full",
            "--full-or-lora",
            help="Fine-tuning mode: 'full' for full fine-tuning, 'lora' for LoRA adapters.",
        ),
        method: str = typer.Option(
            "span",
            "--method",
            help="Architecture method: biencoder, decoder, relex, span, or token.",
        ),
    ):
        """Validate a GLiNER YAML config and print a rich diagnostic report."""
        # Validate enum-like inputs
        if full_or_lora not in ("full", "lora"):
            console.print(f"[bold red]Error:[/] --full-or-lora must be 'full' or 'lora', got {full_or_lora!r}")
            raise typer.Exit(code=1)
        if method not in ("biencoder", "decoder", "relex", "span", "token"):
            console.print(
                f"[bold red]Error:[/] --method must be one of: biencoder, decoder, relex, span, token. "
                f"Got {method!r}"
            )
            raise typer.Exit(code=1)

        result = load_and_validate_config(
            file=file,
            full_or_lora=full_or_lora,
            method=method,
            validate=validate,
        )

        if validate:
            print_and_log_result(result, file)

        if not result.report.is_valid:
            raise typer.Exit(code=1)

        # Even without --validate, always print at least a brief status
        if not validate:
            if result.report.warnings:
                console.print(f"[yellow]Config loaded with {len(result.report.warnings)} warning(s).[/]")
            else:
                console.print("[green]Config loaded successfully.[/]")

    return app


def cli_main():
    """Entry point for `python -m ptbr.config_cli`."""
    app = _build_app()
    app()


if __name__ == "__main__":
    cli_main()
