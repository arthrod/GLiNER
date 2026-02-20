r"""GLiNER fine-tuning configuration validator and CLI.

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

import copy
import json
import logging
from typing import Any, Dict, List, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import field, dataclass

import yaml
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from gliner.config import GLiNERConfig

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
        gliner_config: A GLiNERConfig instance ready for the trainer, or None when invalid.
        validated_gliner: The fully validated/coerced gliner_config dictionary.
        lora_config: A dict of LoRA parameters (or None when full fine-tuning).
        raw_yaml: The original parsed YAML dictionary.
        report: The ValidationReport with all warnings/errors.
        full_or_lora: Whether this is a "full" or "lora" configuration.
        method: The resolved method string.
    """
    gliner_config: Optional[GLiNERConfig]
    validated_gliner: Dict[str, Any]
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
                report.add_error(field_path, "REQUIRED field is missing or null.")
                continue
            else:
                if default is not None:
                    report.add_warning(
                        field_path,
                        f"Not set; using default: {default!r}",
                    )
                result[key] = copy.deepcopy(default)
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
    raw_gliner_section: Optional[Dict[str, Any]] = None,
) -> None:
    """Check cross-field constraints that depend on method and mode."""
    if isinstance(raw_gliner_section, dict):
        explicit_keys = set(raw_gliner_section.keys())
    else:
        explicit_keys = {k for k, v in gliner_data.items() if v is not None}

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

    # --- decoder fields when method is not decoder ---
    if method != "decoder":
        if "labels_decoder" in explicit_keys and gliner_data.get("labels_decoder") is not None:
            report.add_warning(
                "gliner_config.labels_decoder",
                "Field 'labels_decoder' is set while method is not 'decoder'; "
                "decoder architecture may still be selected.",
            )
        if gliner_data.get("labels_decoder") is None:
            for fld in ("decoder_mode", "full_decoder_context", "blank_entity_prob", "decoder_loss_coef"):
                if fld in explicit_keys:
                    report.add_warning(
                        f"gliner_config.{fld}",
                        f"Field {fld!r} is set but labels_decoder is not set; it will be ignored.",
                    )

    # --- relex fields when method is not relex ---
    if method != "relex":
        if "relations_layer" in explicit_keys and gliner_data.get("relations_layer") is not None:
            report.add_warning(
                "gliner_config.relations_layer",
                "Field 'relations_layer' is set while method is not 'relex'; "
                "relex architecture may still be selected.",
            )
        if (
            "triples_layer" in explicit_keys
            and gliner_data.get("triples_layer") is not None
            and gliner_data.get("relations_layer") is None
        ):
            report.add_warning(
                "gliner_config.triples_layer",
                "Field 'triples_layer' is set but relations_layer is not set; it will be ignored.",
            )


def _build_gliner_config(
    gliner_data: Dict[str, Any],
) -> GLiNERConfig:
    """Build a GLiNERConfig from validated data."""
    return GLiNERConfig(**gliner_data)


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

    # --- Resolve + validate gliner_config section ---
    gliner_key = "gliner_config"
    if gliner_key in raw and "model" in raw:
        report.add_warning(
            "gliner_config",
            "Both 'gliner_config' and 'model' sections found; using 'gliner_config'.",
        )
    if gliner_key in raw:
        gliner_section = raw.get(gliner_key)
        selected_gliner_key = gliner_key
    elif "model" in raw:
        gliner_section = raw.get("model")
        selected_gliner_key = "model"
        report.add_warning(
            "gliner_config",
            "Using 'model' section as an alias for 'gliner_config'.",
        )
    else:
        report.add_error(
            "gliner_config",
            "Missing 'gliner_config' section in YAML file (or 'model' alias).",
        )
        gliner_section = {}
        selected_gliner_key = gliner_key

    if gliner_section is None:
        report.add_error(
            "gliner_config",
            f"'{selected_gliner_key}' must be a YAML mapping, got null.",
        )
        gliner_section = {}
    elif not isinstance(gliner_section, dict):
        report.add_error(
            "gliner_config",
            f"'{selected_gliner_key}' must be a YAML mapping, got {type(gliner_section).__name__}.",
        )
        gliner_section = {}

    validated_gliner = _validate_section(
        gliner_section, _GLINER_RULES, "gliner_config", report,
    )

    # --- Cross-field validation ---
    _validate_cross_constraints(
        validated_gliner,
        method,
        full_or_lora,
        report,
        raw_gliner_section=gliner_section,
    )

    # --- Resolve + validate lora_config section (only for lora mode) ---
    validated_lora: Optional[Dict[str, Any]] = None
    if full_or_lora == "lora":
        lora_key = "lora_config"
        if lora_key in raw and "lora" in raw:
            report.add_warning(
                "lora_config",
                "Both 'lora_config' and 'lora' sections found; using 'lora_config'.",
            )
        if lora_key in raw:
            lora_section = raw.get(lora_key)
            selected_lora_key = lora_key
        elif "lora" in raw:
            lora_section = raw.get("lora")
            selected_lora_key = "lora"
            report.add_warning(
                "lora_config",
                "Using 'lora' section as an alias for 'lora_config'.",
            )
        else:
            report.add_warning(
                "lora_config",
                "LoRA mode selected but neither 'lora_config' nor 'lora' section is present; using all defaults.",
            )
            lora_section = {}
            selected_lora_key = lora_key

        if lora_section is None:
            report.add_error(
                "lora_config",
                f"'{selected_lora_key}' must be a YAML mapping, got null.",
            )
            lora_section = {}
        elif not isinstance(lora_section, dict):
            report.add_error(
                "lora_config",
                f"'{selected_lora_key}' must be a YAML mapping, got {type(lora_section).__name__}.",
            )
            lora_section = {}
        validated_lora = _validate_section(
            lora_section, _LORA_RULES, "lora_config", report,
        )

    # --- Build GLiNERConfig object ---
    gliner_config = None
    if report.is_valid:
        gliner_config = _build_gliner_config(validated_gliner)

    result = GLiNERConfigResult(
        gliner_config=gliner_config,
        validated_gliner=validated_gliner,
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

def _is_default_warning(issue: ValidationIssue) -> bool:
    """Return True when a warning indicates a default value was applied."""
    return "using default" in issue.message.lower()


def _make_gliner_table(validated: Dict[str, Any], report: ValidationReport) -> Table:
    """Build a rich table showing each gliner_config field and its status."""
    table = Table(title="GLiNER Configuration", show_lines=True)
    table.add_column("Field", style="cyan", min_width=25)
    table.add_column("Value", style="white", min_width=30)
    table.add_column("Status", style="white", min_width=20)

    for key, _type, required, default, _constraint in _GLINER_RULES:
        full_field = f"gliner_config.{key}"
        field_errors = [i for i in report.errors if i.field == full_field]
        field_warnings = [i for i in report.warnings if i.field == full_field]
        has_default_warning = any(_is_default_warning(i) for i in field_warnings)
        has_other_warning = any(not _is_default_warning(i) for i in field_warnings)
        value = validated.get(key, "[missing]")
        if field_errors:
            status = "[bold red]ERROR[/]"
        elif has_other_warning:
            status = "[yellow]WARNING[/]"
        elif has_default_warning:
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

    for key, _type, required, default, _constraint in _LORA_RULES:
        full_field = f"lora_config.{key}"
        field_errors = [i for i in report.errors if i.field == full_field]
        field_warnings = [i for i in report.warnings if i.field == full_field]
        has_default_warning = any(_is_default_warning(i) for i in field_warnings)
        has_other_warning = any(not _is_default_warning(i) for i in field_warnings)
        value = validated.get(key, "[missing]")
        if field_errors:
            status = "[bold red]ERROR[/]"
        elif has_other_warning:
            status = "[yellow]WARNING[/]"
        elif has_default_warning:
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
    source_file: Optional[Path] = None,
) -> None:
    """Save a structured JSON log of the validation run."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "file": str(source_file) if source_file is not None else "unknown",
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

    console.print(_make_gliner_table(result.validated_gliner, result.report))
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
    _save_validation_log(result, log_path, source_file=file_path)
    console.print(f"\nLog saved to: [cyan]{log_path}[/]")
    console.print()

    return log_path


# ============================================================================
# Typer CLI Application
# ============================================================================

def _build_app():
    """
    Constructs and returns a Typer CLI application for validating and inspecting GLiNER fine-tuning YAML configurations.
    
    The app exposes a `main` command that loads a YAML config file, runs validation (optionally printing a rich summary and saving a log), and exits with a nonzero code on validation failure.
    
    Returns:
        typer.Typer: A configured Typer application exposing the `main` command.
    """
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