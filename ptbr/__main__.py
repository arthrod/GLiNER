"""Central CLI entry point: python -m ptbr <command>

Commands:
    data    Load, validate, and prepare GLiNER datasets
    config  Validate a GLiNER YAML configuration file
    train   Validate config and launch a training run
"""

import json
from typing import Optional

import typer

app = typer.Typer(
    name="ptbr",
    help="GLiNER fine-tuning toolkit: data preparation, config validation, and training.",
    add_completion=False,
)


# ── data subcommand ──────────────────────────────────────────────────────

data_app = typer.Typer(help="Load, validate, and prepare GLiNER datasets.")
app.add_typer(data_app, name="data")


@data_app.callback(invoke_without_command=True)
def data_cmd(
    file_or_repo: str = typer.Option(..., help="Local JSON/JSONL file or HuggingFace dataset repo id."),
    text_column: str = typer.Option("tokenized_text", help="Source column name for tokenized text."),
    ner_column: str = typer.Option("ner", help="Source column name for NER annotations."),
    split: str = typer.Option("train", help="Dataset split (e.g. train, validation, test)."),
    validate: bool = typer.Option(False, help="Validate against GLiNER native format."),
    generate_label_embeddings: Optional[str] = typer.Option(
        None, help="Model name/path for bi-encoder label embeddings."
    ),
    trust_remote_code: bool = typer.Option(
        False, help="Allow custom code execution when loading remote models."
    ),
    output_embeddings_path: str = typer.Option(
        "label_embeddings.pt", help="Output path for label embeddings."
    ),
    output_labels_path: str = typer.Option(
        "labels.json", help="Output path for extracted labels JSON."
    ),
):
    """Load GLiNER data, validate, or generate label embeddings."""
    from ptbr.data import extract_labels, load_data, validate_data

    data = load_data(file_or_repo, text_column, ner_column, split=split)
    labels = extract_labels(data)

    typer.echo(f"Loaded {len(data)} examples from {file_or_repo}")
    typer.echo(f"Found {len(labels)} unique labels")

    if validate:
        is_valid, errors = validate_data(data)
        if is_valid:
            typer.echo("Validation passed.")
        else:
            typer.echo(f"Validation failed with {len(errors)} error(s):")
            for err in errors:
                typer.echo(f"  {err}")
            raise typer.Exit(code=1)

    if generate_label_embeddings:
        import torch

        from gliner import GLiNER

        typer.echo(f"Loading model: {generate_label_embeddings}")
        model = GLiNER.from_pretrained(
            generate_label_embeddings, trust_remote_code=trust_remote_code
        )

        typer.echo(f"Encoding {len(labels)} labels...")
        embeddings = model.encode_labels(labels)

        torch.save(embeddings, output_embeddings_path)
        with open(output_labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

        typer.echo(
            f"Saved {output_embeddings_path} ({tuple(embeddings.shape)}) "
            f"and {output_labels_path}"
        )


# ── config subcommand ────────────────────────────────────────────────────

config_app = typer.Typer(help="Validate GLiNER YAML configuration files.")
app.add_typer(config_app, name="config")


@config_app.callback(invoke_without_command=True)
def config_cmd(
    file: str = typer.Option(..., help="Path to YAML config file."),
    validate: bool = typer.Option(False, "--validate", help="Run full validation with rich report."),
    full_or_lora: str = typer.Option("full", help="Training mode: 'full' or 'lora'."),
    method: str = typer.Option("span", help="GLiNER method: span, token, biencoder, decoder, relex."),
):
    """Validate a GLiNER training configuration YAML file."""
    from ptbr.config_cli import load_and_validate_config, print_and_log_result

    result = load_and_validate_config(
        file, full_or_lora=full_or_lora, method=method, validate=validate,
    )

    if validate:
        print_and_log_result(result, file)

    if not result.report.is_valid:
        raise typer.Exit(code=1)


# ── train subcommand ─────────────────────────────────────────────────────
# training_cli.py already defines a full typer app with its own @app.command().
# We import and add it as a sub-app so `python -m ptbr train ...` works.

from ptbr.training_cli import app as _train_app

app.add_typer(_train_app, name="train")


if __name__ == "__main__":
    app()
