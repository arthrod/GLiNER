"""CLI entry point: python -m ptbr

Uses typer to expose --validate, --file-or-repo, --text-column,
--ner-column, and --generate-label-embeddings.
"""

import json
from typing import Optional

import typer

from ptbr import extract_labels, load_data, validate_data

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    file_or_repo: str = typer.Option(..., help="Local JSON file path or HuggingFace dataset repo id."),
    text_column: str = typer.Option("tokenized_text", help="Source column name for tokenized text."),
    ner_column: str = typer.Option("ner", help="Source column name for NER annotations."),
    validate: bool = typer.Option(False, help="Validate the dataset against GLiNER native format."),
    generate_label_embeddings: Optional[str] = typer.Option(
        None,
        help="Model name/path to generate and save label embeddings (bi-encoder models).",
    ),
):
    """Load GLiNER data from a file or HuggingFace repo, validate, or generate label embeddings."""
    data = load_data(file_or_repo, text_column, ner_column)
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
        model = GLiNER.from_pretrained(generate_label_embeddings)

        typer.echo(f"Encoding {len(labels)} labels...")
        embeddings = model.encode_labels(labels)

        torch.save(embeddings, "label_embeddings.pt")
        with open("labels.json", "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

        typer.echo(f"Saved label_embeddings.pt ({tuple(embeddings.shape)}) and labels.json")


if __name__ == "__main__":
    app()
