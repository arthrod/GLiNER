"""PT-BR data utilities for GLiNER.

Loads, validates, and prepares datasets in GLiNER's native format.
Compatible with all model variants: span, token, bi-encoder, decoder,
relation extraction (relex), and multitask pipelines.

Native format (both span-based and token-based models consume this):
    [
        {
            "tokenized_text": ["word1", "word2", ...],
            "ner": [[start, end, "Label"], ...]
        },
        ...
    ]

Optional fields for relation extraction models:
    "relations": [[head_entity_idx, tail_entity_idx, "rel_type"], ...]

When used as a module:
    from ptbr import prepare
    result = prepare("path/to/data.json")
    # result is a GLiNERData object with .data, .labels, .is_valid, etc.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GLiNERData:
    """Container for a GLiNER-compatible dataset."""

    data: List[Dict[str, Any]]
    labels: List[str] = field(default_factory=list)
    label_embeddings: Optional[Any] = None
    source: str = ""
    validation_errors: List[str] = field(default_factory=list)
    is_valid: bool = False


def load_data(
    file_or_repo: str,
    text_column: str = "tokenized_text",
    ner_column: str = "ner",
    split: str = "train",
) -> List[Dict[str, Any]]:
    """Load data from a local JSON file or a HuggingFace dataset repo.

    Columns are mapped to GLiNER's native keys (tokenized_text, ner).
    Extra columns (relations, ner_negatives, ner_labels, etc.) are preserved.
    """
    if os.path.exists(file_or_repo):
        with open(file_or_repo, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raw = [raw]
        if text_column == "tokenized_text" and ner_column == "ner":
            return raw
        data = []
        for item in raw:
            mapped = dict(item)
            if text_column != "tokenized_text" and text_column in mapped:
                mapped["tokenized_text"] = mapped.pop(text_column)
            if ner_column != "ner" and ner_column in mapped:
                mapped["ner"] = mapped.pop(ner_column)
            data.append(mapped)
        return data

    from datasets import load_dataset

    dataset = load_dataset(file_or_repo, split=split)
    data = []
    for item in dataset:
        mapped = {"tokenized_text": item[text_column], "ner": item[ner_column]}
        for key in item:
            if key not in (text_column, ner_column):
                mapped[key] = item[key]
        data.append(mapped)
    return data


def validate_data(data: List[Dict[str, Any]]) -> tuple:
    """Validate that data conforms to GLiNER's native format.

    Checks the core fields required by all model variants (span, token,
    bi-encoder, decoder, relex, multitask) and optional relation fields.

    Returns:
        (is_valid, errors) tuple.
    """
    errors = []

    if not isinstance(data, list):
        return False, ["Data must be a list of dictionaries"]

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"[{i}] Item is not a dictionary")
            continue

        # --- tokenized_text ---
        if "tokenized_text" not in item:
            errors.append(f"[{i}] Missing 'tokenized_text'")
            continue

        tokens = item["tokenized_text"]
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            errors.append(f"[{i}] 'tokenized_text' must be a list of strings")
            continue

        num_tokens = len(tokens)

        # --- ner ---
        if "ner" not in item:
            errors.append(f"[{i}] Missing 'ner'")
            continue

        ner = item["ner"]
        if not isinstance(ner, list):
            errors.append(f"[{i}] 'ner' must be a list of [start, end, label]")
            continue

        for j, span in enumerate(ner):
            if not isinstance(span, (list, tuple)) or len(span) != 3:
                errors.append(f"[{i}].ner[{j}] Must be [start, end, label], got {span}")
                continue
            start, end, label = span
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"[{i}].ner[{j}] start/end must be integers")
            elif start < 0 or end < 0:
                errors.append(f"[{i}].ner[{j}] start/end must be non-negative")
            elif start > end:
                errors.append(f"[{i}].ner[{j}] start ({start}) > end ({end})")
            elif end >= num_tokens:
                errors.append(f"[{i}].ner[{j}] end ({end}) >= num_tokens ({num_tokens})")
            if not isinstance(label, str):
                errors.append(f"[{i}].ner[{j}] label must be a string")

        # --- relations (optional, for relex models) ---
        if "relations" in item:
            rels = item["relations"]
            if not isinstance(rels, list):
                errors.append(f"[{i}] 'relations' must be a list")
            else:
                num_entities = len(ner)
                for j, rel in enumerate(rels):
                    if not isinstance(rel, (list, tuple)) or len(rel) != 3:
                        errors.append(f"[{i}].relations[{j}] Must be [head, tail, type]")
                        continue
                    head, tail, rel_type = rel
                    if not isinstance(head, int) or not isinstance(tail, int):
                        errors.append(f"[{i}].relations[{j}] head/tail must be integers")
                    elif head >= num_entities or tail >= num_entities:
                        errors.append(
                            f"[{i}].relations[{j}] index out of bounds "
                            f"(head={head}, tail={tail}, num_entities={num_entities})"
                        )
                    if not isinstance(rel_type, str):
                        errors.append(f"[{i}].relations[{j}] relation type must be a string")

    return len(errors) == 0, errors


def extract_labels(data: List[Dict[str, Any]], key: str = "ner") -> List[str]:
    """Extract all unique labels from the dataset."""
    labels = set()
    for item in data:
        for span in item.get(key, []):
            if len(span) >= 3:
                labels.add(span[-1])
    return sorted(labels)


def prepare(
    file_or_repo: str,
    text_column: str = "tokenized_text",
    ner_column: str = "ner",
    validate: bool = True,
    generate_label_embeddings: Optional[str] = None,
    split: str = "train",
) -> GLiNERData:
    """Load, validate, and optionally embed labels. Returns a GLiNERData object.

    Args:
        file_or_repo: Path to a local JSON file or HuggingFace dataset repo id.
        text_column: Column name to map to ``tokenized_text``.
        ner_column: Column name to map to ``ner``.
        validate: Whether to run validation.
        generate_label_embeddings: Model name/path for bi-encoder label
            embedding generation (e.g. ``"urchade/gliner_multi-v2.1"``).
            Only supported for bi-encoder models.
        split: Dataset split when loading from HuggingFace.
    """
    data = load_data(file_or_repo, text_column, ner_column, split)
    labels = extract_labels(data)

    result = GLiNERData(data=data, labels=labels, source=file_or_repo)

    if validate:
        is_valid, errs = validate_data(data)
        result.is_valid = is_valid
        result.validation_errors = errs

    if generate_label_embeddings:
        from gliner import GLiNER

        model = GLiNER.from_pretrained(generate_label_embeddings)
        result.label_embeddings = model.encode_labels(labels)

    return result
