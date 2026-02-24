"""Summarize JSONL datasets in a folder (columns, NER labels, and metadata)."""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List


def _parse_list_like(value: Any) -> list[Any] | None:
    """Return a list from a native list or a serialized list string."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None

    if isinstance(parsed, list):
        return parsed
    return None


def _extract_ner_from_output_entities(item: Dict[str, Any]) -> Counter[str]:
    labels: Counter[str] = Counter()
    output = item.get("output")
    if not isinstance(output, dict):
        return labels
    entities = output.get("entities")
    if not isinstance(entities, dict):
        return labels
    for label, mentions in entities.items():
        if isinstance(label, str) and isinstance(mentions, list):
            labels[label] += len(mentions)
    return labels


def _extract_ner_from_spans(spans: Any) -> Counter[str]:
    labels: Counter[str] = Counter()
    parsed_spans = _parse_list_like(spans)
    if parsed_spans is None:
        return labels
    for span in parsed_spans:
        if isinstance(span, (list, tuple)) and len(span) >= 3 and isinstance(span[2], str):
            labels[span[2]] += 1
            continue
        if isinstance(span, dict):
            label = (
                span.get("label")
                or span.get("entity")
                or span.get("entity_type")
                or span.get("type")
                or span.get("tag")
            )
            if isinstance(label, str):
                normalized = label.strip()
                if normalized:
                    labels[normalized] += 1
    return labels


def _extract_ner_from_privacy_mask(privacy_mask: Any) -> Counter[str]:
    labels: Counter[str] = Counter()
    parsed = _parse_list_like(privacy_mask)
    if parsed is None:
        return labels

    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label") or entry.get("entity") or entry.get("type")
        if isinstance(label, str):
            normalized = label.strip()
            if normalized:
                labels[normalized] += 1
    return labels


def _extract_ner_from_string_tags(tags: list[Any]) -> Counter[str]:
    labels: Counter[str] = Counter()
    previous_label: str | None = None

    for tag in tags:
        if not isinstance(tag, str):
            previous_label = None
            continue

        normalized = tag.strip()
        if not normalized or normalized.upper() == "O":
            previous_label = None
            continue

        prefix = "I"
        label = normalized

        if "-" in normalized:
            maybe_prefix, maybe_label = normalized.split("-", 1)
            if maybe_prefix.upper() in {"B", "I", "L", "U", "S", "E"} and maybe_label:
                prefix, label = maybe_prefix.upper(), maybe_label

        if prefix in {"B", "U", "S"} or previous_label != label:
            labels[label] += 1
        previous_label = label

    return labels


def _extract_ner_from_numeric_tags(tags: list[Any]) -> Counter[str]:
    labels: Counter[str] = Counter()
    previous_label: str | None = None

    for tag in tags:
        value: int | None = None
        if isinstance(tag, int):
            value = tag
        elif isinstance(tag, str) and tag.strip().lstrip("+-").isdigit():
            value = int(tag)

        if value is None or value <= 0:
            previous_label = None
            continue

        # Common HF BIO encoding uses 0=O and consecutive B/I id pairs.
        base_id = value if value % 2 == 1 else value - 1
        label = f"tag_id_{base_id}"

        if previous_label != label:
            labels[label] += 1
        previous_label = label

    return labels


def _extract_ner_from_tag_sequence(tag_sequence: Any) -> Counter[str]:
    parsed = _parse_list_like(tag_sequence)
    if parsed is None:
        return Counter()

    has_string_tags = any(isinstance(tag, str) and any(ch.isalpha() for ch in tag) for tag in parsed)
    if has_string_tags:
        return _extract_ner_from_string_tags(parsed)
    return _extract_ner_from_numeric_tags(parsed)


def _extract_text_length(item: Dict[str, Any], text_column: str) -> int | None:
    value = item.get(text_column)
    if isinstance(value, list) and all(isinstance(token, str) for token in value):
        return len(value)
    if isinstance(value, str):
        return len(value.split())

    # Common fallback text keys in mixed datasets.
    for key in ("text", "input", "source_text", "masked_text", "text_tagged"):
        fallback = item.get(key)
        if isinstance(fallback, str):
            return len(fallback.split())
    return None


def iter_jsonl(path: str, max_rows: int | None = None) -> Iterable[Dict[str, Any]]:
    count = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                yield {"__parse_error__": True}
                continue
            yield item
            count += 1
            if max_rows is not None and count >= max_rows:
                break


def summarize_dataset(
    path: str,
    ner_column: str,
    text_column: str,
    max_rows: int | None,
) -> Dict[str, Any]:
    keys: Counter[str] = Counter()
    ner_labels: Counter[str] = Counter()
    ner_sources: Counter[str] = Counter()
    ner_spans = 0
    token_lengths: List[int] = []
    parse_errors = 0
    total_rows = 0

    for item in iter_jsonl(path, max_rows=max_rows):
        total_rows += 1
        if "__parse_error__" in item:
            parse_errors += 1
            continue
        if not isinstance(item, dict):
            keys["__non_dict__"] += 1
            continue
        keys.update(item.keys())

        text_len = _extract_text_length(item, text_column)
        if text_len is not None:
            token_lengths.append(text_len)

        extracted = Counter()

        # 1) Preferred user-specified ner column
        if ner_column in item:
            extracted.update(_extract_ner_from_spans(item.get(ner_column)))
            if extracted:
                ner_sources[ner_column] += sum(extracted.values())

        # 2) Common alternatives
        if not extracted and "spans" in item:
            spans_labels = _extract_ner_from_spans(item.get("spans"))
            extracted.update(spans_labels)
            if spans_labels:
                ner_sources["spans"] += sum(spans_labels.values())

        if not extracted:
            output_labels = _extract_ner_from_output_entities(item)
            extracted.update(output_labels)
            if output_labels:
                ner_sources["output.entities"] += sum(output_labels.values())

        if not extracted and "privacy_mask" in item:
            privacy_mask_labels = _extract_ner_from_privacy_mask(item.get("privacy_mask"))
            extracted.update(privacy_mask_labels)
            if privacy_mask_labels:
                ner_sources["privacy_mask"] += sum(privacy_mask_labels.values())

        if not extracted and "mbert_token_classes" in item:
            token_class_labels = _extract_ner_from_tag_sequence(item.get("mbert_token_classes"))
            extracted.update(token_class_labels)
            if token_class_labels:
                ner_sources["mbert_token_classes"] += sum(token_class_labels.values())

        if not extracted and "ner_tags" in item:
            ner_tag_labels = _extract_ner_from_tag_sequence(item.get("ner_tags"))
            extracted.update(ner_tag_labels)
            if ner_tag_labels:
                ner_sources["ner_tags"] += sum(ner_tag_labels.values())

        ner_labels.update(extracted)
        ner_spans += sum(extracted.values())

    token_stats: Dict[str, Any] | None = None
    if token_lengths:
        token_stats = {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "avg": sum(token_lengths) / len(token_lengths),
            "count": len(token_lengths),
        }

    return {
        "path": path,
        "rows": total_rows,
        "parse_errors": parse_errors,
        "columns": sorted(keys.keys()),
        "column_counts": dict(sorted(keys.items())),
        "ner_labels": sorted(ner_labels.keys()),
        "ner_label_counts": dict(sorted(ner_labels.items())),
        "ner_span_count": ner_spans,
        "ner_detected_sources": dict(sorted(ner_sources.items())),
        "token_stats": token_stats,
        "text_column": text_column,
        "ner_column": ner_column,
    }


def find_jsonl_files(input_dir: str, recursive: bool) -> List[str]:
    matches: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.endswith(".jsonl"):
                    matches.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl"):
                matches.append(os.path.join(input_dir, filename))
    return sorted(matches)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize JSONL datasets in a folder (columns, NER labels, metadata)."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing JSONL files.")
    parser.add_argument(
        "--output",
        default="dataset_summary.json",
        help="Output JSON file path.",
    )
    parser.add_argument("--ner-column", default="ner", help="NER column name.")
    parser.add_argument(
        "--text-column",
        default="tokenized_text",
        help="Tokenized text column name.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for JSONL files recursively.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max rows per file to scan (for quick sampling).",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    files = find_jsonl_files(input_dir, args.recursive)
    if not files:
        raise SystemExit(f"No .jsonl files found in: {input_dir}")

    datasets = [
        summarize_dataset(
            path, args.ner_column, args.text_column, max_rows=args.max_rows
        )
        for path in files
    ]

    summary = {
        "input_dir": input_dir,
        "file_count": len(files),
        "files": datasets,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output} with {len(files)} dataset summaries.")


if __name__ == "__main__":
    main()
