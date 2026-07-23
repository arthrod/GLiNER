#!/usr/bin/env python3
"""Convert the project's raw relation-extraction data to GLiNER RelEx format."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")


def tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return tokens and their half-open character spans in *text*."""
    matches = list(TOKEN_PATTERN.finditer(text))
    return [match.group() for match in matches], [match.span() for match in matches]


def entity_to_token_span(start: int, end: int, token_spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Map a half-open character span to inclusive token indices."""
    covered = [index for index, (token_start, token_end) in enumerate(token_spans) if token_start < end and token_end > start]
    if not covered:
        return None
    return covered[0], covered[-1]


def convert_record(record: dict) -> tuple[dict, int, int]:
    """Convert one raw record and return it with dropped entity/relation counts."""
    text = record["text"]
    extraction = record["extraction"][0]
    tokens, token_spans = tokenize(text)

    ner = []
    old_to_new_index = {}
    dropped_entities = 0
    for old_index, entity in enumerate(extraction.get("ner", [])):
        token_span = entity_to_token_span(entity["start"], entity["end"], token_spans)
        if token_span is None:
            dropped_entities += 1
            continue
        old_to_new_index[old_index] = len(ner)
        ner.append([token_span[0], token_span[1], entity["label"]])

    relations = []
    dropped_relations = 0
    for relation in extraction.get("relations", []):
        head_index, relation_label, tail_index = relation
        if head_index not in old_to_new_index or tail_index not in old_to_new_index:
            dropped_relations += 1
            continue
        relations.append([old_to_new_index[head_index], old_to_new_index[tail_index], relation_label])

    return {"tokenized_text": tokens, "ner": ner, "relations": relations}, dropped_entities, dropped_relations


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw RelEx JSON to GLiNER RelEx training format.")
    parser.add_argument("input", type=Path, help="Input JSON array with text/extraction fields")
    parser.add_argument("output", type=Path, nargs="?", help="Output JSON array path")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name(f"{input_path.stem}_gliner_relex.json")
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if output_path.exists() and not args.force:
        parser.error(f"Output file already exists: {output_path} (use --force to overwrite)")

    with input_path.open("r", encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list):
        parser.error("Input must be a JSON array")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_entities = 0
    dropped_relations = 0
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write("[\n")
            for index, record in enumerate(records):
                converted, entity_count, relation_count = convert_record(record)
                if index:
                    temporary.write(",\n")
                json.dump(converted, temporary, ensure_ascii=False)
                dropped_entities += entity_count
                dropped_relations += relation_count
            temporary.write("\n]\n")
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    os.replace(temporary_path, output_path)
    print(f"Converted {len(records)} records: {input_path} -> {output_path}")
    print(f"Dropped entities: {dropped_entities}; dropped relations: {dropped_relations}")


if __name__ == "__main__":
    main()
