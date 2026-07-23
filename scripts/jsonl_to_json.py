#!/usr/bin/env python3
"""Convert a JSON Lines dataset into a JSON array without loading it into memory."""

import argparse
import json
from pathlib import Path


def convert(input_path: Path, output_path: Path) -> int:
    """Write JSON objects from *input_path* as one JSON array and return their count."""
    count = 0
    first_item = True

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        destination.write("[\n")
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}: {error.msg}"
                ) from error

            if not first_item:
                destination.write(",\n")
            json.dump(item, destination, ensure_ascii=False)
            first_item = False
            count += 1
        destination.write("\n]\n")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a .jsonl file to a JSON array.")
    parser.add_argument("input", type=Path, help="Input JSON Lines file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output JSON file (default: input path with a .json suffix)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_suffix(".json")
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if output_path.exists() and not args.force:
        parser.error(f"Output file already exists: {output_path} (use --force to overwrite)")

    count = convert(input_path, output_path)
    print(f"Converted {count} records: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
