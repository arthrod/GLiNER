#!/usr/bin/env python3
"""Push GLiNER JSONL splits to Hugging Face Hub as rendered DatasetDict.

Input JSONL rows are expected to look like:
{
  "tokenized_text": [...],
  "ner": [[start, end, label], ...],
  "source": "...",
  "sample_id": "...",
  "is_negative": false,
  "ner_negatives": [...]
}

On Hub, `ner` is normalized to a sequence of structs:
ner: [{"start": int, "end": int, "label": str}, ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push GLiNER JSONL train/eval files as a Hugging Face DatasetDict."
    )
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. user/dataset-name")
    parser.add_argument("--train-jsonl", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument("--eval-jsonl", type=Path, required=True, help="Path to eval.jsonl")
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create/push private dataset repo (default: true).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report file to upload as build_report.json.",
    )
    parser.add_argument(
        "--readme-path",
        type=Path,
        default=None,
        help="Optional README file to upload as README.md after push.",
    )
    return parser.parse_args()


def make_features() -> Features:
    return Features(
        {
            "tokenized_text": Sequence(Value("string")),
            "ner": [
                {
                    "start": Value("int32"),
                    "end": Value("int32"),
                    "label": Value("string"),
                }
            ],
            "source": Value("string"),
            "sample_id": Value("string"),
            "is_negative": Value("bool"),
            "ner_negatives": Sequence(Value("string")),
        }
    )


def _normalize_ner(ner: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(ner, list):
        return out
    for item in ner:
        if isinstance(item, list) and len(item) >= 3:
            try:
                start = int(item[0])
                end = int(item[1])
                label_val = item[2]
            except (TypeError, ValueError):
                continue
            if label_val is None:
                continue
            label = str(label_val).strip()
            if not label:
                continue
            out.append({"start": start, "end": end, "label": label})
        elif isinstance(item, dict):
            try:
                start = int(item.get("start"))
                end = int(item.get("end"))
                label_val = item.get("label")
            except (TypeError, ValueError):
                continue
            if label_val is None:
                continue
            label = str(label_val).strip()
            if not label:
                continue
            out.append({"start": start, "end": end, "label": label})
    return out


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return False


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    tokenized_text = row.get("tokenized_text")
    if not isinstance(tokenized_text, list):
        tokenized_text = []

    ner_negatives = row.get("ner_negatives")
    if not isinstance(ner_negatives, list):
        ner_negatives = []

    return {
        "tokenized_text": [str(tok) for tok in tokenized_text],
        "ner": _normalize_ner(row.get("ner", [])),
        "source": str(row.get("source", "")),
        "sample_id": str(row.get("sample_id", "")),
        "is_negative": _coerce_bool(row.get("is_negative", False)),
        "ner_negatives": [str(lbl) for lbl in ner_negatives],
    }


def iter_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield _normalize_row(row)


def build_dataset(path: Path, features: Features) -> Dataset:
    return Dataset.from_generator(
        iter_jsonl,
        features=features,
        gen_kwargs={"path": str(path)},
    )


def main() -> None:
    args = parse_args()
    features = make_features()

    print(f"[info] Building train split from {args.train_jsonl}")
    train_ds = build_dataset(args.train_jsonl, features=features)
    print(f"[info] Building eval split from {args.eval_jsonl}")
    eval_ds = build_dataset(args.eval_jsonl, features=features)

    ds = DatasetDict({"train": train_ds, "eval": eval_ds})
    print(f"[info] Pushing DatasetDict to {args.repo_id} (private={args.private})")
    ds.push_to_hub(args.repo_id, private=args.private)

    api = HfApi()
    if args.report_path is not None and args.report_path.exists():
        api.upload_file(
            path_or_fileobj=str(args.report_path),
            path_in_repo="build_report.json",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"[info] Uploaded {args.report_path} as build_report.json")
    if args.readme_path is not None and args.readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(args.readme_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"[info] Uploaded {args.readme_path} as README.md")

    print("[done] Push complete.")


if __name__ == "__main__":
    main()
