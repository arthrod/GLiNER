"""Download specified datasets and save as JSONL."""

from __future__ import annotations

import json
import argparse
from typing import Any, Iterable
from pathlib import Path

from datasets import load_dataset, get_dataset_split_names

DATASETS: dict[str, dict[str, Any]] = {
    "nemotron_pii": {
        "repos": ["nvidia/Nemotron-PII"],
        "preferred_split": "train",
    },
    "open_pii_masking_500k": {
        "repos": ["ai4privacy/open-pii-masking-500k-ai4privacy"],
        "preferred_split": "train",
    },
    "pii_masking_400k": {
        "repos": ["ai4privacy/pii-masking-400k"],
        "preferred_split": "train",
    },
    "synthetic_pii_mistral_v1": {
        "repos": ["urchade/synthetic-pii-ner-mistral-v1"],
        "preferred_split": "train",
    },
    "lener_br": {
        "repos": ["peluz/lener_br", "lener_br"],
        "preferred_split": "train",
    },
    "harem": {
        "repos": ["Linguateca/harem", "harem"],
        "preferred_split": "train",
    },
    "portuguese_ner": {
        "repos": ["lfcc/portuguese_ner"],
        "preferred_split": None,
    },
    "sms_spam_multilingual": {
        "repos": [
            "dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset",
            "ucirvine/sms_spam",
        ],
        "preferred_split": "train",
    },
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(val) for val in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    try:
        import numpy as np

        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ModuleNotFoundError:
        pass
    return value


def resolve_repo_and_splits(repos: list[str], preferred_split: str | None) -> tuple[str, list[str]]:
    last_error: Exception | None = None
    for repo in repos:
        try:
            splits = get_dataset_split_names(repo)
            if preferred_split and preferred_split not in splits:
                print(
                    f"[warn] Preferred split '{preferred_split}' not found for {repo}. "
                    f"Available: {splits}. Falling back to all splits.",
                )
                return repo, list(splits)
            if preferred_split:
                return repo, [preferred_split]
            return repo, list(splits)
        except Exception as exc:
            last_error = exc
            print(f"[warn] Could not inspect dataset repo '{repo}': {exc}")
    raise RuntimeError(f"Unable to resolve any dataset repo from candidates={repos}: {last_error}")


def iter_splits(repo: str, splits: list[str]) -> Iterable[tuple[str, Any]]:
    for split_name in splits:
        yield split_name, load_dataset(repo, split=split_name)


def write_jsonl(dataset: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False))
            handle.write("\n")


def validate_export(dataset: Any, output_path: Path) -> None:
    if len(dataset) == 0:
        raise ValueError(f"Dataset split is empty; refusing to export: {output_path}")
    if output_path.stat().st_size == 0:
        raise ValueError(f"Output file is empty after export: {output_path}")
    with output_path.open(encoding="utf-8") as handle:
        line_count = sum(1 for line in handle if line.strip())
    if line_count != len(dataset):
        raise ValueError(
            f"Line count mismatch for {output_path}: expected {len(dataset)}, got {line_count}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets and export JSONL files.")
    parser.add_argument(
        "--output-dir",
        default="training_feb_22/dataset/preprocessed",
        help="Base directory for JSONL outputs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Optional dataset key from DATASETS to process (repeatable).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-export validation checks.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    selected = args.dataset or list(DATASETS.keys())

    for key in selected:
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset key: {key}")
        config = DATASETS[key]
        repos = config["repos"]
        preferred_split = config.get("preferred_split")
        repo, splits = resolve_repo_and_splits(repos, preferred_split)
        for split_name, dataset in iter_splits(repo, splits):
            output_path = output_dir / key / f"{split_name}.jsonl"
            print(f"Writing {repo}:{split_name} -> {output_path}")
            write_jsonl(dataset, output_path)
            if not args.no_validate:
                validate_export(dataset, output_path)
                print(f"Validated {output_path}")


if __name__ == "__main__":
    main()
