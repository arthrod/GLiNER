#!/usr/bin/env python3
"""Rebalance subject-label train/eval distribution by moving eval rows to train.

This script operates on already-normalized GLiNER JSONL datasets:
- canonical_pii/{train,eval}.jsonl
- canonical_pii_ready/{train,eval}.jsonl

It computes deterministic eval->train moves on canonical_pii, then applies the
same sample_id moves to canonical_pii_ready so positives remain aligned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = {
    "subject described political opinion": 0.70,
    "subject described religious conviction": 0.70,
    "subject described organization affiliation": 0.70,
    "sex or gender": 0.75,
    "political view": 0.80,
    "religious belief": 0.80,
}


@dataclass(frozen=True)
class EvalCandidate:
    sample_id: str
    labels: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebalance target subject labels to non-equal train/eval splits."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("training_feb_22/dataset/normalized_v3"),
        help="Input root containing canonical_pii and canonical_pii_ready.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training_feb_22/dataset/normalized_v4"),
        help="Output root for rebalanced datasets.",
    )
    parser.add_argument(
        "--canonical-dir",
        type=str,
        default="canonical_pii",
        help="Subdirectory name for positive-only dataset.",
    )
    parser.add_argument(
        "--ready-dir",
        type=str,
        default="canonical_pii_ready",
        help="Subdirectory name for positive+negative dataset.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Override target as 'label=fraction' (repeatable). "
            "Example: --target 'sex or gender=0.75'"
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def target_labels_in_row(row: dict[str, Any], label_set: set[str]) -> frozenset[str]:
    labels: set[str] = set()
    for ent in row.get("ner", []):
        if isinstance(ent, list) and len(ent) >= 3 and isinstance(ent[2], str) and ent[2] in label_set:
            labels.add(ent[2])
    return frozenset(labels)


def deterministic_key(sample_id: str) -> str:
    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()


def parse_targets(raw_targets: list[str]) -> dict[str, float]:
    targets = dict(DEFAULT_TARGETS)
    for raw in raw_targets:
        if "=" not in raw:
            raise ValueError(f"Invalid --target value: {raw!r}. Expected 'label=fraction'.")
        label, frac_text = raw.split("=", 1)
        label = label.strip()
        try:
            frac = float(frac_text.strip())
        except ValueError:
            raise ValueError(f"Invalid fraction '{frac_text.strip()}' in --target: {raw!r}") from None
        if not label:
            raise ValueError(f"Invalid empty label in --target value: {raw!r}")
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"Target fraction out of range [0,1]: {raw!r}")
        targets[label] = frac
    return targets


def compute_moves(
    canonical_train_path: Path,
    canonical_eval_path: Path,
    targets: dict[str, float],
) -> tuple[set[str], dict[str, Any]]:
    label_set = set(targets)
    train_counts: Counter[str] = Counter()
    eval_counts: Counter[str] = Counter()
    eval_candidates: list[EvalCandidate] = []

    for row in read_jsonl(canonical_train_path):
        present = target_labels_in_row(row, label_set)
        for label in present:
            train_counts[label] += 1

    for row in read_jsonl(canonical_eval_path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        present = target_labels_in_row(row, label_set)
        if not present:
            continue
        eval_candidates.append(EvalCandidate(sample_id=sample_id, labels=present))
        for label in present:
            eval_counts[label] += 1

    totals = {label: train_counts[label] + eval_counts[label] for label in targets}
    target_train = {
        label: min(totals[label], int(math.floor(totals[label] * targets[label] + 0.5)))
        for label in targets
    }
    deficits = {label: max(0, target_train[label] - train_counts[label]) for label in targets}

    selected: set[str] = set()
    moved_counts: Counter[str] = Counter()

    candidates_sorted = sorted(eval_candidates, key=lambda x: deterministic_key(x.sample_id))
    for cand in candidates_sorted:
        useful = [label for label in cand.labels if deficits[label] > 0]
        if not useful:
            continue
        selected.add(cand.sample_id)
        for label in cand.labels:
            moved_counts[label] += 1
            if deficits[label] > 0:
                deficits[label] -= 1
        if all(value <= 0 for value in deficits.values()):
            break

    after_train = {label: train_counts[label] + moved_counts[label] for label in targets}
    after_eval = {label: eval_counts[label] - moved_counts[label] for label in targets}

    report = {
        "targets": targets,
        "before": {
            "train": dict(train_counts),
            "eval": dict(eval_counts),
            "totals": totals,
        },
        "desired_train_counts": target_train,
        "moved_counts": dict(moved_counts),
        "after": {
            "train": after_train,
            "eval": after_eval,
            "train_pct": {
                label: round((after_train[label] * 100.0 / totals[label]), 4) if totals[label] else 0.0
                for label in targets
            },
            "eval_pct": {
                label: round((after_eval[label] * 100.0 / totals[label]), 4) if totals[label] else 0.0
                for label in targets
            },
        },
        "rows_selected_to_move": len(selected),
        "rows_considered_with_targets": len(eval_candidates),
        "remaining_deficit": deficits,
    }
    return selected, report


def write_rebalanced_dataset(
    input_dir: Path,
    output_dir: Path,
    sample_ids_to_move: set[str],
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counters = Counter()

    train_in = input_dir / "train.jsonl"
    eval_in = input_dir / "eval.jsonl"
    train_out = output_dir / "train.jsonl"
    eval_out = output_dir / "eval.jsonl"

    with (
        train_out.open("w", encoding="utf-8") as train_handle,
        eval_out.open("w", encoding="utf-8") as eval_handle,
    ):
        for row in read_jsonl(train_in):
            counters["train_kept"] += 1
            train_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        for row in read_jsonl(eval_in):
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id in sample_ids_to_move:
                counters["eval_moved_to_train"] += 1
                train_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                counters["eval_kept"] += 1
                eval_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return dict(counters)


def main() -> None:
    args = parse_args()
    targets = parse_targets(args.target)

    canonical_in = args.input_root / args.canonical_dir
    ready_in = args.input_root / args.ready_dir
    canonical_out = args.output_root / args.canonical_dir
    ready_out = args.output_root / args.ready_dir

    selected_ids, split_report = compute_moves(
        canonical_train_path=canonical_in / "train.jsonl",
        canonical_eval_path=canonical_in / "eval.jsonl",
        targets=targets,
    )

    canonical_counts = write_rebalanced_dataset(canonical_in, canonical_out, selected_ids)
    ready_counts = write_rebalanced_dataset(ready_in, ready_out, selected_ids)

    report = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "selected_sample_ids_to_move": len(selected_ids),
        "split_report": split_report,
        "write_counts": {
            "canonical_pii": canonical_counts,
            "canonical_pii_ready": ready_counts,
        },
    }

    report_path = args.output_root / "rebalance_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] Rebalanced datasets written to {args.output_root}")
    print(f"[done] Moved sample_ids: {len(selected_ids)}")
    print(f"[done] Report: {report_path}")


if __name__ == "__main__":
    main()

