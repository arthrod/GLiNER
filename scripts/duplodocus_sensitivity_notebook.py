#!/usr/bin/env python3
# %% [markdown]
# # Duplodocus Sensitivity Notebook
#
# Notebook-style script to compare MinHash sensitivity levels and analyze:
# - how many docs are removed at each level
# - which labels are more prone to deduplication
# - cluster-size behavior (from annotate mode)
#
# Uses local duplodocus binary + HF dataset materialization.

# %%
from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

# %%
@dataclass(frozen=True)
class SweepRun:
    name: str
    num_buckets: int
    bucket_size: int
    ngram_size: int = 5
    tokenizer: str = "cl100k"


@dataclass(frozen=True)
class Config:
    repo_id: str = "arthrod/gliner-flex-pii-ready-v4"
    split: str = "train"
    start: int = 0
    end: int = 120_000
    rows_per_file: int = 20_000
    text_key: str = "text"
    run_root: Path = Path("/Users/arthrod/temp/T/_train/duplodocus_runs/gliner_ready_v4_sensitivity_nb")
    duplodocus_bin: Path = Path("/Users/arthrod/temp/T/_train/duplodocus/target/release/duplodocus")
    runs: tuple[SweepRun, ...] = (
        SweepRun(name="base_20x5", num_buckets=20, bucket_size=5),
        SweepRun(name="moderate_25x4", num_buckets=25, bucket_size=4),
        SweepRun(name="sensitive_30x3", num_buckets=30, bucket_size=3),
    )


CFG = Config()
INPUT_DIR = CFG.run_root / "input"

for p in [CFG.run_root, INPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("duplodocus:", CFG.duplodocus_bin)
print("run root:", CFG.run_root)

# %% [markdown]
# ## 1) Materialize HF subset to JSONL (`text` + `labels`)

# %%
def row_to_text(row: dict) -> str:
    value = row.get(CFG.text_key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    toks = row.get("tokenized_text")
    if isinstance(toks, list):
        return " ".join(str(tok) for tok in toks).strip()
    return ""


def materialize_input() -> tuple[int, int]:
    ds = load_dataset(CFG.repo_id, split=CFG.split)
    end = min(CFG.end, len(ds))
    subset = ds.select(range(CFG.start, end))

    batch: list[dict] = []
    file_idx = 0
    written = 0

    for i, row in enumerate(subset):
        text = row_to_text(row)
        if not text:
            continue

        labels: list[str] = []
        ner = row.get("ner")
        if isinstance(ner, list):
            labels = sorted(
                {
                    ent.get("label")
                    for ent in ner
                    if isinstance(ent, dict) and isinstance(ent.get("label"), str)
                }
            )

        item = {
            "id": f"{CFG.split}_{CFG.start + i}",
            "sample_id": row.get("sample_id", ""),
            "source": row.get("source", ""),
            "text": text,
            "labels": labels,
            "is_negative": bool(row.get("is_negative", False)),
        }
        batch.append(item)

        if len(batch) >= CFG.rows_per_file:
            out_path = INPUT_DIR / f"part_{file_idx:04d}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for x in batch:
                    handle.write(json.dumps(x, ensure_ascii=False) + "\n")
            written += len(batch)
            file_idx += 1
            batch = []

    if batch:
        out_path = INPUT_DIR / f"part_{file_idx:04d}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for x in batch:
                handle.write(json.dumps(x, ensure_ascii=False) + "\n")
        written += len(batch)
        file_idx += 1

    return file_idx, written


files, rows = materialize_input()
print("input files:", files)
print("rows:", rows)

# %% [markdown]
# ## 2) Run duplodocus for each sensitivity level

# %%
def run_duplodocus(run: SweepRun, *, annotate: bool, remove_duplicates: bool) -> subprocess.CompletedProcess:
    storage_dir = CFG.run_root / f"{run.name}__storage__{'annot' if annotate else 'remove'}"
    output_dir = CFG.run_root / f"{run.name}__output__{'annot' if annotate else 'remove'}"
    storage_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(CFG.duplodocus_bin),
        "minhash-memory",
        "--input-dir",
        str(INPUT_DIR),
        "--storage-dir",
        str(storage_dir),
        "--output-dir",
        str(output_dir),
        "--text-key",
        "text",
        "--num-buckets",
        str(run.num_buckets),
        "--bucket-size",
        str(run.bucket_size),
        "--ngram-size",
        str(run.ngram_size),
        "--tokenizer",
        run.tokenizer,
        "--annotate",
        "true" if annotate else "false",
        "--annotate-key",
        "dedup_info",
        "--remove-duplicates",
        "true" if remove_duplicates else "false",
    ]
    print("running:", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


for run in CFG.runs:
    remove_proc = run_duplodocus(run, annotate=False, remove_duplicates=True)
    print(f"\n[{run.name} remove] stdout:\n{remove_proc.stdout}")

    annot_proc = run_duplodocus(run, annotate=True, remove_duplicates=False)
    print(f"\n[{run.name} annotate] stdout:\n{annot_proc.stdout}")

# %% [markdown]
# ## 3) Analyze sensitivity impact + label propensity

# %%
def load_input_docs(input_dir: Path) -> tuple[dict[str, dict], Counter[str], Counter[str]]:
    docs: dict[str, dict] = {}
    label_total: Counter[str] = Counter()
    source_total: Counter[str] = Counter()
    for p in sorted(input_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = row["id"]
                labels = [lbl for lbl in row.get("labels", []) if isinstance(lbl, str)]
                labels = sorted(set(labels))
                src = str(row.get("source", ""))
                docs[doc_id] = {
                    "labels": labels,
                    "source": src,
                    "is_negative": bool(row.get("is_negative", False)),
                }
                source_total[src] += 1
                if not docs[doc_id]["is_negative"]:
                    for lbl in labels:
                        label_total[lbl] += 1
    return docs, label_total, source_total


def kept_ids(out_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in sorted(out_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                ids.add(json.loads(line)["id"])
    return ids


def analyze_annot(annot_dir: Path) -> dict:
    cc_size_dist: Counter[int] = Counter()
    cc_idx_dist: Counter[int] = Counter()
    rows = 0
    for p in sorted(annot_dir.glob("*.jsonl")):
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                info = row.get("dedup_info")
                if not isinstance(info, dict):
                    continue
                rows += 1
                cc_size = info.get("cc_size")
                cc_idx = info.get("cc_idx")
                if isinstance(cc_size, int):
                    cc_size_dist[cc_size] += 1
                if isinstance(cc_idx, int):
                    cc_idx_dist[cc_idx] += 1
    return {
        "annot_rows": rows,
        "remove_candidates": sum(v for k, v in cc_idx_dist.items() if k > 0),
        "largest_cc_size": max(cc_size_dist) if cc_size_dist else 0,
        "cc_size_distribution_top": cc_size_dist.most_common(20),
        "cc_idx_distribution": dict(cc_idx_dist),
    }


docs, label_total, source_total = load_input_docs(INPUT_DIR)
all_ids = set(docs)

removed_by_run: dict[str, set[str]] = {}
report: dict = {
    "rows_total": len(all_ids),
    "source_total": dict(source_total),
    "runs": {},
    "label_top_by_rate": {},
    "delta_vs_base": {},
}

for run in CFG.runs:
    remove_out = CFG.run_root / f"{run.name}__output__remove"
    annot_out = CFG.run_root / f"{run.name}__output__annot"
    removed = all_ids - kept_ids(remove_out)
    removed_by_run[run.name] = removed

    source_removed = Counter(docs[i]["source"] for i in removed)
    label_removed = Counter()
    for i in removed:
        if docs[i]["is_negative"]:
            continue
        for lbl in docs[i]["labels"]:
            label_removed[lbl] += 1

    report["runs"][run.name] = {
        "params": {
            "num_buckets": run.num_buckets,
            "bucket_size": run.bucket_size,
            "ngram_size": run.ngram_size,
            "tokenizer": run.tokenizer,
        },
        "removed": len(removed),
        "removed_rate_pct": round((len(removed) * 100.0 / len(all_ids)), 6),
        "source_removed": dict(source_removed),
        "annot": analyze_annot(annot_out),
    }

    by_rate: list[dict] = []
    for lbl, total in label_total.items():
        if total < 100:
            continue
        rm = label_removed[lbl]
        by_rate.append(
            {
                "label": lbl,
                "support": total,
                "removed": rm,
                "removed_rate_pct": round((rm * 100.0 / total), 4),
            }
        )
    by_rate.sort(key=lambda x: x["removed_rate_pct"], reverse=True)
    report["label_top_by_rate"][run.name] = by_rate[:40]

base_name = CFG.runs[0].name
base_removed = removed_by_run[base_name]

for run in CFG.runs[1:]:
    extra = removed_by_run[run.name] - base_removed
    extra_label = Counter()
    for i in extra:
        if docs[i]["is_negative"]:
            continue
        for lbl in docs[i]["labels"]:
            extra_label[lbl] += 1
    rows = []
    for lbl, total in label_total.items():
        if total < 100:
            continue
        extra_n = extra_label[lbl]
        rows.append(
            {
                "label": lbl,
                "support": total,
                "extra_removed": extra_n,
                "extra_pct_of_label": round((extra_n * 100.0 / total), 4),
            }
        )
    rows.sort(key=lambda x: x["extra_removed"], reverse=True)
    report["delta_vs_base"][run.name] = rows[:40]

print(json.dumps(report, indent=2, ensure_ascii=False))

# %%
report_path = CFG.run_root / "sensitivity_compare_report.json"
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved report:", report_path)

