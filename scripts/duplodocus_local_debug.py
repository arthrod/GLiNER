#!/usr/bin/env python3
# %% [markdown]
# # Duplodocus Local Debug Notebook
#
# Notebook-style debug flow for local fuzzy deduplication with:
# - HF dataset -> JSONL materialization (`text` key)
# - `duplodocus minhash-memory` run (remove mode)
# - `duplodocus minhash-memory` run (annotate mode)
# - per-source and cluster-level duplicate analysis
#
# Run cell-by-cell in Jupyter/VS Code (`# %%`) or run as a script.

# %%
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

# %%
@dataclass(frozen=True)
class Config:
    repo_id: str = "arthrod/gliner-flex-pii-ready-v4"
    split: str = "train"
    start: int = 0
    end: int = 120_000
    rows_per_file: int = 20_000
    text_key: str = "text"
    run_root: Path = Path(
        os.environ.get(
            "DUPLODOCUS_RUN_ROOT",
            str(Path.home() / "duplodocus_runs" / "gliner_ready_v4_r0_120k_nb"),
        )
    )
    duplodocus_bin: Path = Path(os.environ.get("DUPLODOCUS_BIN", "duplodocus"))
    num_buckets: int = 20
    bucket_size: int = 5
    ngram_size: int = 5
    tokenizer: str = "cl100k"  # supported: p50k, cl100k, uniseg, bytes


CFG = Config()

INPUT_DIR = CFG.run_root / "input"
REMOVE_STORAGE_DIR = CFG.run_root / "storage_remove"
REMOVE_OUTPUT_DIR = CFG.run_root / "output_remove"
ANNOT_STORAGE_DIR = CFG.run_root / "storage_annot"
ANNOT_OUTPUT_DIR = CFG.run_root / "output_annot"

for p in [INPUT_DIR, REMOVE_STORAGE_DIR, REMOVE_OUTPUT_DIR, ANNOT_STORAGE_DIR, ANNOT_OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("duplodocus bin:", CFG.duplodocus_bin)
print("run root:", CFG.run_root)

# %% [markdown]
# ## 1) Materialize HF subset into JSONL with `text` field

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
    rows_written = 0

    for i, row in enumerate(subset):
        text = row_to_text(row)
        if not text:
            continue
        batch.append(
            {
                "id": f"{CFG.split}_{CFG.start + i}",
                "sample_id": row.get("sample_id", ""),
                "source": row.get("source", ""),
                "text": text,
            }
        )
        if len(batch) >= CFG.rows_per_file:
            out_path = INPUT_DIR / f"part_{file_idx:04d}.jsonl"
            with out_path.open("w", encoding="utf-8") as handle:
                for item in batch:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            rows_written += len(batch)
            file_idx += 1
            batch = []

    if batch:
        out_path = INPUT_DIR / f"part_{file_idx:04d}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for item in batch:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        rows_written += len(batch)
        file_idx += 1

    return file_idx, rows_written


num_files, rows_written = materialize_input()
print("input files:", num_files)
print("rows written:", rows_written)

# %% [markdown]
# ## 2) Run MinHash memory mode (`remove_duplicates=true`)

# %%
def run_duplodocus(output_dir: Path, storage_dir: Path, *, annotate: bool, remove_duplicates: bool) -> subprocess.CompletedProcess:
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
        str(CFG.num_buckets),
        "--bucket-size",
        str(CFG.bucket_size),
        "--ngram-size",
        str(CFG.ngram_size),
        "--tokenizer",
        CFG.tokenizer,
        "--annotate",
        "true" if annotate else "false",
        "--annotate-key",
        "dedup_info",
        "--remove-duplicates",
        "true" if remove_duplicates else "false",
    ]
    print("running:", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


remove_result = run_duplodocus(
    output_dir=REMOVE_OUTPUT_DIR,
    storage_dir=REMOVE_STORAGE_DIR,
    annotate=False,
    remove_duplicates=True,
)
print(remove_result.stdout)

# %% [markdown]
# ## 3) Run MinHash annotate mode (`annotate=true`, `remove_duplicates=false`)

# %%
annot_result = run_duplodocus(
    output_dir=ANNOT_OUTPUT_DIR,
    storage_dir=ANNOT_STORAGE_DIR,
    annotate=True,
    remove_duplicates=False,
)
print(annot_result.stdout)

# %% [markdown]
# ## 4) Analyze duplicate-prone examples and source contribution

# %%
def analyze_annotated_output(output_dir: Path) -> dict:
    source_total = Counter()
    source_dup = Counter()
    cc_size_distribution = Counter()
    cc_members = Counter()
    cc_examples: dict[int, list[dict]] = defaultdict(list)
    cc_idx_distribution = Counter()

    rows = 0
    annotated_rows = 0

    for path in sorted(output_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                src = str(row.get("source", ""))
                source_total[src] += 1

                info = row.get("dedup_info")
                if not isinstance(info, dict):
                    continue

                annotated_rows += 1
                source_dup[src] += 1

                cc_id = info.get("cc_id")
                cc_idx = info.get("cc_idx")
                cc_size = info.get("cc_size")
                if isinstance(cc_idx, int):
                    cc_idx_distribution[cc_idx] += 1
                if isinstance(cc_size, int):
                    cc_size_distribution[cc_size] += 1
                if isinstance(cc_id, int):
                    cc_members[cc_id] += 1
                    if len(cc_examples[cc_id]) < 3:
                        cc_examples[cc_id].append(
                            {
                                "sample_id": row.get("sample_id", ""),
                                "source": src,
                                "cc_idx": cc_idx,
                                "cc_size": cc_size,
                                "text_preview": str(row.get("text", ""))[:260],
                            }
                        )

    top_sources = []
    for source, total in source_total.most_common():
        dup = source_dup[source]
        rate = (dup * 100.0 / total) if total else 0.0
        if dup > 0:
            top_sources.append(
                {
                    "source": source,
                    "duplicate_rows": dup,
                    "source_rows": total,
                    "duplicate_rate_pct": round(rate, 4),
                }
            )

    top_clusters = []
    for cc_id, members in sorted(cc_members.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        top_clusters.append(
            {
                "cc_id": cc_id,
                "members": members,
                "examples": cc_examples[cc_id],
            }
        )

    remove_candidates = sum(v for k, v in cc_idx_distribution.items() if isinstance(k, int) and k > 0)

    return {
        "rows": rows,
        "annotated_rows": annotated_rows,
        "annotated_rate_pct": round((annotated_rows * 100.0 / rows) if rows else 0.0, 4),
        "remove_candidates": remove_candidates,
        "remove_candidates_rate_pct": round((remove_candidates * 100.0 / rows) if rows else 0.0, 4),
        "source_duplicate_stats": top_sources,
        "cc_idx_distribution": dict(cc_idx_distribution),
        "cc_size_distribution_top": dict(sorted(cc_size_distribution.items(), key=lambda kv: kv[0], reverse=True)[:20]),
        "top_clusters": top_clusters,
    }


summary = analyze_annotated_output(ANNOT_OUTPUT_DIR)
print(json.dumps(summary, indent=2, ensure_ascii=False))

# %%
summary_path = CFG.run_root / "analysis_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved:", summary_path)
