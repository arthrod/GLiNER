#!/usr/bin/env python3
"""Colab-oriented fuzzy dedup sweep for HF datasets with range support.

This script:
1) Loads a split from Hugging Face `datasets`.
2) Materializes one or more row ranges to parquet input files.
3) Runs NeMo Curator fuzzy dedup for multiple hyperparameter combinations.
4) Produces run metrics + top duplicate cluster examples for analysis.

Designed for GPU-enabled environments (e.g., Google Colab with CUDA).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


def parse_int_values(spec: str) -> list[int]:
    """Parse integer values from comma-separated values and ranges.

    Supported tokens:
    - "24"
    - "20:28:4" (inclusive stop)
    """
    values: list[int] = []
    for token in [part.strip() for part in spec.split(",") if part.strip()]:
        if ":" not in token:
            values.append(int(token))
            continue
        parts = [p.strip() for p in token.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid int range token: {token!r}. Use start:stop:step.")
        start, stop, step = map(int, parts)
        if step == 0:
            raise ValueError(f"Step cannot be 0 in token: {token!r}")
        if (stop - start) * step < 0:
            raise ValueError(f"Range direction and step mismatch in token: {token!r}")
        i = start
        if step > 0:
            while i <= stop:
                values.append(i)
                i += step
        else:
            while i >= stop:
                values.append(i)
                i += step
    if not values:
        raise ValueError("No integer values parsed.")
    return sorted(set(values))


def parse_row_ranges(spec: str, max_len: int) -> list[tuple[int, int]]:
    """Parse row ranges like '0:200000,200000:400000,400000:'."""
    ranges: list[tuple[int, int]] = []
    tokens = [part.strip() for part in spec.split(",") if part.strip()]
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"Invalid row range token: {token!r}. Use start:end.")
        left, right = token.split(":", 1)
        start = int(left) if left.strip() else 0
        end = int(right) if right.strip() else max_len
        start = max(0, start)
        end = min(max_len, end)
        if end <= start:
            continue
        ranges.append((start, end))
    if not ranges:
        raise ValueError("No valid row ranges parsed.")
    return ranges


def parse_str_values(spec: str) -> list[str]:
    vals = [part.strip() for part in spec.split(",") if part.strip()]
    if not vals:
        raise ValueError("No values parsed.")
    return vals


def ensure_nemo_deps() -> None:
    """
    Verify that the Python packages required for the fuzzy dedup workflow are importable.
    
    Checks for `torch`, `ray`, and `nemo_curator`. If any package is not installed, raises a RuntimeError listing the missing packages and advising installation. If an import fails for reasons other than missing distribution (for example CUDA/driver issues for `torch`), raises a RuntimeError with a short diagnostic message.
    
    Raises:
        RuntimeError: If one or more required packages are missing or if an import fails due to other errors (includes guidance for troubleshooting).
    """
    missing = []
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        missing.append("torch")
    except Exception as exc:
        raise RuntimeError("Failed to import torch; check CUDA/driver compatibility.") from exc
    try:
        import ray  # noqa: F401
    except ModuleNotFoundError:
        missing.append("ray")
    except Exception as exc:
        raise RuntimeError("Failed to import ray; check installation.") from exc
    try:
        import nemo_curator  # noqa: F401
    except ModuleNotFoundError:
        missing.append("nemo-curator")
    except Exception as exc:
        raise RuntimeError("Failed to import nemo_curator; check installation.") from exc
    if missing:
        raise RuntimeError(
            "Missing required dependencies for fuzzy dedup: "
            + ", ".join(missing)
            + ". Install them in Colab before running."
        )


def row_to_text(row: dict[str, Any], text_field: str) -> str:
    if text_field in row and isinstance(row[text_field], str):
        return row[text_field]
    toks = row.get("tokenized_text")
    if isinstance(toks, list):
        return " ".join(str(t) for t in toks)
    return ""


def materialize_range_to_parquet(
    ds,
    start: int,
    end: int,
    parquet_dir: Path,
    text_field: str,
) -> pd.DataFrame:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    subset = ds.select(range(start, end))

    rows: list[dict[str, Any]] = []
    for local_idx, row in enumerate(subset):
        text = row_to_text(row, text_field=text_field).strip()
        if not text:
            continue
        source = str(row.get("source", ""))
        sample_id = str(row.get("sample_id", ""))
        rows.append(
            {
                "id": f"{start + local_idx}",
                "text": text,
                "source": source,
                "sample_id": sample_id,
                "__local_row_id": len(rows),  # maps directly to generated dedup id in this input
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No non-empty text rows in range [{start}, {end}).")

    df.to_parquet(parquet_dir / "part_0.parquet", index=False)
    df.to_parquet(parquet_dir / "input_index.parquet", index=False)
    return df


@dataclass(frozen=True)
class SweepConfig:
    range_start: int
    range_end: int
    char_ngrams: int
    minhashes_per_band: int
    bands_per_iteration: int
    input_blocksize: str


def connected_components_metrics(cc_df: pd.DataFrame) -> dict[str, Any]:
    if cc_df.empty:
        return {
            "cc_rows": 0,
            "cc_groups": 0,
            "largest_cluster_size": 0,
            "p95_cluster_size": 0,
            "p99_cluster_size": 0,
        }
    counts = cc_df["_duplicate_group_id"].value_counts()
    return {
        "cc_rows": int(len(cc_df)),
        "cc_groups": int(counts.shape[0]),
        "largest_cluster_size": int(counts.iloc[0]),
        "p95_cluster_size": float(counts.quantile(0.95)),
        "p99_cluster_size": float(counts.quantile(0.99)),
    }


def extract_top_cluster_examples(
    cc_df: pd.DataFrame,
    input_df: pd.DataFrame,
    top_k_clusters: int,
    examples_per_cluster: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if cc_df.empty:
        return out

    cluster_sizes = cc_df["_duplicate_group_id"].value_counts().head(top_k_clusters)
    for group_id, size in cluster_sizes.items():
        group_rows = cc_df.loc[cc_df["_duplicate_group_id"] == group_id, "_curator_dedup_id"].tolist()
        examples = []
        for dedup_id in group_rows[:examples_per_cluster]:
            if 0 <= int(dedup_id) < len(input_df):
                row = input_df.iloc[int(dedup_id)]
                examples.append(
                    {
                        "dedup_id": int(dedup_id),
                        "source": str(row.get("source", "")),
                        "sample_id": str(row.get("sample_id", "")),
                        "text_preview": str(row["text"])[:280],
                        "text_len": int(len(str(row["text"]))),
                    }
                )
        out.append(
            {
                "duplicate_group_id": int(group_id),
                "cluster_size": int(size),
                "examples": examples,
            }
        )
    return out


def run_single_sweep(
    cfg: SweepConfig,
    output_root: Path,
    input_df: pd.DataFrame,
    io_kwargs: dict[str, Any] | None,
    seed: int,
    num_top_clusters: int,
    examples_per_cluster: int,
    run_removal: bool,
) -> dict[str, Any]:
    """
    Run a single fuzzy-deduplication sweep with the given configuration and persist results.
    
    Parameters:
        cfg (SweepConfig): Sweep configuration for this run (range, n-gram and hashing params, blocksize).
        output_root (Path): Root directory where run outputs and analysis will be written.
        input_df (pd.DataFrame): Dataframe containing the input rows for the configured range; must include a `text` column.
        io_kwargs (dict[str, Any] | None): Optional read/write kwargs passed to workflow stages (e.g., storage options); may be None.
        seed (int): RNG seed used by the deduplication workflow.
        num_top_clusters (int): Number of largest duplicate clusters to extract examples for.
        examples_per_cluster (int): Number of example rows to include for each selected cluster.
        run_removal (bool): If true, run the duplicate-removal stage using produced duplicate IDs and write a deduplicated dataset.
    
    Returns:
        dict[str, Any]: Metrics summary for the run including `run_id`, `docs_total`, `duplicate_ids_to_remove`,
        `duplicate_rate_pct`, timing fields (`identify_seconds`, `removal_seconds`), the sweep configuration fields,
        and connected-components summary metrics.
    
    Raises:
        RuntimeError: If `run_removal` is true but the duplicate IDs file expected from the identify stage does not exist.
    """
    from nemo_curator.backends.experimental.ray_data import RayDataExecutor
    from nemo_curator.stages.deduplication.fuzzy import FuzzyDeduplicationWorkflow
    from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
    from nemo_curator.stages.text.deduplication import TextDuplicatesRemovalWorkflow

    run_id = (
        f"r{cfg.range_start}_{cfg.range_end}"
        f"__cg{cfg.char_ngrams}"
        f"__mhb{cfg.minhashes_per_band}"
        f"__bpi{cfg.bands_per_iteration}"
        f"__blk{cfg.input_blocksize.replace('i', '').replace('B', '').lower()}"
    )
    run_dir = output_root / run_id
    input_dir = run_dir / "input"
    fuzzy_output_dir = run_dir / "fuzzy_outputs"
    cache_dir = fuzzy_output_dir / "cache"
    deduped_out = fuzzy_output_dir / "fuzzy_deduped_dataset"

    # Persist range input for this run.
    input_dir.mkdir(parents=True, exist_ok=True)
    input_df.to_parquet(input_dir / "part_0.parquet", index=False)

    st = time.time()
    identify = FuzzyDeduplicationWorkflow(
        cache_path=str(cache_dir),
        output_path=str(fuzzy_output_dir),
        input_path=str(input_dir),
        input_filetype="parquet",
        input_blocksize=cfg.input_blocksize,
        text_field="text",
        seed=seed,
        char_ngrams=cfg.char_ngrams,
        minhashes_per_band=cfg.minhashes_per_band,
        bands_per_iteration=cfg.bands_per_iteration,
        read_kwargs=io_kwargs,
        cache_kwargs=io_kwargs,
        write_kwargs=io_kwargs,
    )
    identify.run()
    identify_sec = time.time() - st

    duplicate_ids_path = fuzzy_output_dir / "FuzzyDuplicateIds"
    cc_path = cache_dir / "ConnectedComponentsStage"

    duplicate_count = 0
    if duplicate_ids_path.exists():
        duplicate_count = int(len(pd.read_parquet(duplicate_ids_path)))
    cc_df = pd.read_parquet(cc_path) if cc_path.exists() else pd.DataFrame()
    cc_metrics = connected_components_metrics(cc_df)
    top_clusters = extract_top_cluster_examples(
        cc_df=cc_df,
        input_df=input_df,
        top_k_clusters=num_top_clusters,
        examples_per_cluster=examples_per_cluster,
    )

    removal_sec = None
    if run_removal:
        if not duplicate_ids_path.exists():
            raise RuntimeError(
                f"Expected duplicate IDs at {duplicate_ids_path} but none were produced. "
                "Disable --run-removal or inspect the identify stage output."
            )
        st2 = time.time()
        removal = TextDuplicatesRemovalWorkflow(
            input_path=str(input_dir),
            ids_to_remove_path=str(duplicate_ids_path),
            output_path=str(deduped_out),
            input_filetype="parquet",
            input_blocksize=cfg.input_blocksize,
            duplicate_id_field=CURATOR_DEDUP_ID_STR,
            id_generator_path=str(fuzzy_output_dir / "fuzzy_id_generator.json"),
            output_filetype="parquet",
            input_kwargs=io_kwargs,
            duplicate_id_read_kwargs=io_kwargs,
            id_generator_storage_options=None,
            output_kwargs=io_kwargs,
        )
        removal.run(executor=RayDataExecutor())
        removal_sec = time.time() - st2

    docs_total = int(len(input_df))
    metrics = {
        "run_id": run_id,
        "docs_total": docs_total,
        "duplicate_ids_to_remove": duplicate_count,
        "duplicate_rate_pct": round((duplicate_count * 100.0 / docs_total) if docs_total else 0.0, 4),
        "identify_seconds": round(identify_sec, 3),
        "removal_seconds": round(removal_sec, 3) if removal_sec is not None else None,
        **asdict(cfg),
        **cc_metrics,
    }

    run_summary = {
        "metrics": metrics,
        "top_duplicate_clusters": top_clusters,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fuzzy dedup range sweeps on HF datasets (Colab/GPU).")
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id to study.")
    parser.add_argument("--split", default="train", choices=["train", "eval"], help="Split to load.")
    parser.add_argument(
        "--row-ranges",
        default="0:200000",
        help="Comma-separated ranges start:end. Example: 0:200000,200000:400000",
    )
    parser.add_argument("--char-ngrams", default="24", help="Ints and ranges. Example: 20:28:4,32")
    parser.add_argument("--minhashes-per-band", default="13", help="Ints/ranges. Example: 11,13,15")
    parser.add_argument("--bands-per-iteration", default="10", help="Ints/ranges. Example: 5,10,15")
    parser.add_argument(
        "--input-blocksizes",
        default="512MiB",
        help="Comma-separated block sizes. Example: 512MiB,1GiB",
    )
    parser.add_argument("--output-root", type=Path, default=Path("./fuzzy_sweep_outputs"))
    parser.add_argument("--text-field", default="text", help="Text field name; falls back to tokenized_text join.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--top-k-clusters", type=int, default=5)
    parser.add_argument("--examples-per-cluster", type=int, default=3)
    parser.add_argument("--run-removal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap for number of combinations (0=all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LOGURU_LEVEL", "ERROR")

    print(f"[info] Loading dataset: {args.repo_id} split={args.split}")
    ds = load_dataset(args.repo_id, split=args.split)
    total_rows = len(ds)
    row_ranges = parse_row_ranges(args.row_ranges, max_len=total_rows)

    char_ngrams_vals = parse_int_values(args.char_ngrams)
    minhashes_per_band_vals = parse_int_values(args.minhashes_per_band)
    bands_per_iteration_vals = parse_int_values(args.bands_per_iteration)
    blocksizes = parse_str_values(args.input_blocksizes)

    cfgs: list[SweepConfig] = []
    for start, end in row_ranges:
        for cg, mhb, bpi, blocksize in itertools.product(
            char_ngrams_vals, minhashes_per_band_vals, bands_per_iteration_vals, blocksizes
        ):
            cfgs.append(
                SweepConfig(
                    range_start=start,
                    range_end=end,
                    char_ngrams=cg,
                    minhashes_per_band=mhb,
                    bands_per_iteration=bpi,
                    input_blocksize=blocksize,
                )
            )
    if args.max_runs > 0:
        cfgs = cfgs[: args.max_runs]

    print(f"[info] Planned runs: {len(cfgs)}")
    for c in cfgs:
        print(f"  - {c}")
    if args.dry_run:
        print("[done] Dry run only. Exiting.")
        return

    ensure_nemo_deps()
    import torch
    from nemo_curator.core.client import RayClient

    if torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError(
            f"Requested num_gpus={args.num_gpus} but only {torch.cuda.device_count()} visible."
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    io_kwargs = None

    range_cache: dict[tuple[int, int], pd.DataFrame] = {}
    for start, end in row_ranges:
        range_dir = args.output_root / f"range_{start}_{end}"
        range_input_dir = range_dir / "materialized_input"
        df = materialize_range_to_parquet(
            ds=ds,
            start=start,
            end=end,
            parquet_dir=range_input_dir,
            text_field=args.text_field,
        )
        range_cache[(start, end)] = df
        print(f"[info] Materialized range [{start}, {end}) with {len(df)} rows")

    client = RayClient(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    client.start()
    all_metrics: list[dict[str, Any]] = []
    try:
        for i, cfg in enumerate(cfgs, start=1):
            print(f"[run {i}/{len(cfgs)}] {cfg}")
            input_df = range_cache[(cfg.range_start, cfg.range_end)]
            metrics = run_single_sweep(
                cfg=cfg,
                output_root=args.output_root,
                input_df=input_df,
                io_kwargs=io_kwargs,
                seed=args.seed,
                num_top_clusters=args.top_k_clusters,
                examples_per_cluster=args.examples_per_cluster,
                run_removal=args.run_removal,
            )
            all_metrics.append(metrics)
            print(
                f"[run {i}] duplicate_rate={metrics['duplicate_rate_pct']}% "
                f"largest_cluster={metrics['largest_cluster_size']}"
            )
    finally:
        client.stop()

    summary_csv = args.output_root / "sweep_summary.csv"
    summary_json = args.output_root / "sweep_summary.json"
    pd.DataFrame(all_metrics).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] Wrote {summary_csv}")
    print(f"[done] Wrote {summary_json}")


if __name__ == "__main__":
    main()

