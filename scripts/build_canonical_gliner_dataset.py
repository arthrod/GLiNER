#!/usr/bin/env python3
"""Build canonical GLiNER-format datasets and optionally push them to the Hub.

Two datasets are produced:
1) canonical_pii: only positive/annotated PII samples normalized to GLiNER syntax.
2) canonical_pii_ready: canonical_pii + deterministic negative samples.

GLiNER syntax used by this script:
{
  "tokenized_text": ["..."],
  "ner": [[start_token_idx, end_token_idx, "label"], ...]
}
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, whoami


TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S", flags=re.UNICODE)

CANONICAL_SOURCE = "gliner2_pii_ptbr_reward_split"

GLINER2_LABEL_ALIASES = {
    "subject described ethnicity": "race or ethnicity",
}

# Alias-only collapsing to keep GLiNER label flexibility while removing obvious
# duplicate synonyms such as first_name/given name.
NEMOTRON_LABEL_ALIASES = {
    "first_name": "first name",
    "last_name": "last name",
    "date_of_birth": "dob",
    "email": "email address",
    "phone_number": "phone number",
    "street_address": "location full address",
    "city": "location city",
    "state": "location state",
    "postcode": "location zip",
    "credit_debit_card": "credit card",
    "user_name": "user name",
    "gender": "sex or gender",
    "ssn": "social security number",
    "tax_id": "tax id number",
    "national_id": "id card number",
    "political_view": "political view",
    "religious_belief": "religious belief",
    "race_ethnicity": "race or ethnicity",
}

MASKING_LABEL_ALIASES = {
    "GIVENNAME": "first name",
    "SURNAME": "last name",
    "DATEOFBIRTH": "dob",
    "EMAIL": "email address",
    "TELEPHONENUM": "phone number",
    "CITY": "location city",
    "STREET": "location street",
    "BUILDINGNUM": "location building number",
    "ZIPCODE": "location zip",
    "CREDITCARDNUMBER": "credit card",
    "AGE": "age",
    "DATE": "date",
    "TIME": "time",
    "DRIVERLICENSENUM": "driver license number",
    "GENDER": "sex or gender",
    "SEX": "sex or gender",
    "IDCARDNUM": "id card number",
    "PASSPORTNUM": "passport number",
    "SOCIALNUM": "social security number",
    "TAXNUM": "tax id number",
    "TITLE": "title",
    "ACCOUNTNUM": "account number",
    "PASSWORD": "password",
    "USERNAME": "user name",
}


@dataclass(frozen=True)
class SourceFile:
    source: str
    relative_path: str
    split: str
    kind: str
    text_fields: tuple[str, ...] = ()


POSITIVE_FILES: tuple[SourceFile, ...] = (
    SourceFile(CANONICAL_SOURCE, "gliner2_pii_ptbr_reward_split/train.jsonl", "train", "gliner2"),
    SourceFile(CANONICAL_SOURCE, "gliner2_pii_ptbr_reward_split/validation.jsonl", "eval", "gliner2"),
    SourceFile("nemotron_pii", "nemotron_pii/train.jsonl", "train", "nemotron"),
    SourceFile("nemotron_pii", "nemotron_pii/test.jsonl", "eval", "nemotron"),
    SourceFile("open_pii_masking_500k", "open_pii_masking_500k/train.jsonl", "train", "masking"),
    SourceFile("open_pii_masking_500k", "open_pii_masking_500k/validation.jsonl", "eval", "masking"),
    SourceFile("pii_masking_400k", "pii_masking_400k/train.jsonl", "train", "masking"),
    SourceFile("pii_masking_400k", "pii_masking_400k/validation.jsonl", "eval", "masking"),
)

NEGATIVE_FILES_EXPLICIT_SPLIT: tuple[SourceFile, ...] = (
    SourceFile("phishing_drorrabin", "phishing_drorrabin/train.jsonl", "train", "negative", ("text",)),
    SourceFile("phishing_drorrabin", "phishing_drorrabin/test.jsonl", "eval", "negative", ("text",)),
    SourceFile("enron_spam_setfit", "enron_spam_setfit/train.jsonl", "train", "negative", ("text", "message")),
    SourceFile("enron_spam_setfit", "enron_spam_setfit/test.jsonl", "eval", "negative", ("text", "message")),
    SourceFile("spam_messages_mshenoda", "spam_messages_mshenoda/train.jsonl", "train", "negative", ("text",)),
    SourceFile("spam_messages_mshenoda", "spam_messages_mshenoda/validation.jsonl", "eval", "negative", ("text",)),
    SourceFile("spam_messages_mshenoda", "spam_messages_mshenoda/test.jsonl", "eval", "negative", ("text",)),
)

NEGATIVE_FILES_HASH_SPLIT: tuple[SourceFile, ...] = (
    SourceFile("sms_spam_multilingual", "sms_spam_multilingual/train.jsonl", "hash", "negative", ("text",)),
    SourceFile("phishing_darkknight", "phishing_darkknight/train.jsonl", "hash", "negative", ("body",)),
    SourceFile("enron_spam_bvk", "enron_spam_bvk/train.jsonl", "hash", "negative", ("email",)),
    SourceFile("spamassassin", "spamassassin/train.jsonl", "hash", "negative", ("data",)),
    SourceFile("phishing_zefang", "phishing_zefang/train.jsonl", "hash", "negative", ("Email Text",)),
)

FALLBACK_TEXT_FIELDS: tuple[str, ...] = (
    "text",
    "input",
    "source_text",
    "body",
    "message",
    "email",
    "data",
)


class BuildStats:
    def __init__(self) -> None:
        self.rows_seen: Counter[str] = Counter()
        self.rows_written: Counter[str] = Counter()
        self.rows_with_entities: Counter[str] = Counter()
        self.dropped_empty_text: Counter[str] = Counter()
        self.dropped_no_mapped_entities: Counter[str] = Counter()
        self.parse_errors: Counter[str] = Counter()
        self.span_alignment_misses: Counter[str] = Counter()
        self.gliner2_mention_misses: Counter[str] = Counter()
        self.unmapped_labels: dict[str, Counter[str]] = defaultdict(Counter)
        self.contribution: dict[str, dict[str, Counter[str]]] = {
            "canonical_pii": {"train": Counter(), "eval": Counter()},
            "canonical_pii_ready": {"train": Counter(), "eval": Counter()},
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "rows_seen": dict(self.rows_seen),
            "rows_written": dict(self.rows_written),
            "rows_with_entities": dict(self.rows_with_entities),
            "dropped_empty_text": dict(self.dropped_empty_text),
            "dropped_no_mapped_entities": dict(self.dropped_no_mapped_entities),
            "parse_errors": dict(self.parse_errors),
            "span_alignment_misses": dict(self.span_alignment_misses),
            "gliner2_mention_misses": dict(self.gliner2_mention_misses),
            "unmapped_labels_top": {
                src: counts.most_common(30) for src, counts in self.unmapped_labels.items()
            },
            "contribution": {
                ds_name: {
                    split: contribution_rows(counter)
                    for split, counter in split_map.items()
                }
                for ds_name, split_map in self.contribution.items()
            },
        }


def stream_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield idx, None
                continue
            yield idx, row


def tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def char_to_token_span(offsets: list[tuple[int, int]], start: Any, end: Any) -> tuple[int, int] | None:
    try:
        st = int(start)
        en = int(end)
    except (TypeError, ValueError):
        return None

    if en < st:
        st, en = en, st
    if en == st:
        en = st + 1

    idxs = [i for i, (tok_st, tok_en) in enumerate(offsets) if tok_en > st and tok_st < en]
    if not idxs:
        idxs = [i for i, (tok_st, tok_en) in enumerate(offsets) if tok_en > st and tok_st < (en + 1)]
    if not idxs:
        return None
    return idxs[0], idxs[-1]


def choose_text(row: dict[str, Any], preferred_fields: tuple[str, ...]) -> str | None:
    for key in preferred_fields + FALLBACK_TEXT_FIELDS:
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def normalize_label_text(raw_label: str) -> str:
    text = raw_label.strip().replace("_", " ").replace("-", " ").lower()
    return " ".join(text.split())


def normalize_label(raw_label: str, mapping: dict[str, str], allow_fallback: bool = True) -> str | None:
    key = raw_label.strip()
    if not key:
        return None
    if key in mapping:
        return mapping[key]
    upper_key = key.upper()
    if upper_key in mapping:
        return mapping[upper_key]
    lower_key = key.lower()
    if lower_key in mapping:
        return mapping[lower_key]
    if allow_fallback:
        return normalize_label_text(key)
    return None


def dedupe_and_sort_spans(spans: list[tuple[int, int, str]]) -> list[list[Any]]:
    seen: set[tuple[int, int, str]] = set()
    unique: list[tuple[int, int, str]] = []
    for span in spans:
        if span in seen:
            continue
        seen.add(span)
        unique.append(span)
    unique.sort(key=lambda x: (x[0], x[1], x[2]))
    return [[s, e, lbl] for s, e, lbl in unique]


def find_span_for_mention(
    mention: str,
    text_cf: str,
    offsets: list[tuple[int, int]],
    tokens_cf: list[str],
    used_spans: set[tuple[int, int]],
) -> tuple[int, int] | None:
    mention_cf = mention.strip().casefold()
    if not mention_cf:
        return None

    cursor = 0
    while True:
        idx = text_cf.find(mention_cf, cursor)
        if idx < 0:
            break
        cursor = idx + 1
        span = char_to_token_span(offsets, idx, idx + len(mention_cf))
        if span is not None and span not in used_spans:
            return span

    mention_tokens = [tok.casefold() for tok in TOKEN_RE.findall(mention)]
    if not mention_tokens:
        return None
    width = len(mention_tokens)
    max_i = len(tokens_cf) - width + 1
    for i in range(max_i):
        if tokens_cf[i : i + width] == mention_tokens:
            span = (i, i + width - 1)
            if span not in used_spans:
                return span
    return None


def parse_list_maybe(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []


def normalize_gliner2_row(
    row: dict[str, Any],
    canonical_labels: list[str],
    stats: BuildStats,
    source: str,
) -> dict[str, Any] | None:
    text = row.get("input")
    if not isinstance(text, str) or not text.strip():
        stats.dropped_empty_text[source] += 1
        return None

    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        stats.dropped_empty_text[source] += 1
        return None

    entities_obj = row.get("output")
    entity_map = entities_obj.get("entities") if isinstance(entities_obj, dict) else None
    if not isinstance(entity_map, dict):
        entity_map = {}

    text_cf = text.casefold()
    tokens_cf = [tok.casefold() for tok in tokens]
    spans: list[tuple[int, int, str]] = []

    for label in canonical_labels:
        mentions = entity_map.get(label, [])
        if not isinstance(mentions, list):
            continue
        normalized_label = GLINER2_LABEL_ALIASES.get(label, label)
        used_for_label: set[tuple[int, int]] = set()
        for mention in mentions:
            if not isinstance(mention, str):
                continue
            span = find_span_for_mention(mention, text_cf, offsets, tokens_cf, used_for_label)
            if span is None:
                stats.gliner2_mention_misses[source] += 1
                continue
            used_for_label.add(span)
            spans.append((span[0], span[1], normalized_label))

    ner = dedupe_and_sort_spans(spans)
    if not ner:
        stats.dropped_no_mapped_entities[source] += 1
        return None

    return {"tokenized_text": tokens, "ner": ner}


def normalize_char_span_row(
    text: str,
    entries: list[Any],
    label_mapping: dict[str, str],
    source: str,
    stats: BuildStats,
    drop_if_no_entities: bool = True,
    allow_unmapped_fallback: bool = True,
) -> dict[str, Any] | None:
    if not text.strip():
        stats.dropped_empty_text[source] += 1
        return None

    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        stats.dropped_empty_text[source] += 1
        return None

    spans: list[tuple[int, int, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_label = entry.get("label")
        if not isinstance(raw_label, str):
            continue
        canonical_label = normalize_label(raw_label, label_mapping, allow_fallback=allow_unmapped_fallback)
        if canonical_label is None:
            stats.unmapped_labels[source][raw_label] += 1
            continue

        span = char_to_token_span(offsets, entry.get("start"), entry.get("end"))
        if span is None:
            stats.span_alignment_misses[source] += 1
            continue

        spans.append((span[0], span[1], canonical_label))

    ner = dedupe_and_sort_spans(spans)
    if drop_if_no_entities and not ner:
        stats.dropped_no_mapped_entities[source] += 1
        return None

    return {"tokenized_text": tokens, "ner": ner}


def normalize_nemotron_row(row: dict[str, Any], stats: BuildStats, source: str) -> dict[str, Any] | None:
    text = row.get("text")
    if not isinstance(text, str):
        stats.dropped_empty_text[source] += 1
        return None
    spans = parse_list_maybe(row.get("spans"))
    return normalize_char_span_row(
        text=text,
        entries=spans,
        label_mapping=NEMOTRON_LABEL_ALIASES,
        source=source,
        stats=stats,
        drop_if_no_entities=True,
        allow_unmapped_fallback=True,
    )


def normalize_masking_row(row: dict[str, Any], stats: BuildStats, source: str) -> dict[str, Any] | None:
    text = row.get("source_text")
    if not isinstance(text, str):
        stats.dropped_empty_text[source] += 1
        return None
    entries = row.get("privacy_mask")
    entries_list = entries if isinstance(entries, list) else []
    return normalize_char_span_row(
        text=text,
        entries=entries_list,
        label_mapping=MASKING_LABEL_ALIASES,
        source=source,
        stats=stats,
        drop_if_no_entities=True,
        allow_unmapped_fallback=True,
    )


def normalize_negative_row(
    row: dict[str, Any],
    source: str,
    canonical_labels: list[str],
    text_fields: tuple[str, ...],
    stats: BuildStats,
) -> dict[str, Any] | None:
    text = choose_text(row, text_fields)
    if text is None or not text.strip():
        stats.dropped_empty_text[source] += 1
        return None

    tokens, _ = tokenize_with_offsets(text)
    if not tokens:
        stats.dropped_empty_text[source] += 1
        return None

    return {
        "tokenized_text": tokens,
        "ner": [],
        "ner_negatives": canonical_labels,
        "is_negative": True,
    }


def choose_hash_split(source: str, text: str) -> str:
    """
    Deterministically assigns a row to the "train" or "eval" split using a SHA-256 hash of the source and text.
    
    Parameters:
        source (str): Identifier of the data source.
        text (str): The text content used to derive the hash.
    
    Returns:
        split (str): `"eval"` for approximately 10% of inputs (when the hash-based selector equals 0), `"train"` otherwise.
    """
    digest = hashlib.sha256(f"{source}\n{text}".encode("utf-8")).hexdigest()
    return "eval" if (int(digest[:8], 16) % 10 == 0) else "train"


def stable_sample_id(source: str, split: str, row_idx: int, text: str) -> str:
    """
    Produce a deterministic short sample identifier from the source, split, row index, and text.
    
    Returns:
        str: A 20-character hexadecimal identifier derived from the SHA-256 digest of the concatenated inputs.
    """
    digest = hashlib.sha256(f"{source}|{split}|{row_idx}|{text}".encode("utf-8")).hexdigest()
    return digest[:20]


def contribution_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    rows: list[dict[str, Any]] = []
    if total == 0:
        return rows
    for source, count in counter.most_common():
        rows.append(
            {
                "source": source,
                "rows": count,
                "pct": round((100.0 * count) / total, 4),
            }
        )
    return rows


def load_canonical_labels(input_root: Path) -> list[str]:
    paths = (
        input_root / "gliner2_pii_ptbr_reward_split" / "train.jsonl",
        input_root / "gliner2_pii_ptbr_reward_split" / "validation.jsonl",
    )
    labels: set[str] = set()
    for path in paths:
        for _, row in stream_jsonl(path):
            if row is None:
                continue
            output = row.get("output") if isinstance(row, dict) else None
            entities = output.get("entities") if isinstance(output, dict) else None
            if isinstance(entities, dict):
                labels.update(entities.keys())
    if labels:
        return sorted(labels)
    raise RuntimeError("Could not infer canonical labels from gliner2_pii_ptbr_reward_split.")


def jsonl_handles(base_dir: Path) -> tuple[Path, Any, Any]:
    base_dir.mkdir(parents=True, exist_ok=True)
    train_path = base_dir / "train.jsonl"
    eval_path = base_dir / "eval.jsonl"
    train_handle = train_path.open("w", encoding="utf-8")
    eval_handle = eval_path.open("w", encoding="utf-8")
    return base_dir, train_handle, eval_handle


def write_jsonl(handle: Any, item: dict[str, Any]) -> None:
    handle.write(json.dumps(item, ensure_ascii=False))
    handle.write("\n")


def build(
    input_root: Path,
    output_root: Path,
    canonical_repo_id: str,
    ready_repo_id: str,
    push: bool,
    private: bool,
) -> dict[str, Any]:
    stats = BuildStats()
    source_labels = load_canonical_labels(input_root)
    merged_label_inventory: set[str] = set()

    print(f"[info] Source-of-truth labels loaded from {CANONICAL_SOURCE}: {len(source_labels)} labels")

    canonical_dir, canonical_train, canonical_eval = jsonl_handles(output_root / "canonical_pii")
    ready_dir, ready_train, ready_eval = jsonl_handles(output_root / "canonical_pii_ready")

    def handle_for_split(split: str, *, ready_dataset: bool) -> Any:
        if split == "train":
            return ready_train if ready_dataset else canonical_train
        if split == "eval":
            return ready_eval if ready_dataset else canonical_eval
        raise ValueError(f"Unsupported split: {split}")

    # 1) Build positive normalized dataset (and mirror into ready dataset).
    for cfg in POSITIVE_FILES:
        path = input_root / cfg.relative_path
        print(f"[info] Normalizing positive source: {cfg.source} ({cfg.split}) from {path}")
        for row_idx, row in stream_jsonl(path):
            source_key = f"{cfg.source}:{cfg.split}"
            stats.rows_seen[source_key] += 1

            if row is None:
                stats.parse_errors[cfg.source] += 1
                continue

            if cfg.kind == "gliner2":
                normalized = normalize_gliner2_row(row, source_labels, stats, cfg.source)
            elif cfg.kind == "nemotron":
                normalized = normalize_nemotron_row(row, stats, cfg.source)
            elif cfg.kind == "masking":
                normalized = normalize_masking_row(row, stats, cfg.source)
            else:
                raise ValueError(f"Unsupported positive source kind: {cfg.kind}")

            if normalized is None:
                continue

            for _st, _en, lbl in normalized["ner"]:
                merged_label_inventory.add(lbl)

            text_join = " ".join(normalized["tokenized_text"])
            sample_id = stable_sample_id(cfg.source, cfg.split, row_idx, text_join)
            normalized["source"] = cfg.source
            normalized["sample_id"] = sample_id
            normalized["is_negative"] = False

            write_jsonl(handle_for_split(cfg.split, ready_dataset=False), normalized)
            write_jsonl(handle_for_split(cfg.split, ready_dataset=True), normalized)

            stats.rows_written[f"canonical_pii:{cfg.split}"] += 1
            stats.rows_written[f"canonical_pii_ready:{cfg.split}"] += 1
            stats.contribution["canonical_pii"][cfg.split][cfg.source] += 1
            stats.contribution["canonical_pii_ready"][cfg.split][cfg.source] += 1
            if normalized["ner"]:
                stats.rows_with_entities[f"canonical_pii:{cfg.split}"] += 1
                stats.rows_with_entities[f"canonical_pii_ready:{cfg.split}"] += 1

    negative_label_list = sorted(merged_label_inventory) if merged_label_inventory else sorted(source_labels)
    print(f"[info] Final merged positive label inventory: {len(negative_label_list)} labels")

    # 2) Add deterministic negatives to "ready" dataset only.
    for cfg in NEGATIVE_FILES_EXPLICIT_SPLIT + NEGATIVE_FILES_HASH_SPLIT:
        path = input_root / cfg.relative_path
        print(f"[info] Adding negatives from {cfg.source} ({cfg.split}) from {path}")
        for row_idx, row in stream_jsonl(path):
            source_key = f"{cfg.source}:{cfg.split}"
            stats.rows_seen[source_key] += 1
            if row is None:
                stats.parse_errors[cfg.source] += 1
                continue

            normalized = normalize_negative_row(
                row=row,
                source=cfg.source,
                canonical_labels=negative_label_list,
                text_fields=cfg.text_fields,
                stats=stats,
            )
            if normalized is None:
                continue

            split = cfg.split
            if split == "hash":
                split = choose_hash_split(cfg.source, " ".join(normalized["tokenized_text"]))

            sample_id = stable_sample_id(
                cfg.source,
                split,
                row_idx,
                " ".join(normalized["tokenized_text"]),
            )
            normalized["source"] = cfg.source
            normalized["sample_id"] = sample_id

            write_jsonl(handle_for_split(split, ready_dataset=True), normalized)
            stats.rows_written[f"canonical_pii_ready:{split}"] += 1
            stats.contribution["canonical_pii_ready"][split][cfg.source] += 1

    canonical_train.close()
    canonical_eval.close()
    ready_train.close()
    ready_eval.close()

    report = stats.to_json_dict()
    report["source_labels"] = sorted(source_labels)
    report["label_inventory"] = negative_label_list
    report["source_of_truth"] = CANONICAL_SOURCE
    report["datasets"] = {
        "canonical_pii": {
            "train_path": str(canonical_dir / "train.jsonl"),
            "eval_path": str(canonical_dir / "eval.jsonl"),
        },
        "canonical_pii_ready": {
            "train_path": str(ready_dir / "train.jsonl"),
            "eval_path": str(ready_dir / "eval.jsonl"),
        },
    }

    # Simple assessment: cap may be needed if top source exceeds 35% in train split.
    assessment: dict[str, Any] = {}
    for ds_name, split_map in stats.contribution.items():
        train_counter = split_map["train"]
        total_train = sum(train_counter.values())
        top_source = train_counter.most_common(1)[0] if total_train else None
        needs_cap = False
        top_pct = 0.0
        if top_source and total_train:
            top_pct = (100.0 * top_source[1]) / total_train
            needs_cap = top_pct > 35.0
        assessment[ds_name] = {
            "train_total_rows": total_train,
            "top_source": top_source[0] if top_source else None,
            "top_source_rows": top_source[1] if top_source else 0,
            "top_source_pct": round(top_pct, 4),
            "suggest_rebalance_cap": needs_cap,
            "threshold_pct": 35.0,
        }
    report["per_source_assessment"] = assessment

    report_path = output_root / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[info] Build report written to {report_path}")

    if push:
        api = HfApi()
        push_dataset(
            api=api,
            repo_id=canonical_repo_id,
            data_dir=canonical_dir,
            private=private,
            report_path=report_path,
            canonical_labels=negative_label_list,
            title="Flexible GLiNER PII Dataset",
            description="Positive-only normalized dataset with alias-only label merging and broad label flexibility.",
        )
        push_dataset(
            api=api,
            repo_id=ready_repo_id,
            data_dir=ready_dir,
            private=private,
            report_path=report_path,
            canonical_labels=negative_label_list,
            title="Flexible GLiNER PII Dataset (Ready + Deterministic Negatives)",
            description="Alias-merged flexible dataset plus deterministic negative samples.",
        )

    return report


def push_dataset(
    api: HfApi,
    repo_id: str,
    data_dir: Path,
    private: bool,
    report_path: Path,
    canonical_labels: list[str],
    title: str,
    description: str,
) -> None:
    print(f"[info] Pushing dataset to Hub: {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(data_dir / "train.jsonl"),
        path_in_repo="train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(data_dir / "eval.jsonl"),
        path_in_repo="eval.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )

    readme = data_dir / "README.md"
    readme_text = "\n".join(
        [
            f"# {title}",
            "",
            description,
            "",
            "## Format",
            "- `tokenized_text`: token list",
            "- `ner`: list of `[start, end, label]` token spans",
            "- optional `ner_negatives` for deterministic negatives (ready dataset only)",
            "",
            "## Canonical Labels",
            *[f"- `{label}`" for label in canonical_labels],
            "",
            "## Splits",
            "- `train`",
            "- `eval`",
        ]
    )
    readme.write_text(readme_text, encoding="utf-8")

    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(report_path),
        path_in_repo="build_report.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[info] Uploaded README.md and build_report.json to {repo_id}")


def default_repo_ids() -> tuple[str, str]:
    info = whoami()
    username = info.get("name") or info.get("fullname")
    if not username:
        raise RuntimeError("Could not resolve Hugging Face username from whoami().")
    day = date.today().strftime("%Y%m%d")
    canonical = f"{username}/gliner-canonical-ptbr-pii-{day}"
    ready = f"{username}/gliner-canonical-ptbr-pii-ready-{day}"
    return canonical, ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical GLiNER datasets and push private repos to Hugging Face Hub."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("training_feb_22/dataset/preprocessed"),
        help="Input directory containing preprocessed source JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training_feb_22/dataset/normalized"),
        help="Output directory for generated JSONL datasets and report.",
    )
    parser.add_argument(
        "--canonical-repo-id",
        type=str,
        default=None,
        help="Hub repo id for canonical positive-only dataset. Defaults to <user>/gliner-canonical-ptbr-pii-YYYYMMDD.",
    )
    parser.add_argument(
        "--ready-repo-id",
        type=str,
        default=None,
        help="Hub repo id for canonical+negatives dataset. Defaults to <user>/gliner-canonical-ptbr-pii-ready-YYYYMMDD.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Build locally but do not push to the Hub.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Push as public repos (default is private).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    push = not args.no_push
    canonical_repo_id = args.canonical_repo_id
    ready_repo_id = args.ready_repo_id

    if push and (canonical_repo_id is None or ready_repo_id is None):
        canonical_default, ready_default = default_repo_ids()
        canonical_repo_id = canonical_repo_id or canonical_default
        ready_repo_id = ready_repo_id or ready_default

    canonical_repo_id = canonical_repo_id or "<local>"
    ready_repo_id = ready_repo_id or "<local>"

    report = build(
        input_root=args.input_root,
        output_root=args.output_root,
        canonical_repo_id=canonical_repo_id,
        ready_repo_id=ready_repo_id,
        push=push,
        private=not args.public,
    )

    print("[done] Dataset build complete.")
    print(f"[done] Canonical repo: {canonical_repo_id}")
    print(f"[done] Ready repo: {ready_repo_id}")
    print("[done] Per-source assessment:")
    print(json.dumps(report["per_source_assessment"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
