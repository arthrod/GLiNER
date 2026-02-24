"""Export Hugging Face datasets to JSONL/JSON using native Dataset.to_json."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import get_dataset_split_names, load_dataset

DATASETS: dict[str, dict[str, Any]] = {
    #"nemotron_pii": {"repos": ["nvidia/Nemotron-PII"]},
    #"open_pii_masking_500k": {"repos": ["ai4privacy/open-pii-masking-500k-ai4privacy"]},
    #"pii_masking_400k": {"repos": ["ai4privacy/pii-masking-400k"]},
    #"synthetic_pii_mistral_v1": {"repos": ["urchade/synthetic-pii-ner-mistral-v1"]},
    #"lener_br": {"repos": ["peluz/lener_br", "lener_br"]},
    #"harem": {"repos": ["Linguateca/harem", "harem"]},
    #"portuguese_ner": {"repos": ["lfcc/portuguese_ner"]},
    #"sms_spam_multilingual": {
    #    "repos": [
    #        "dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset",
    #        "ucirvine/sms_spam",
    #    ]
    #},
    #"gliner2_pii_ptbr_reward_split": {"repos": ["arthrod/gliner2-pii-ptbr-reward-split"]},
    # --- Spam / phishing carrier-text corpora ---
    "spamassassin": {"repos": ["talby/spamassassin", "bvk/SpamAssassin-spam"]},
    "enron_spam_setfit": {"repos": ["SetFit/enron_spam"]},
    "enron_spam_bvk": {"repos": ["bvk/ENRON-spam"]},
    "phishing_ealvaradob": {"repos": ["ealvaradob/phishing-dataset"], "trust_remote_code": True},
    "phishing_drorrabin": {"repos": ["drorrabin/phishing_emails-data"]},
    "phishing_darkknight": {"repos": ["darkknight25/phishing_benign_email_dataset"]},
    "phishing_zefang": {"repos": ["zefang-liu/phishing-email-dataset"]},
    "spam_messages_mshenoda": {"repos": ["mshenoda/spam-messages"]},
    "sms_spam_multilingual": {
        "repos": [
            "dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset",
            "ucirvine/sms_spam",
        ]
    },
}


def resolve_repo_and_splits(repos: list[str], *, trust_remote_code: bool = False) -> tuple[str, list[str]]:
    last_error: Exception | None = None
    kwargs: dict[str, Any] = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    for repo in repos:
        try:
            splits = get_dataset_split_names(repo, **kwargs)
            return repo, list(splits)
        except Exception as exc:
            last_error = exc
            print(f"[warn] Could not inspect dataset repo '{repo}': {exc}")
    raise RuntimeError(f"Unable to resolve any dataset repo from candidates={repos}: {last_error}")


def export_split(repo: str, split_name: str, output_path: Path, batch_size: int | None, *, trust_remote_code: bool = False) -> int:
    kwargs: dict[str, Any] = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    dataset = load_dataset(repo, split=split_name, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {repo}:{split_name} -> {output_path}")
    return dataset.to_json(str(output_path), batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Hugging Face datasets to JSONL via Dataset.to_json.")
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
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for Dataset.to_json (defaults to HF config default).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    selected = args.dataset or list(DATASETS.keys())

    for key in selected:
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset key: {key}")
        config = DATASETS[key]
        repos = config["repos"]
        trust = config.get("trust_remote_code", False)
        repo, splits = resolve_repo_and_splits(repos, trust_remote_code=trust)
        for split_name in splits:
            output_path = output_dir / key / f"{split_name}.jsonl"
            export_split(repo, split_name, output_path, batch_size=args.batch_size, trust_remote_code=trust)


if __name__ == "__main__":
    main()
