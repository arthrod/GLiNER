import argparse

from gliner import GLiNER
from gliner.evaluation import get_for_all_path


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--model", type=str, default="models/checkpoint-21000", help="Path to model folder")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to model folder")
    parser.add_argument('--data', type=str, default='data/NER', help='Path to the eval datasets directory')
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # The checkpoint backbone is stored in bfloat16; load every GLiNER layer
    # in the same dtype to avoid mixed bfloat16/float32 operations.
    model = GLiNER.from_pretrained(args.model, load_tokenizer=True, dtype="bf16").to("cuda:0")
    get_for_all_path(model, -1, args.log_dir, args.data)
