from datasets import DatasetDict, load_dataset
from tqdm import tqdm
import glob
import json

def gliner_data_format(data: list[dict]) -> list[dict]:
    formatted_data = []
    for item in data:
        formatted_data.append({
            "tokenized_text": item["tokens"],
            "ner": [[p['start'], p['end'] - 1, p['label']] for p in item["token_spans"]]
        })
    return formatted_data

if __name__ == "__main__":
    with open("/vol/tmp/goldejon/ner/data/conll2003/validation.jsonl", "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    
    data = gliner_data_format(data)

    with open(f"/vol/tmp/goldejon/multilingual_ner/data/gliner_format/conll2003_validation.json", "w") as f:
        json.dump(data, f)