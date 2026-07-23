import os
import glob
import json
import random

import torch
from tqdm import tqdm


def open_content(path):
    """Load train, dev, test, and label files from a dataset directory.

    Searches for JSON files in the specified directory and loads them based on
    filename patterns (train, dev, test, labels).

    Args:
        path: Path to the directory containing dataset JSON files.

    Returns:
        A tuple of (train, dev, test, labels) where:
        - train: List of training examples loaded from *train*.json, or None if not found
        - dev: List of development examples loaded from *dev*.json, or None if not found
        - test: List of test examples loaded from *test*.json, or None if not found
        - labels: List of entity type labels loaded from *labels*.json, or None if not found

    Note:
        Files are identified by checking if their filename contains 'train', 'dev',
        'test', or 'labels'. All files are expected to be in JSON format with UTF-8 encoding.
    """
    paths = glob.glob(os.path.join(path, "*.json"))
    train, dev, test, labels = None, None, None, None
    for p in paths:
        if "train" in p:
            with open(p, encoding="utf-8") as f:
                train = json.load(f)
        elif "dev" in p:
            with open(p, encoding="utf-8") as f:
                dev = json.load(f)
        elif "test" in p:
            with open(p, encoding="utf-8") as f:
                test = json.load(f)
        elif "labels" in p:
            with open(p, encoding="utf-8") as f:
                labels = json.load(f)
    return train, dev, test, labels


def process(data):
    """Convert character-level entity annotations to word-level annotations.

    Takes a data sample with character-level entity positions and converts them
    to word-level positions by tokenizing the sentence on whitespace.

    Args:
        data: Dictionary containing:
            - 'sentence': String of the full sentence
            - 'entities': List of entity dictionaries, each with:
                - 'pos': Tuple of (start_char, end_char) character positions
                - 'type': String entity type label

    Returns:
        Dictionary containing:
        - 'tokenized_text': List of words from the sentence
        - 'ner': List of tuples (start_word, end_word, entity_type) where
          start_word and end_word are word-level indices and entity_type
          is the lowercased entity type

    Note:
        This function assumes whitespace-separated words and that character
        positions align exactly with word boundaries (including spaces).
    """
    words = data["sentence"].split()
    entities = []  # List of entities (start, end, type)
    entity_indices = {}

    def add_entity(entity):
        """Add an entity once and return its index, or None for invalid offsets."""
        position = entity.get("pos", [])
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            return None

        entity_key = (tuple(position), entity["type"].lower(), entity.get("name"))
        if entity_key in entity_indices:
            return entity_indices[entity_key]

        start_char, end_char = position

        # Initialize variables to keep track of word positions
        start_word = None
        end_word = None

        # Iterate through words and find the word positions
        char_count = 0
        for i, word in enumerate(words):
            word_length = len(word)
            if char_count == start_char:
                start_word = i
            if char_count + word_length == end_char:
                end_word = i
                break
            char_count += word_length + 1  # Add 1 for the space

        # Skip annotations whose character offsets do not match word boundaries.
        # Such annotations cannot be represented as token spans and would later
        # fail when entities are sorted during batching.
        if start_word is None or end_word is None:
            return None

        # Append the word positions to the list
        entity_indices[entity_key] = len(entities)
        entities.append((start_word, end_word, entity["type"].lower()))
        return entity_indices[entity_key]

    for entity in data.get("entities", []):
        add_entity(entity)

    relations = []
    for relation in data.get("relations", []):
        # RE datasets use inline head/tail entity definitions.
        if not isinstance(relation, dict) or "head" not in relation or "tail" not in relation:
            continue
        head_idx = add_entity(relation["head"])
        tail_idx = add_entity(relation["tail"])
        if head_idx is not None and tail_idx is not None:
            relations.append((head_idx, tail_idx, relation["type"].lower()))

    # Create a list of word positions for each entity
    sample = {"tokenized_text": words, "ner": entities}
    if "relations" in data:
        sample["relations"] = relations

    return sample


def create_dataset(path):
    """Create train, dev, and test datasets from a directory of JSON files.

    Loads all dataset splits and processes them to convert character-level
    annotations to word-level annotations. Also normalizes entity type labels
    to lowercase.

    Args:
        path: Path to the directory containing dataset JSON files.

    Returns:
        A tuple of (train_dataset, dev_dataset, test_dataset, labels) where:
        - train_dataset: List of processed training samples
        - dev_dataset: List of processed development samples
        - test_dataset: List of processed test samples
        - labels: List of entity type labels (lowercased)

    Note:
        Each sample in the datasets is a dictionary with 'tokenized_text'
        and 'ner' keys as returned by the process() function.
    """
    train, dev, test, labels = open_content(path)
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for data in train or []:
        train_dataset.append(process(data))
    for data in dev or []:
        dev_dataset.append(process(data))
    for data in test or []:
        test_dataset.append(process(data))
    labels = [label.lower() for label in labels]
    return train_dataset, dev_dataset, test_dataset, labels


@torch.no_grad()
def get_for_one_path(path, model):
    """Evaluate a model on a single dataset.

    Loads the test set from the specified path and evaluates the model's
    performance. Automatically determines whether to use flat NER evaluation
    based on the dataset name.

    Args:
        path: Path to the dataset directory.
        model: NER model instance with an evaluate() method.

    Returns:
        A tuple of (data_name, results, f1) where:
        - data_name: String name of the dataset (extracted from path)
        - results: Detailed evaluation results dictionary from model.evaluate()
        - f1: F1 score (float) for the dataset

    Note:
        Datasets with 'ACE', 'GENIA', or 'Corpus' in their name are evaluated
        with flat_ner=False, all others use flat_ner=True. Evaluation uses
        a threshold of 0.5 and batch size of 12.
    """
    # load the dataset
    _, _, test_dataset, labels = create_dataset(path)
    has_relations = any("relations" in example for example in test_dataset)
    entity_types = sorted({entity[2] for example in test_dataset for entity in example["ner"]}) if has_relations else labels

    data_name = path.split("/")[-1]  # get the name of the dataset

    # check if the dataset is flat_ner
    flat_ner = True
    if any(i in data_name for i in ["ACE", "GENIA", "Corpus"]):
        flat_ner = False

    # evaluate the model
    evaluation_kwargs = {
        "flat_ner": flat_ner,
        "threshold": 0.6,
        "batch_size": 12,
        "entity_types": entity_types,
    }
    if has_relations:
        evaluation_kwargs["relation_types"] = labels
    evaluation = model.evaluate(test_dataset, **evaluation_kwargs)

    # GLiNER NER models return ``(output, f1)``, while the relation-aware
    # variant returns ``((ner_output, ner_f1), (rel_output, rel_f1))``.
    if isinstance(evaluation[0], tuple):
        results, f1 = evaluation[0]
        relation_results, relation_f1 = evaluation[1] if has_relations else (None, None)
    else:
        results, f1 = evaluation
        relation_results, relation_f1 = None, None

    return data_name, results, f1, relation_results, relation_f1


def get_for_all_path(model, steps, log_dir, data_paths):
    """Evaluate a model across multiple datasets and log results.

    Evaluates the model on all datasets in the specified directory, separating
    results into standard benchmarks and zero-shot benchmarks. Writes detailed
    results to log files and computes average scores.

    Args:
        model: NER model instance with an evaluate() method and PyTorch parameters.
        steps: Integer representing the current training step (for logging).
        log_dir: Directory path where result files will be saved.
        data_paths: Path to directory containing multiple dataset subdirectories.

    Note:
        Creates two log files in log_dir:
        - 'results.txt': Detailed results for each dataset
        - 'tables.txt': Formatted tables with averages for benchmarks

        Zero-shot benchmark datasets (not included in main average):
        - mit-movie, mit-restaurant
        - CrossNER_AI, CrossNER_literature, CrossNER_music,
          CrossNER_politics, CrossNER_science

        Datasets with 'sample_' in their path are skipped.
    """
    all_paths = glob.glob(f"{data_paths}/*")

    all_paths = sorted(all_paths)

    # move the model to the device
    device = next(model.parameters()).device
    model.to(device)
    # set the model to eval mode
    model.eval()

    # log the results
    save_path = os.path.join(log_dir, "results.txt")

    with open(save_path, "a") as f:
        f.write("##############################################\n")
        # write step
        f.write("step: " + str(steps) + "\n")

    zero_shot_benc = [
        "mit-movie",
        "mit-restaurant",
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
    ]

    zero_shot_benc_results = {}
    all_results = {}  # without crossNER
    relation_results = {}

    for p in tqdm(all_paths):
        if "sample_" not in p:
            data_name, results, f1, relation_output, relation_f1 = get_for_one_path(p, model)
            # write to file
            with open(save_path, "a") as f:
                f.write(data_name + "\n")
                f.write(str(results) + "\n")
                if relation_output is not None:
                    f.write("Relations: " + str(relation_output) + "\n")

            if data_name in zero_shot_benc:
                zero_shot_benc_results[data_name] = f1
            else:
                all_results[data_name] = f1
            if relation_f1 is not None:
                relation_results[data_name] = relation_f1

    avg_all = sum(all_results.values()) / len(all_results) if all_results else None
    avg_zs = sum(zero_shot_benc_results.values()) / len(zero_shot_benc_results) if zero_shot_benc_results else None
    avg_relations = sum(relation_results.values()) / len(relation_results) if relation_results else None

    save_path_table = os.path.join(log_dir, "tables.txt")

    # results for all datasets except crossNER
    table_bench_all = ""
    for k, v in all_results.items():
        table_bench_all += f"{k:20}: {v:.1%}\n"
    # (20 size aswell for average i.e. :20)
    if avg_all is not None:
        table_bench_all += f"{'Average':20}: {avg_all:.1%}"

    # results for zero-shot benchmark
    table_bench_zeroshot = ""
    for k, v in zero_shot_benc_results.items():
        table_bench_zeroshot += f"{k:20}: {v:.1%}\n"
    if avg_zs is not None:
        table_bench_zeroshot += f"{'Average':20}: {avg_zs:.1%}"

    table_relations = ""
    for k, v in relation_results.items():
        table_relations += f"{k:20}: {v:.1%}\n"
    if avg_relations is not None:
        table_relations += f"{'Average':20}: {avg_relations:.1%}"

    # write to file
    with open(save_path_table, "a") as f:
        f.write("##############################################\n")
        f.write("step: " + str(steps) + "\n")
        f.write("Table for all datasets except crossNER\n")
        f.write(table_bench_all + "\n\n")
        f.write("Table for zero-shot benchmark\n")
        f.write(table_bench_zeroshot + "\n")
        if table_relations:
            f.write("Table for relations\n")
            f.write(table_relations + "\n")
        f.write("##############################################\n\n")


def sample_train_data(data_paths, sample_size=10000):
    """Sample training data from multiple datasets for combined training.

    Creates a combined training set by sampling a fixed number of examples
    from each dataset (excluding zero-shot benchmark datasets). Shuffles
    each dataset before sampling to ensure diversity.

    Args:
        data_paths: Path to directory containing multiple dataset subdirectories.
        sample_size: Maximum number of samples to take from each dataset.
            Defaults to 10000.

    Returns:
        List of training samples, where each sample is a dictionary with:
        - 'tokenized_text': List of words
        - 'ner': List of entity tuples (start, end, type)
        - 'label': List of all entity type labels for this dataset

    Note:
        Excludes zero-shot benchmark datasets:
        - CrossNER_AI, CrossNER_literature, CrossNER_music,
          CrossNER_politics, CrossNER_science, ACE 2004

        Each dataset is shuffled before sampling to ensure random selection.
        If a dataset has fewer than sample_size examples, all examples are used.
    """
    all_paths = glob.glob(f"{data_paths}/*")

    all_paths = sorted(all_paths)

    # to exclude the zero-shot benchmark datasets
    zero_shot_benc = [
        "CrossNER_AI",
        "CrossNER_literature",
        "CrossNER_music",
        "CrossNER_politics",
        "CrossNER_science",
        "ACE 2004",
    ]

    new_train = []
    # take 10k samples from each dataset
    for p in tqdm(all_paths):
        if any(i in p for i in zero_shot_benc):
            continue
        train, _, _, labels = create_dataset(p)

        # add label key to the train data
        for i in range(len(train)):
            train[i]["label"] = labels

        random.shuffle(train)
        train = train[:sample_size]
        new_train.extend(train)

    return new_train
