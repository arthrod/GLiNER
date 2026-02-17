"""Generate a large JSONL dataset, inject noise, and validate.

Usage:
    python ptbr/tests/generate_noisy_jsonl.py

Creates:
    ptbr/tests/base_valid.jsonl   -- 50 000 clean entries
    ptbr/tests/noisy.jsonl        -- same data with ~30 % corrupted
    Prints a full validation report at the end.
"""

import copy
import json
import random
import sys
import time
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ptbr import validate_data

# ---------------------------------------------------------------------------
# Vocabulary pools for generating realistic-looking valid entries
# ---------------------------------------------------------------------------
NAMES = [
    "Maria", "Joao", "Pedro", "Ana", "Carlos", "Fernanda", "Lucas", "Julia",
    "Rafael", "Beatriz", "Gustavo", "Camila", "Thiago", "Larissa", "Diego",
    "Mariana", "Bruno", "Isabela", "Felipe", "Leticia", "Matheus", "Gabriela",
]
CITIES = [
    "Sao Paulo", "Rio de Janeiro", "Brasilia", "Salvador", "Fortaleza",
    "Belo Horizonte", "Manaus", "Curitiba", "Recife", "Porto Alegre",
    "Belem", "Goiania", "Guarulhos", "Campinas", "Sao Luis", "Natal",
]
ORGS = [
    "Petrobras", "Itau", "Bradesco", "Vale", "Embraer", "Natura",
    "Magazine Luiza", "Globo", "BNDES", "Fiocruz", "USP", "UNICAMP",
]
VERBS = ["visited", "founded", "joined", "managed", "reported", "announced"]
FILLERS = ["the", "a", "in", "of", "and", "with", "for", "on", "at", "to"]
LABELS_NER = [
    "Person", "City", "Organization", "Date", "Country", "Event",
    "Product", "Money", "Percent", "Location", "Time", "Quantity",
]
RELATION_TYPES = ["located_in", "works_for", "born_in", "CEO_of", "founded_by"]

# ---------------------------------------------------------------------------
# Random valid entry generator
# ---------------------------------------------------------------------------

def random_valid_entry(idx: int) -> dict:
    """Generate a single valid GLiNER entry with realistic content."""
    # Build a sentence
    name = random.choice(NAMES)
    verb = random.choice(VERBS)
    city = random.choice(CITIES).split()
    org = random.choice(ORGS).split()
    filler1 = random.choice(FILLERS)
    filler2 = random.choice(FILLERS)

    tokens = [name, verb, filler1, *org, filler2, *city, "."]
    n = len(tokens)

    name_end = 0
    org_start, org_end = 3, 3 + len(org) - 1
    city_start, city_end = org_end + 2, org_end + 1 + len(city)

    spans = [[0, name_end, random.choice(["Person", "Name"])]]
    if org_end < n:
        spans.append([org_start, org_end, "Organization"])
    if city_end < n:
        spans.append([city_start, city_end, "City"])

    entry = {"id": f"gen-{idx:06d}", "tokenized_text": tokens, "ner": spans}

    # Occasionally add relations
    if random.random() < 0.15 and len(spans) >= 2:
        entry["relations"] = [[0, 1, random.choice(RELATION_TYPES)]]

    return entry


# ---------------------------------------------------------------------------
# 30 noise injection functions -- each corrupts exactly one thing
# ---------------------------------------------------------------------------

def noise_text_as_string(e):
    e["tokenized_text"] = " ".join(e["tokenized_text"])

def noise_text_has_int(e):
    tokens = e["tokenized_text"]
    if tokens:
        tokens[random.randrange(len(tokens))] = random.randint(0, 999)

def noise_text_has_none(e):
    tokens = e["tokenized_text"]
    if tokens:
        tokens[random.randrange(len(tokens))] = None

def noise_text_has_float(e):
    tokens = e["tokenized_text"]
    if tokens:
        tokens[random.randrange(len(tokens))] = 3.14

def noise_text_is_dict(e):
    e["tokenized_text"] = {"tokens": e["tokenized_text"]}

def noise_text_missing(e):
    del e["tokenized_text"]

def noise_text_is_none(e):
    e["tokenized_text"] = None

def noise_ner_missing(e):
    del e["ner"]

def noise_ner_is_dict(e):
    e["ner"] = {str(i): s for i, s in enumerate(e["ner"])}

def noise_ner_is_string(e):
    e["ner"] = str(e["ner"])

def noise_ner_is_none(e):
    e["ner"] = None

def noise_ner_is_int(e):
    e["ner"] = 0

def noise_span_too_short(e):
    if e["ner"]:
        e["ner"][0] = e["ner"][0][:2]  # drop label

def noise_span_too_long(e):
    if e["ner"]:
        e["ner"][0] = e["ner"][0] + [0.95]  # add confidence

def noise_span_as_string(e):
    if e["ner"]:
        s = e["ner"][0]
        e["ner"][0] = f"{s[0]},{s[1]},{s[2]}"

def noise_span_as_dict(e):
    if e["ner"]:
        s = e["ner"][0]
        e["ner"][0] = {"start": s[0], "end": s[1], "label": s[2]}

def noise_float_indices(e):
    if e["ner"]:
        s = e["ner"][0]
        e["ner"][0] = [float(s[0]), float(s[1]), s[2]]

def noise_negative_index(e):
    if e["ner"]:
        e["ner"][0][0] = -random.randint(1, 5)

def noise_start_gt_end(e):
    if e["ner"]:
        s = e["ner"][0]
        e["ner"][0] = [s[1] + 1, s[0], s[2]]

def noise_oob_index(e):
    if e["ner"]:
        n = len(e["tokenized_text"])
        e["ner"][0][1] = n + random.randint(0, 10)

def noise_label_as_int(e):
    if e["ner"]:
        e["ner"][0][2] = random.randint(0, 99)

def noise_label_as_none(e):
    if e["ner"]:
        e["ner"][0][2] = None

def noise_label_as_list(e):
    if e["ner"]:
        lbl = e["ner"][0][2]
        e["ner"][0][2] = [lbl, "extra"]

def noise_label_as_bool(e):
    if e["ner"]:
        e["ner"][0][2] = True

def noise_item_is_list(e):
    """Returns a list instead of dict -- special handling needed."""
    return [e.get("tokenized_text", []), e.get("ner", [])]

def noise_item_is_string(e):
    """Returns a string instead of dict -- special handling needed."""
    return json.dumps(e)

def noise_relations_oob(e):
    e.setdefault("relations", [])
    e["relations"] = [[0, 99, "bad_rel"]]

def noise_relations_type_int(e):
    e.setdefault("relations", [])
    e["relations"] = [[0, 0, 42]]

def noise_relations_wrong_shape(e):
    e.setdefault("relations", [])
    e["relations"] = [[0, 1]]

def noise_relations_is_string(e):
    e["relations"] = "bad"


ALL_NOISE = [
    ("text_as_string",       noise_text_as_string),
    ("text_has_int",         noise_text_has_int),
    ("text_has_none",        noise_text_has_none),
    ("text_has_float",       noise_text_has_float),
    ("text_is_dict",         noise_text_is_dict),
    ("text_missing",         noise_text_missing),
    ("text_is_none",         noise_text_is_none),
    ("ner_missing",          noise_ner_missing),
    ("ner_is_dict",          noise_ner_is_dict),
    ("ner_is_string",        noise_ner_is_string),
    ("ner_is_none",          noise_ner_is_none),
    ("ner_is_int",           noise_ner_is_int),
    ("span_too_short",       noise_span_too_short),
    ("span_too_long",        noise_span_too_long),
    ("span_as_string",       noise_span_as_string),
    ("span_as_dict",         noise_span_as_dict),
    ("float_indices",        noise_float_indices),
    ("negative_index",       noise_negative_index),
    ("start_gt_end",         noise_start_gt_end),
    ("oob_index",            noise_oob_index),
    ("label_as_int",         noise_label_as_int),
    ("label_as_none",        noise_label_as_none),
    ("label_as_list",        noise_label_as_list),
    ("label_as_bool",        noise_label_as_bool),
    ("item_is_list",         noise_item_is_list),
    ("item_is_string",       noise_item_is_string),
    ("relations_oob",        noise_relations_oob),
    ("relations_type_int",   noise_relations_type_int),
    ("relations_wrong_shape", noise_relations_wrong_shape),
    ("relations_is_string",  noise_relations_is_string),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TOTAL = 50_000
CORRUPTION_RATE = 0.30

def main():
    random.seed(42)
    out_dir = Path(__file__).resolve().parent
    base_path = out_dir / "base_valid.jsonl"
    noisy_path = out_dir / "noisy.jsonl"

    # ---- Step 1: generate base valid JSONL --------------------------------
    print(f"Generating {TOTAL} valid entries...")
    t0 = time.time()
    base = [random_valid_entry(i) for i in range(TOTAL)]
    with open(base_path, "w", encoding="utf-8") as f:
        for entry in base:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Wrote {base_path}  ({time.time()-t0:.1f}s)")

    # Quick sanity: base should be 100 % valid
    is_valid, errs = validate_data(base)
    assert is_valid, f"Base data has {len(errs)} unexpected errors:\n" + "\n".join(errs[:10])
    print(f"  Base validation: PASSED ({len(base)} entries, 0 errors)")

    # ---- Step 2: inject noise ---------------------------------------------
    num_corrupt = int(TOTAL * CORRUPTION_RATE)
    corrupt_indices = set(random.sample(range(TOTAL), num_corrupt))

    # Track which noise type was applied to each corrupted index
    noise_log = {}  # idx -> noise_type_name

    # Ensure every noise type is used at least once
    noise_queue = list(range(len(ALL_NOISE)))
    random.shuffle(noise_queue)
    corrupt_list = sorted(corrupt_indices)

    # Assign guaranteed-at-least-once for each noise type
    guaranteed = {}
    for ni in range(len(ALL_NOISE)):
        ci = corrupt_list[ni % len(corrupt_list)]
        guaranteed[ci] = ni
    # Fill remaining corruptions with random noise types
    for ci in corrupt_list:
        if ci not in guaranteed:
            guaranteed[ci] = random.randrange(len(ALL_NOISE))

    print(f"Injecting noise into {num_corrupt} entries ({len(ALL_NOISE)} noise types)...")
    t0 = time.time()
    noisy = []
    for i, entry in enumerate(base):
        if i in corrupt_indices:
            noise_idx = guaranteed[i]
            name, fn = ALL_NOISE[noise_idx]
            e = copy.deepcopy(entry)
            result = fn(e)
            if result is not None:
                # noise_item_is_list / noise_item_is_string return a new object
                noisy.append(result)
            else:
                noisy.append(e)
            noise_log[i] = name
        else:
            noisy.append(copy.deepcopy(entry))

    with open(noisy_path, "w", encoding="utf-8") as f:
        for entry in noisy:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Wrote {noisy_path}  ({time.time()-t0:.1f}s)")

    # ---- Step 3: validate the noisy data ----------------------------------
    print(f"\nValidating noisy dataset ({TOTAL} entries, {num_corrupt} corrupted)...")
    t0 = time.time()
    is_valid, errors = validate_data(noisy)
    elapsed = time.time() - t0
    print(f"  Validation completed in {elapsed:.2f}s")
    print(f"  is_valid = {is_valid}")
    print(f"  Total errors found: {len(errors)}")

     # Parse error indices from messages like "[123] ..."
     error_indices = set()
     for err in errors:
         if err.startswith("["):
            # Extract top-level index from "[123] ..." or "[123].ner[0] ..."
            error_indices.add(int(err[1:err.index("]")]))

    detected = error_indices & corrupt_indices
    missed = corrupt_indices - error_indices
    false_positives = error_indices - corrupt_indices

    print(f"\n{'='*60}")
    print(f"  Corrupted entries:     {num_corrupt}")
    print(f"  Detected by validator: {len(detected)}")
    print(f"  Missed (false neg):    {len(missed)}")
    print(f"  False positives:       {len(false_positives)}")
    print(f"{'='*60}")

    # ---- Step 4: per-noise-type breakdown ---------------------------------
    type_counts = {}
    type_detected = {}
    for idx, ntype in noise_log.items():
        type_counts[ntype] = type_counts.get(ntype, 0) + 1
        if idx in error_indices:
            type_detected[ntype] = type_detected.get(ntype, 0) + 1

    print(f"\n{'Noise Type':<28} {'Injected':>8} {'Caught':>8} {'Rate':>8}")
    print("-" * 56)
    all_caught = True
    for ntype in sorted(type_counts):
        injected = type_counts[ntype]
        caught = type_detected.get(ntype, 0)
        rate = caught / injected if injected else 0
        flag = "" if caught == injected else " *** MISSED ***"
        if caught != injected:
            all_caught = False
        print(f"  {ntype:<26} {injected:>8} {caught:>8} {rate:>7.0%}{flag}")

    print()
    if all_caught and not false_positives:
        print("ALL NOISE TYPES CAUGHT. ZERO FALSE POSITIVES.")
    else:
        if missed:
            print(f"WARNING: {len(missed)} corrupted entries were NOT caught!")
            for idx in sorted(missed)[:20]:
                ntype = noise_log[idx]
                print(f"  [{idx}] noise={ntype}  data={json.dumps(noisy[idx])[:120]}")
        if false_positives:
            print(f"WARNING: {len(false_positives)} false positives!")
            for idx in sorted(false_positives)[:20]:
                print(f"  [{idx}] data={json.dumps(noisy[idx])[:120]}")

    # Return exit code
    if missed or false_positives:
        sys.exit(1)


if __name__ == "__main__":
    main()
