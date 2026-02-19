"""Comprehensive validation tests for ptbr.

Runs every mock file through the validator and checks that:
  - Valid data passes
  - Each invalid mock produces the expected error patterns
  - The CLI works end-to-end
  - JSONL loading works
  - Column remapping works

Usage:
    python ptbr/tests/test_validation.py
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ptbr import GLiNERData, extract_labels, load_data, prepare, validate_data

MOCKS = Path(__file__).resolve().parent / "mocks"
PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def load_mock(filename):
    path = MOCKS / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===================================================================
# 1. Valid data should pass
# ===================================================================

def test_valid_with_extras():
    data = load_mock("valid_with_extras.json")
    ok, errs = validate_data(data)
    report("valid_with_extras.json passes", ok, f"{len(errs)} errors")
    # Extra columns should NOT cause errors
    report("extra columns preserved", all("id" in d for d in data))


def test_sample_data():
    """The repo's own sample_data.json must pass."""
    sample = Path(__file__).resolve().parents[2] / "examples" / "sample_data.json"
    if not sample.exists():
        report("examples/sample_data.json", False, "file not found")
        return
    with open(sample, "r", encoding="utf-8") as f:
        data = json.load(f)
    ok, errs = validate_data(data)
    report("examples/sample_data.json passes", ok, f"{len(data)} entries, {len(errs)} errors")


# ===================================================================
# 2. Invalid mocks -- each must produce errors
# ===================================================================

def test_text_is_raw_string():
    data = load_mock("text_is_raw_string.json")
    ok, errs = validate_data(data)
    report("text_is_raw_string catches all", not ok and len(errs) == 2,
           f"ok={ok}, errors={len(errs)}")
    report("error mentions list of strings", any("list of strings" in e for e in errs))


def test_text_has_mixed_types():
    data = load_mock("text_has_mixed_types.json")
    ok, errs = validate_data(data)
    report("text_has_mixed_types catches all 3", not ok and len(errs) == 3,
           f"ok={ok}, errors={len(errs)}")


def test_indices_are_floats():
    data = load_mock("indices_are_floats.json")
    ok, errs = validate_data(data)
    report("indices_are_floats fails", not ok, f"errors={len(errs)}")
    report("error mentions integers", any("integers" in e for e in errs))


def test_spans_wrong_shape():
    data = load_mock("spans_wrong_shape.json")
    ok, errs = validate_data(data)
    report("spans_wrong_shape catches all 4", not ok and len(errs) == 4,
           f"ok={ok}, errors={len(errs)}")


def test_boundary_violations():
    data = load_mock("boundary_violations.json")
    ok, errs = validate_data(data)
    # 5 entries: start>end, oob, neg start, neg end, oob span
    report("boundary_violations fails", not ok, f"errors={len(errs)}")
    has_gt = any(">" in e and "start" in e for e in errs)
    has_oob = any(">=" in e and "num_tokens" in e for e in errs)
    has_neg = any("non-negative" in e for e in errs)
    report("catches start > end", has_gt)
    report("catches out-of-bounds", has_oob)
    report("catches negative indices", has_neg)


def test_bad_labels():
    data = load_mock("bad_labels.json")
    ok, errs = validate_data(data)
    # 5 entries, each with at least 1 bad label
    report("bad_labels fails", not ok, f"errors={len(errs)}")
    report("catches non-string labels", any("label must be a string" in e for e in errs))
    # int, null, list, bool, float -- should catch all 5 items
    flagged = set()
    for e in errs:
        m = re.match(r"\[(\d+)\]", e)
        if m:
            flagged.add(int(m.group(1)))
    report("all 5 bad-label items flagged", len(flagged) == 5, f"flagged={sorted(flagged)}")


def test_missing_fields():
    data = load_mock("missing_fields.json")
    ok, errs = validate_data(data)
    report("missing_fields fails", not ok, f"errors={len(errs)}")
    # 5 items: no ner, no text, wrong text key, wrong ner key, empty
    flagged = set()
    for e in errs:
        m = re.match(r"\[(\d+)\]", e)
        if m:
            flagged.add(int(m.group(1)))
    report("all 5 items with missing fields flagged", len(flagged) == 5,
           f"flagged={sorted(flagged)}")


def test_ner_wrong_type():
    data = load_mock("ner_wrong_type.json")
    ok, errs = validate_data(data)
    report("ner_wrong_type catches all 4", not ok and len(errs) == 4,
           f"ok={ok}, errors={len(errs)}")


def test_item_wrong_type():
    data = load_mock("item_wrong_type.json")
    ok, errs = validate_data(data)
    # Items 0-3 are not dicts, item 4 is valid
    report("item_wrong_type flags non-dict items", not ok, f"errors={len(errs)}")
    flagged = set()
    for e in errs:
        m = re.match(r"\[(\d+)\]", e)
        if m:
            flagged.add(int(m.group(1)))
    report("exactly items 0-3 flagged", flagged == {0, 1, 2, 3},
           f"flagged={sorted(flagged)}")


def test_relations_bad():
    data = load_mock("relations_bad.json")
    ok, errs = validate_data(data)
    report("relations_bad fails", not ok, f"errors={len(errs)}")
    flagged = set()
    for e in errs:
        m = re.match(r"\[(\d+)\]", e)
        if m:
            flagged.add(int(m.group(1)))
    report("all 4 bad relation items flagged", len(flagged) == 4,
           f"flagged={sorted(flagged)}")


def test_sneaky_mixed():
    """12 entries, rows 2/4/6/8/10 are subtly broken."""
    data = load_mock("sneaky_mixed.json")
    ok, errs = validate_data(data)
    report("sneaky_mixed fails", not ok, f"errors={len(errs)}")
    flagged = set()
    for e in errs:
        m = re.match(r"\[(\d+)\]", e)
        if m:
            flagged.add(int(m.group(1)))
    expected_bad = {2, 4, 6, 8, 10}
    report("catches exactly the 5 bad rows", flagged == expected_bad,
           f"flagged={sorted(flagged)}, expected={sorted(expected_bad)}")
    # Verify the 7 good rows produce NO errors
    good_flagged = flagged - expected_bad
    report("zero false positives on good rows", len(good_flagged) == 0)


def test_text_is_dict():
    data = load_mock("text_is_dict.json")
    ok, errs = validate_data(data)
    report("text_is_dict fails", not ok, f"errors={len(errs)}")


def test_empty_edge_cases():
    data = load_mock("empty_edge_cases.json")
    ok, errs = validate_data(data)
    # item 0: empty tokens + empty ner = OK
    # item 1: empty tokens + ner pointing at index 0 = OOB
    # item 2: single token, ner [0,0] = OK
    # item 3: single token, ner [0,1] = OOB
    report("empty_edge_cases fails", not ok, f"errors={len(errs)}")
    flagged = set()
    for e in errs:
        m = re.match(r"\[(\d+)\]", e)
        if m:
            flagged.add(int(m.group(1)))
    report("catches OOB on empty tokens (item 1)", 1 in flagged)
    report("catches OOB on single token (item 3)", 3 in flagged)
    report("does not flag valid empty (item 0)", 0 not in flagged)
    report("does not flag valid single (item 2)", 2 not in flagged)


# ===================================================================
# 3. JSONL loading
# ===================================================================

def test_jsonl_loading():
    entries = [
        {"tokenized_text": ["Hello", "world"], "ner": [[0, 1, "Greeting"]]},
        {"tokenized_text": ["Foo", "bar", "baz"], "ner": [[0, 0, "Name"]]},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        tmp = f.name
    try:
        data = load_data(tmp)
        ok, _errs = validate_data(data)
        report("JSONL loading works", len(data) == 2 and ok,
               f"loaded={len(data)}, valid={ok}")
    finally:
        os.unlink(tmp)


def test_jsonl_with_noise():
    entries = [
        {"tokenized_text": ["Hello", "world"], "ner": [[0, 1, "Greeting"]]},
        {"tokenized_text": "not a list", "ner": [[0, 1, "Bad"]]},
        {"tokenized_text": ["OK", "entry"], "ner": [[0, 0, "Fine"]]},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
        tmp = f.name
    try:
        data = load_data(tmp)
        ok, errs = validate_data(data)
        report("JSONL noise detected", not ok and len(errs) == 1,
               f"errors={len(errs)}")
        report("error on correct row", errs and errs[0].startswith("[1]"))
    finally:
        os.unlink(tmp)


# ===================================================================
# 4. Column remapping
# ===================================================================

def test_column_remapping():
    entries = [
        {"text": ["Hello", "world"], "entities": [[0, 1, "Greeting"]], "id": 1},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(entries, f)
        tmp = f.name
    try:
        data = load_data(tmp, text_column="text", ner_column="entities")
        ok, _errs = validate_data(data)
        report("column remapping works", ok and "tokenized_text" in data[0],
               f"keys={list(data[0].keys())}")
        report("extra columns preserved after remap", "id" in data[0])
    finally:
        os.unlink(tmp)


def test_column_remapping_missing_custom_column():
    entries = [
        {"text": ["Hello", "world"], "entities": [[0, 1, "Greeting"]], "id": 1},
        {"text": ["No", "entity"], "id": 2},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(entries, f)
        tmp = f.name
    try:
        try:
            load_data(tmp, text_column="text", ner_column="entities")
        except ValueError as exc:
            message = str(exc)
            report("missing remapped column raises ValueError",
                   "Missing required column(s): 'entities'" in message, message)
            report("missing remapped column reports local available columns",
                   "Available columns in local file" in message, message)
        else:
            report("missing remapped column raises ValueError", False, "no exception")
    finally:
        os.unlink(tmp)


# ===================================================================
# 5. prepare() returns GLiNERData
# ===================================================================

def test_prepare_module():
    sample = Path(__file__).resolve().parents[2] / "examples" / "sample_data.json"
    if not sample.exists():
        report("prepare() module test", False, "sample_data.json not found")
        return
    result = prepare(str(sample))
    report("prepare returns GLiNERData", isinstance(result, GLiNERData))
    report("prepare sets is_valid", result.is_valid is True)
    report("prepare extracts labels", len(result.labels) > 0,
           f"{len(result.labels)} labels")
    report("prepare sets source", result.source == str(sample))


# ===================================================================
# 6. CLI smoke test
# ===================================================================

def _run_validate_cli(path):
    if importlib.util.find_spec("typer") is None:
        return None
    return subprocess.run(
        [sys.executable, "-m", "ptbr", "--file-or-repo", str(path), "--validate"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def test_cli_validate():
    sample = Path(__file__).resolve().parents[2] / "examples" / "sample_data.json"
    if not sample.exists():
        report("CLI validate", False, "sample_data.json not found")
        return
    result = _run_validate_cli(sample)
    if result is None:
        report("CLI --validate exits 0 on valid data", True, "skipped: typer not installed")
        return
    report("CLI --validate exits 0 on valid data", result.returncode == 0)


def test_cli_validate_bad():
    bad = MOCKS / "text_is_raw_string.json"
    result = _run_validate_cli(bad)
    if result is None:
        report("CLI --validate exits non-zero on bad data", True, "skipped: typer not installed")
        return
    report("CLI --validate exits non-zero on bad data", result.returncode != 0)


# ===================================================================
# Run all
# ===================================================================

def main():
    print("=" * 60)
    print("ptbr validation test suite")
    print("=" * 60)

    print("\n--- Valid data ---")
    test_valid_with_extras()
    test_sample_data()

    print("\n--- Text field errors ---")
    test_text_is_raw_string()
    test_text_has_mixed_types()
    test_text_is_dict()

    print("\n--- NER field errors ---")
    test_ner_wrong_type()
    test_spans_wrong_shape()
    test_indices_are_floats()
    test_boundary_violations()
    test_bad_labels()
    test_missing_fields()
    test_empty_edge_cases()

    print("\n--- Structural errors ---")
    test_item_wrong_type()
    test_relations_bad()

    print("\n--- Sneaky mixed (5 bad in 12) ---")
    test_sneaky_mixed()

    print("\n--- JSONL loading ---")
    test_jsonl_loading()
    test_jsonl_with_noise()

    print("\n--- Column remapping ---")
    test_column_remapping()
    test_column_remapping_missing_custom_column()

    print("\n--- Module API ---")
    test_prepare_module()

    print("\n--- CLI ---")
    test_cli_validate()
    test_cli_validate_bad()

    print(f"\n{'=' * 60}")
    print(f"  PASSED: {PASS}")
    print(f"  FAILED: {FAIL}")
    print(f"  TOTAL:  {PASS + FAIL}")
    print(f"{'=' * 60}")

    if FAIL:
        sys.exit(1)
    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()
