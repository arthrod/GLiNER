import os, json, glob
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
from tqdm import tqdm

@dataclass(frozen=True)
class Span:
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    text: str
    label: str

def iou(a: Span, b: Span) -> float:
    """IoU in token space; only meaningful when labels match (we gate by label elsewhere)."""
    inter = max(0, min(a.token_end, b.token_end) - max(a.token_start, b.token_start))
    union = (a.token_end - a.token_start) + (b.token_end - b.token_start) - inter
    return inter / union if union > 0 else 0.0

def overlaps(a: Span, b: Span) -> bool:
    return max(0, min(a.token_end, b.token_end) - max(a.token_start, b.token_start)) > 0

def to_exclusive(s_incl: int, e_incl: int) -> tuple[int, int]:
    return s_incl, e_incl + 1

def cluster_spans(spans: Iterable[tuple[Span, float]], iou_thresh: float = 0.5) -> dict[Span, list[float]]:
    """
    Merge near-duplicates with SAME label and IoU>=iou_thresh into a canonical span.
    Canonical boundaries = median start/end for stability.
    Returns {canonical_span: [confidences]}.
    """
    items = list(spans)
    used = [False]*len(items)
    clusters = []

    for i, (si, _) in enumerate(items):
        if used[i]:
            continue
        bucket = [i]
        used[i] = True
        for j in range(i+1, len(items)):
            if used[j]: 
                continue
            sj, _ = items[j]
            if si.label == sj.label and iou(si, sj) >= iou_thresh:
                used[j] = True
                bucket.append(j)
        clusters.append(bucket)

    def median_int(vals: list[int]) -> int:
        s = sorted(vals)
        n = len(s)
        return s[n//2] if n % 2 else (s[n//2 - 1] + s[n//2]) // 2

    canon: dict[Span, list[float]] = {}
    for idxs in clusters:
        token_starts = [items[k][0].token_start for k in idxs]
        token_ends   = [items[k][0].token_end   for k in idxs]
        char_starts = [items[k][0].char_start for k in idxs]
        char_ends   = [items[k][0].char_end   for k in idxs]
        texts = {items[k][0].text for k in idxs}
        labels = {items[k][0].label for k in idxs}
        assert len(labels) == 1
        can = Span(median_int(token_starts), median_int(token_ends), median_int(char_starts), median_int(char_ends), next(iter(texts)), next(iter(labels)))
        confs = [items[k][1] for k in idxs]
        canon.setdefault(can, []).extend(confs)
    return canon

def shrink_char_spans_to_gold(gold_spans_with_conf: list, char_spans: list) -> list:
    """
    Given gold spans (subset, same order) and a longer list of predicted char_spans,
    return the subsequence of char_spans that matches the gold sequence by label.

    Example:
      gold labels: ['x','z','x']
      char labels: ['x','y','z','x']  -> returns the items for ['x','z','x']
    """
    if not gold_spans_with_conf:
        return []

    filtered_char_spans = []
    i, j = 0, 0
    n_gold, n_char = len(gold_spans_with_conf), len(char_spans)

    while i < n_gold and j < n_char:
        gold_label = gold_spans_with_conf[i][2]
        char_label = char_spans[j]["tag"]
        if char_label == gold_label:
            filtered_char_spans.append(char_spans[j])  
            i += 1
            j += 1
        else:
            j += 1

    return filtered_char_spans

def priority_gold_merge_singlefold(
    gold_spans_with_conf: list[list],
    tokens: list[str],
    thresholds=(0.99, 0.90, 0.70, 0.50),
    iou_thresh: float = 0.5,
    treat_missing_gold_as: float = 0.0,
):
    """
    Single-prediction-set variant of Option 1 (K=1):
      score(gold-match) = max(gold_conf, pred_conf)
      score(non-gold)   = pred_conf   (agreement==1 if predicted, 0 otherwise)
    Conflict resolution: gold > higher score > longer span > earlier start.
    Returns dict with tiers per threshold and per-span scores.
    """
    # Map each token index to its (char_start, char_end) span in the text, including whitespace after each token
    token_to_char = {}
    char_pos = 0
    for i, token in enumerate(tokens):
        start = char_pos
        end = start + len(token)
        token_to_char[i] = (start, end)
        char_pos = end + 1

    # Parse gold to canonical
    gold_items = []
    for idx, (s_incl, e_incl, label, conf) in enumerate(gold_spans_with_conf):
        s, e = to_exclusive(s_incl, e_incl)
        text = " ".join(tokens[s:e])
        char_start, char_end = token_to_char[s][0], token_to_char[e-1][1]
        cg = float(conf) if conf is not None else treat_missing_gold_as
        gold_items.append((Span(s, e, char_start, char_end, text, label), cg))
    gold_can = cluster_spans(gold_items, iou_thresh=iou_thresh)

    candidates = set(gold_can.keys())
    is_gold = set(gold_can.keys())

    # Scores
    scores: dict[Span, float] = {}
    for z in candidates:
        cg = max(gold_can.get(z, [treat_missing_gold_as]))
        scores[z] = cg

    def select(t: float) -> list[Span]:
        pool = [z for z in candidates if scores[z] >= t]
        pool.sort(key=lambda z: (z in is_gold, scores[z], (z.token_end - z.token_start), -z.token_start), reverse=True)
        chosen: list[Span] = []
        for z in pool:
            conflict = any(overlaps(z, q) and z.label != q.label for q in chosen)
            if not conflict:
                chosen.append(z)
        return chosen

    tiers = {t: select(t) for t in thresholds}

    # Monotonicity sanity
    t_sorted = sorted(thresholds, reverse=True)
    for hi, lo in zip(t_sorted, t_sorted[1:]):
        assert set(tiers[hi]).issubset(set(tiers[lo])), "Non-nested tiers; check inputs."

    # Pack for output
    def span_to_obj(s: Span):
        return {
            "token_start": s.token_start, 
            "token_end": s.token_end, 
            "char_start": s.char_start, 
            "char_end": s.char_end, 
            "label": s.label, 
            "score": scores[s], 
            "text": s.text, 
            "is_gold": s in is_gold
        }

    def remove_overlaps_and_sort(spans_list):
        """Remove overlapping spans, filter out non-gold predictions, and sort by start position."""
        if not spans_list:
            return spans_list
        
        # Filter out non-gold predictions (is_gold == False)
        gold_spans = [span for span in spans_list if span["is_gold"]]
        
        # Sort by start position first
        gold_spans.sort(key=lambda x: x["token_start"])
        
        # Remove overlaps - keep the first span when there's an overlap
        filtered_spans = {"token_spans": [], "char_spans": []}
        for span in gold_spans:
            has_overlap = False
            for existing_span in filtered_spans["token_spans"]:
                # Check if spans overlap (end is exclusive, so end==start is not overlap)
                if (span["token_start"] < existing_span["end"] and 
                    span["token_end"] > existing_span["start"]):
                    has_overlap = True
                    break
            
            if not has_overlap:
                filtered_spans["token_spans"].append({
                    "start": span["token_start"],
                    "end": span["token_end"],
                    "tag": span["label"],
                    "confidence": span["score"],
                    "text": span["text"],
                })
                filtered_spans["char_spans"].append({
                    "start": span["char_start"],
                    "end": span["char_end"],
                    "tag": span["label"],
                    "confidence": span["score"],
                    "text": span["text"],
                })
        
        return filtered_spans

    out = {t: remove_overlaps_and_sort([span_to_obj(s) for s in spans]) for t, spans in tiers.items()}
    return out


def build_splits(
    input_glob: str,
    out_root: str = "out",
    thresholds=(0.99, 0.90, 0.70, 0.50),
    iou_thresh: float = 0.5,
):
    os.makedirs(out_root, exist_ok=True)
    tier_dirs = {t: os.path.join(out_root, f"conf{int(t*100):02d}") for t in thresholds}
    for d in tier_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Buffers per language per tier: we will write one JSONL per language per tier
    buffers: dict[float, dict[str, list[dict]]] = {t: defaultdict(list) for t in thresholds}

    for file in glob.glob(input_glob):
        if "errors" in file:
            continue
        with open(file, "r") as f:
            data = json.load(f)

        for dp in tqdm(data):
            # language from id like "uz_12345"
            lang = dp["id"].split("_")[0]

            gold = dp.get("gold_spans_with_confidence", [])
            tokens = dp.get("tokens", [])

            tiers = priority_gold_merge_singlefold(
                gold_spans_with_conf=gold,
                tokens=tokens,
                thresholds=thresholds,
                iou_thresh=iou_thresh,
            )

            if not tiers[thresholds[0]]:
                del tiers[thresholds[0]]
            if not tiers[thresholds[1]]:
                del tiers[thresholds[1]]
            if not tiers[thresholds[2]]:
                del tiers[thresholds[2]]
            if not tiers[thresholds[3]]:
                del tiers[thresholds[3]]

            for t in tiers.keys():
                rec = {
                    "id": dp["id"],
                    "tokens": tokens,
                    "text": " ".join(tokens),
                    "token_spans": tiers[t]["token_spans"],
                    "char_spans": tiers[t]["char_spans"],
                }
                buffers[t][lang].append(rec)

    # Write JSONL files per tier per language
    for t, langs in buffers.items():
        for lang, records in langs.items():
            out_path = os.path.join(tier_dirs[t], f"{lang}.jsonl")
            with open(out_path, "w", encoding="utf-8") as w:
                for r in records:
                    w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Done. Wrote:", {t: tier_dirs[t] for t in thresholds})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", default="/vol/tmp/goldejon/multilingual_ner/data/confidence_annotations/*.json", help="Glob to input JSON files")
    ap.add_argument("--out_root", default="/vol/tmp/goldejon/multilingual_ner/data/confidence_splits", help="Output root directory")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for merging dup spans")
    args = ap.parse_args()
    build_splits(args.input_glob, out_root=args.out_root, thresholds=(0.99, 0.90, 0.70, 0.50), iou_thresh=args.iou)