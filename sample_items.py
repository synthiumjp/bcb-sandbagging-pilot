"""
BCB Pilot — Item sampler.
Downloads MMLU-Pro, filters to 10-option items in target domains,
samples 500 per domain (seed=42), assigns Condition C sub-prompts
via block-randomisation within domain.

Run once before inference. Outputs: data/sampled_items.jsonl
"""

import json
import random
import os
import sys

# Add parent to path for config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    SEED, DOMAINS, ITEMS_PER_DOMAIN, REQUIRED_NUM_OPTIONS, DATA_DIR
)


def load_mmlu_pro():
    """Load MMLU-Pro test split from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        return ds
    except ImportError:
        print("ERROR: `datasets` package not installed.")
        print("  pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        print("If network-restricted, place a local copy at data/mmlu_pro_test.jsonl")
        print("with fields: question_id, question, options, answer, category")
        sys.exit(1)


def load_local_fallback(path):
    """Load from a local JSONL file if HF download fails."""
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


def filter_and_sample(dataset):
    """
    Filter to 10-option items in target domains, sample 500 per domain.
    Returns list of dicts with item metadata and C sub-prompt assignment.
    """
    rng = random.Random(SEED)

    # Group by domain, filter to 10-option
    domain_items = {d: [] for d in DOMAINS}

    for item in dataset:
        cat = item["category"].lower().strip()
        if cat not in DOMAINS:
            continue
        # Filter: exactly 10 options
        opts = item["options"]
        if len(opts) != REQUIRED_NUM_OPTIONS:
            continue
        domain_items[cat].append(item)

    sampled = []
    stats = {}

    for domain in DOMAINS:
        pool = domain_items[domain]
        # Sort by question_id (or index) for deterministic ordering
        # MMLU-Pro uses 'question_id' field
        if pool and "question_id" in pool[0]:
            pool.sort(key=lambda x: x["question_id"])
        else:
            # Fallback: sort by question text hash for determinism
            pool.sort(key=lambda x: hash(x["question"]))

        n_available = len(pool)
        n_sample = min(ITEMS_PER_DOMAIN, n_available)

        if n_available < ITEMS_PER_DOMAIN:
            print(f"  WARNING: {domain} has {n_available} items "
                  f"(< {ITEMS_PER_DOMAIN}). Using all available.")

        selected = rng.sample(pool, n_sample)
        # Re-sort selected items by question_id for C sub-prompt assignment
        if selected and "question_id" in selected[0]:
            selected.sort(key=lambda x: x["question_id"])
        else:
            selected.sort(key=lambda x: hash(x["question"]))

        # Block-randomise C sub-prompt assignment within domain (seed=42)
        c_rng = random.Random(SEED)
        c_labels = ["C1", "C2", "C3"] * ((n_sample // 3) + 1)
        c_labels = c_labels[:n_sample]
        c_rng.shuffle(c_labels)

        for i, item in enumerate(selected):
            # Format options as "A. option_text\nB. option_text\n..."
            option_letters = [chr(65 + j) for j in range(len(item["options"]))]
            formatted_options = "\n".join(
                f"{letter}. {opt}"
                for letter, opt in zip(option_letters, item["options"])
            )

            sampled.append({
                "item_id": item.get("question_id", f"{domain}_{i}"),
                "domain": domain,
                "question": item["question"],
                "options_raw": item["options"],
                "options_formatted": formatted_options,
                "answer": item["answer"],  # correct answer letter
                "c_subprompt": c_labels[i],
                "item_index_in_domain": i,
            })

        stats[domain] = {
            "pool_size_10opt": n_available,
            "sampled": n_sample,
            "c1_count": c_labels[:n_sample].count("C1"),
            "c2_count": c_labels[:n_sample].count("C2"),
            "c3_count": c_labels[:n_sample].count("C3"),
        }

    return sampled, stats


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "sampled_items.jsonl")

    # Try local fallback first, then HF
    local_path = os.path.join(DATA_DIR, "mmlu_pro_test.jsonl")
    if os.path.exists(local_path):
        print(f"Loading from local file: {local_path}")
        dataset = load_local_fallback(local_path)
    else:
        print("Loading MMLU-Pro from HuggingFace...")
        dataset = load_mmlu_pro()

    print(f"Total items loaded: {len(dataset)}")
    print(f"Target domains: {DOMAINS}")
    print(f"Required options: {REQUIRED_NUM_OPTIONS}")
    print(f"Items per domain: {ITEMS_PER_DOMAIN}")
    print(f"Seed: {SEED}")
    print()

    sampled, stats = filter_and_sample(dataset)

    # Write sampled items
    with open(output_path, "w") as f:
        for item in sampled:
            f.write(json.dumps(item) + "\n")

    # Print summary
    print("=" * 60)
    print("SAMPLING SUMMARY")
    print("=" * 60)
    total = 0
    for domain, s in stats.items():
        print(f"  {domain}:")
        print(f"    10-option pool: {s['pool_size_10opt']}")
        print(f"    Sampled: {s['sampled']}")
        print(f"    C sub-prompts: C1={s['c1_count']}, "
              f"C2={s['c2_count']}, C3={s['c3_count']}")
        total += s["sampled"]
    print(f"  Total sampled items: {total}")
    print(f"  Output: {output_path}")
    print()

    # Write stats for audit trail
    stats_path = os.path.join(DATA_DIR, "sampling_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
