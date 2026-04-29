"""
generate_permutations.py

Generates the cyclic shift assignments for the positional-bias detection study.
For each of the 2,000 items, samples a shift value k ~ Uniform{1,...,9} with seed 84.
Outputs a CSV mapping item_id to shift_k.

Pre-registered: OSF [insert link after registration]
Predecessor pilot: https://osf.io/6zftv/

Usage:
    python generate_permutations.py --items-file sampled_items.csv --output permutation_assignments.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


PERMUTATION_SEED = 84
SHIFT_MIN = 1
SHIFT_MAX = 9
NUM_OPTIONS = 10
OPTION_LETTERS = list("ABCDEFGHIJ")


def generate_shifts(n_items: int, seed: int = PERMUTATION_SEED) -> np.ndarray:
    """Sample cyclic shift values k ~ Uniform{1,...,9} for each item."""
    rng = np.random.default_rng(seed)
    return rng.integers(SHIFT_MIN, SHIFT_MAX + 1, size=n_items)


def apply_cyclic_shift(options: list[str], correct_idx: int, k: int) -> tuple[list[str], int]:
    """
    Apply cyclic shift of k positions to a list of options.
    
    Args:
        options: list of 10 option texts in original order
        correct_idx: index (0-9) of the correct answer in original order
        k: shift value (1-9)
    
    Returns:
        shifted_options: list of 10 option texts after cyclic shift
        new_correct_idx: index of the correct answer after shift
    """
    n = len(options)
    shifted = [options[(i - k) % n] for i in range(n)]
    new_correct_idx = (correct_idx + k) % n
    return shifted, new_correct_idx


def correct_letter_after_shift(original_correct_letter: str, k: int) -> str:
    """Compute the new correct answer letter after cyclic shift."""
    original_idx = OPTION_LETTERS.index(original_correct_letter)
    new_idx = (original_idx + k) % NUM_OPTIONS
    return OPTION_LETTERS[new_idx]


def response_position_after_shift(original_position: str, k: int) -> str:
    """
    Compute where content from original_position ends up after cyclic shift.
    Used for content-tracking classification.
    If model responded at position p under B-original, and content-tracking
    predicts response at (p + k) mod 10 under B-perm.
    """
    original_idx = OPTION_LETTERS.index(original_position)
    shifted_idx = (original_idx + k) % NUM_OPTIONS
    return OPTION_LETTERS[shifted_idx]


def main():
    parser = argparse.ArgumentParser(
        description="Generate cyclic shift permutation assignments for positional-bias study"
    )
    parser.add_argument(
        "--items-file",
        type=str,
        required=True,
        help="Path to the sampled items CSV (must have item_id column)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="permutation_assignments.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PERMUTATION_SEED,
        help=f"Random seed for shift sampling (default: {PERMUTATION_SEED})"
    )
    args = parser.parse_args()

    # Load items (supports both JSONL and CSV)
    items_file = args.items_file
    if items_file.endswith(".jsonl"):
        items = []
        with open(items_file) as f:
            for line in f:
                items.append(json.loads(line.strip()))
        items_df = pd.DataFrame(items)
    else:
        items_df = pd.read_csv(items_file)
    n_items = len(items_df)
    print(f"Loaded {n_items} items from {items_file}")

    # Generate shifts
    shifts = generate_shifts(n_items, seed=args.seed)

    # Build output
    perm_df = pd.DataFrame({
        "item_id": items_df["item_id"] if "item_id" in items_df.columns else range(n_items),
        "shift_k": shifts,
    })

    # Add original and shifted correct answer if available
    answer_col = None
    for col in ["answer", "correct_answer", "answer_key"]:
        if col in items_df.columns:
            answer_col = col
            break

    if answer_col is not None:
        perm_df["original_correct"] = items_df[answer_col].values
        perm_df["shifted_correct"] = [
            correct_letter_after_shift(letter, k)
            for letter, k in zip(items_df[answer_col], shifts)
        ]

    perm_df.to_csv(args.output, index=False)
    print(f"Saved {n_items} permutation assignments to {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Shift distribution: {dict(zip(*np.unique(shifts, return_counts=True)))}")


if __name__ == "__main__":
    main()
