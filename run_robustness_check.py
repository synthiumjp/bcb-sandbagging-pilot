"""
run_robustness_check.py

Pre-registered stochastic robustness check.
Triggered by ambiguous H1 result at T=0.

Runs B_original and B_perm at T=0.7 with K=5 samples per item
for Llama-3-8B only. The modal response across 5 samples is used
for the per-item classification.

20,000 total trials (2,000 items x 2 conditions x 5 samples).

Usage:
    python run_robustness_check.py
    python run_robustness_check.py --condition B_original  # one at a time
    python run_robustness_check.py --condition B_perm

Outputs: results/positional_bias/robustness/{model}_{condition}_T0.7.jsonl
Each item appears 5 times (sample_k = 0..4).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    MODELS, SYSTEM_PROMPTS,
    MAX_TOKENS, N_GPU_LAYERS, CONTEXT_SIZE,
    RESULTS_DIR, DATA_DIR
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPTION_LETTERS = list("ABCDEFGHIJ")
MODEL_ID = "llama3-8b"
TEMPERATURE = 0.7
K_SAMPLES = 5
CONDITIONS = ["B_original", "B_perm"]
CONDITION_TO_PROMPT = {"B_original": "B", "B_perm": "B"}
PERM_FILE = os.path.join(DATA_DIR, "permutation_assignments.csv")
OUTPUT_SUBDIR = os.path.join(RESULTS_DIR, "positional_bias", "robustness")


# ---------------------------------------------------------------------------
# Functions (reused from run_positional_bias.py)
# ---------------------------------------------------------------------------
def load_sampled_items():
    path = os.path.join(DATA_DIR, "sampled_items.jsonl")
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    print(f"Loaded {len(items)} items")
    return items


def get_option_texts(item):
    if "options_raw" in item and isinstance(item["options_raw"], list):
        return list(item["options_raw"])
    elif "options" in item and isinstance(item["options"], list):
        return list(item["options"])
    else:
        raise ValueError(f"Item {item.get('item_id', '?')} has no options list")


def format_options_string(option_texts):
    return "\n".join(f"{OPTION_LETTERS[i]}. {text}" for i, text in enumerate(option_texts))


def apply_cyclic_shift(option_texts, k):
    n = len(option_texts)
    return [option_texts[(i - k) % n] for i in range(n)]


def shift_answer_letter(letter, k):
    return OPTION_LETTERS[(OPTION_LETTERS.index(letter) + k) % 10]


def load_model(model_id):
    from llama_cpp import Llama
    model_cfg = MODELS[model_id]
    gguf_path = model_cfg["gguf_path"]
    if gguf_path is None or not os.path.exists(gguf_path):
        try:
            from configs.local_config import MODEL_PATHS
            gguf_path = MODEL_PATHS.get(model_id)
        except ImportError:
            pass
    if gguf_path is None or not os.path.exists(gguf_path):
        print(f"ERROR: GGUF path not set for {model_id}.")
        sys.exit(1)
    print(f"Loading {model_cfg['name']} from {gguf_path}...")
    t0 = time.time()
    llm = Llama(model_path=gguf_path, n_gpu_layers=N_GPU_LAYERS,
                n_ctx=CONTEXT_SIZE, verbose=False)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return llm


def parse_response(raw_output):
    if not raw_output or raw_output.startswith("ERROR:"):
        return None, False
    text = raw_output.strip()
    if len(text) == 1 and text.upper() in "ABCDEFGHIJ":
        return text.upper(), True
    if len(text) >= 1 and text[0].upper() in "ABCDEFGHIJ":
        if len(text) == 1 or text[1] in ".,:;) \t\n":
            return text[0].upper(), True
    match = re.search(r'\b([A-J])\b', text.upper())
    if match:
        found = match.group(1)
        start, end = match.start(), match.end()
        before_ok = (start == 0 or not text[start - 1].isalpha())
        after_ok = (end == len(text) or not text[end].isalpha() if end < len(text) else True)
        if before_ok and after_ok:
            return found, True
    return None, False


def run_single_trial(llm, system_prompt, user_message):
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_condition(llm, condition, items, perm_df):
    os.makedirs(OUTPUT_SUBDIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_SUBDIR, f"{MODEL_ID}_{condition}_T0.7.jsonl")

    is_permuted = condition.endswith("_perm")
    prompt_key = CONDITION_TO_PROMPT[condition]

    # Resume: track (item_id, sample_k) pairs
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                completed.add((rec["item_id"], rec["sample_k"]))
        print(f"  Resuming: {len(completed)} trials already completed.")

    # Build shift lookup
    shift_lookup = {}
    if is_permuted:
        for _, row in perm_df.iterrows():
            shift_lookup[row["item_id"]] = int(row["shift_k"])

    total = len(items) * K_SAMPLES
    done = len(completed)

    print(f"  Running {MODEL_ID} / {condition} T=0.7 K={K_SAMPLES}")
    print(f"  ({done}/{total} done, {total - done} remaining)")

    with open(output_path, "a") as f:
        for item in items:
            item_id = item["item_id"]
            domain = item["domain"]

            option_texts = get_option_texts(item)
            original_answer = item["answer"]
            shift_k = 0

            if is_permuted:
                shift_k = shift_lookup.get(item_id, 0)
                option_texts = apply_cyclic_shift(option_texts, shift_k)
                correct_answer = shift_answer_letter(original_answer, shift_k)
            else:
                correct_answer = original_answer

            system_prompt = SYSTEM_PROMPTS[prompt_key].format(domain=domain)
            options_str = format_options_string(option_texts)
            user_message = f"{item['question']}\n\n{options_str}"

            for sample_k in range(K_SAMPLES):
                if (item_id, sample_k) in completed:
                    continue

                t0 = time.time()
                raw_output = run_single_trial(llm, system_prompt, user_message)
                elapsed = time.time() - t0

                parsed_letter, is_valid = parse_response(raw_output)
                correct = int(parsed_letter == correct_answer) if is_valid and parsed_letter else None

                record = {
                    "model_id": MODEL_ID,
                    "domain": domain,
                    "condition": condition,
                    "item_id": item_id,
                    "sample_k": sample_k,
                    "shift_k": shift_k,
                    "original_answer": original_answer,
                    "answer_key": correct_answer,
                    "raw_output": raw_output,
                    "parsed_response": parsed_letter,
                    "is_valid": is_valid,
                    "correct": correct,
                    "temperature": TEMPERATURE,
                    "elapsed_s": round(elapsed, 3),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                f.write(json.dumps(record) + "\n")
                f.flush()

                done += 1
                if done % 250 == 0 or done == total:
                    status = "correct" if correct else "wrong/missing"
                    print(f"    [{done}/{total}] item={item_id} k={sample_k} "
                          f"resp={parsed_letter} ({status}) ({elapsed:.2f}s)")

    print(f"  {condition} complete. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stochastic robustness check — T=0.7, K=5, Llama only"
    )
    parser.add_argument(
        "--condition", default="all",
        choices=["B_original", "B_perm", "all"],
        help="Condition to run (default: all = B_original then B_perm)"
    )
    args = parser.parse_args()

    items = load_sampled_items()

    perm_df = pd.read_csv(PERM_FILE)
    print(f"Loaded {len(perm_df)} permutation assignments")

    llm = load_model(MODEL_ID)

    conditions = CONDITIONS if args.condition == "all" else [args.condition]
    print(f"\nRobustness check: {' -> '.join(conditions)}")
    print(f"Temperature: {TEMPERATURE}, K={K_SAMPLES}")
    print(f"Total trials: {len(items) * K_SAMPLES * len(conditions)}\n")

    for cond in conditions:
        run_condition(llm, cond, items, perm_df)

    print("\nRobustness check complete.")


if __name__ == "__main__":
    main()
