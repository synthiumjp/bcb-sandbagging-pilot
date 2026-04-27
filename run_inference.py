"""
BCB Pilot — Inference harness.
Runs each model through all conditions on sampled items.
Condition A runs first (pre-registered requirement).

Outputs: results/{model_id}_{condition}.jsonl

Usage:
    python run_inference.py --model qwen2.5-7b --condition A
    python run_inference.py --model qwen2.5-7b --condition B
    python run_inference.py --model qwen2.5-7b --condition C
    python run_inference.py --model qwen2.5-7b --condition D
    python run_inference.py --model qwen2.5-7b --condition all
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    MODELS, SYSTEM_PROMPTS, USER_TEMPLATE, DOMAINS,
    TEMPERATURE, MAX_TOKENS, N_GPU_LAYERS, CONTEXT_SIZE,
    RESULTS_DIR, DATA_DIR
)


def load_sampled_items():
    """Load pre-sampled items from JSONL."""
    path = os.path.join(DATA_DIR, "sampled_items.jsonl")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run sample_items.py first.")
        sys.exit(1)
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


def load_model(model_id):
    """Load a GGUF model via llama-cpp-python with Vulkan backend."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed.")
        print("  pip install llama-cpp-python")
        sys.exit(1)

    model_cfg = MODELS[model_id]
    gguf_path = model_cfg["gguf_path"]

    if gguf_path is None or not os.path.exists(gguf_path):
        # Try local_config override
        try:
            from configs.local_config import MODEL_PATHS
            gguf_path = MODEL_PATHS.get(model_id)
        except ImportError:
            pass

    if gguf_path is None or not os.path.exists(gguf_path):
        print(f"ERROR: GGUF path not set for {model_id}.")
        print(f"  Create configs/local_config.py with:")
        print(f"  MODEL_PATHS = {{")
        print(f'      "qwen2.5-7b": "/path/to/qwen2.5-7b-instruct-q5_k_m.gguf",')
        print(f'      "llama3-8b": "/path/to/llama-3-8b-instruct-q5_k_m.gguf",')
        print(f'      "phi3.5-mini": "/path/to/phi-3.5-mini-instruct-q5_k_m.gguf",')
        print(f"  }}")
        sys.exit(1)

    print(f"Loading {model_cfg['name']} from {gguf_path}...")
    t0 = time.time()
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=CONTEXT_SIZE,
        verbose=False,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return llm


def format_system_prompt(condition, domain, c_subprompt=None):
    """Get the system prompt for a given condition and domain."""
    if condition == "C":
        key = c_subprompt  # "C1", "C2", or "C3"
    else:
        key = condition

    prompt = SYSTEM_PROMPTS[key]
    return prompt.format(domain=domain)


def format_user_message(item):
    """Format the user message with question and options."""
    return USER_TEMPLATE.format(
        question=item["question"],
        options=item["options_formatted"],
    )


def run_single_trial(llm, system_prompt, user_message, temperature=TEMPERATURE):
    """Run a single trial and return the raw output."""
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        raw_output = response["choices"][0]["message"]["content"]
        return raw_output.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def parse_response(raw_output):
    """
    Parse model output to extract a single letter A-J.
    Returns (parsed_letter, is_valid).
    """
    if not raw_output or raw_output.startswith("ERROR:"):
        return None, False

    text = raw_output.strip()

    # Try exact single character
    if len(text) == 1 and text.upper() in "ABCDEFGHIJ":
        return text.upper(), True

    # Try first character if followed by punctuation or whitespace
    if len(text) >= 1 and text[0].upper() in "ABCDEFGHIJ":
        if len(text) == 1 or text[1] in ".,:;) \t\n":
            return text[0].upper(), True

    # Try regex for common patterns like "The answer is B" or "B."
    match = re.search(r'\b([A-J])\b', text.upper())
    if match:
        # Only accept if it's a single isolated letter
        found = match.group(1)
        # Check this isn't part of a word
        start = match.start()
        end = match.end()
        before_ok = (start == 0 or not text[start - 1].isalpha())
        after_ok = (end == len(text) or not text[end].isalpha()
                    if end < len(text) else True)
        if before_ok and after_ok:
            return found, True

    return None, False


def get_conditions_to_run(condition_arg):
    """Expand 'all' into ordered condition list. A must run first."""
    if condition_arg == "all":
        return ["A", "B", "C", "D"]
    return [condition_arg]


def run_condition(llm, model_id, condition, items, temperature=TEMPERATURE):
    """Run all items for one model × one condition."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    suffix = f"_T{temperature}" if temperature != TEMPERATURE else ""
    output_path = os.path.join(
        RESULTS_DIR, f"{model_id}_{condition}{suffix}.jsonl"
    )

    # Check for existing results (resume support)
    completed_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                completed_ids.add(rec["item_id"])
        print(f"  Resuming: {len(completed_ids)} trials already completed.")

    # Filter to relevant items
    if condition == "C":
        # All items, but each uses its assigned C sub-prompt
        relevant = items
    else:
        relevant = items

    remaining = [it for it in relevant if it["item_id"] not in completed_ids]
    total = len(relevant)
    done = len(completed_ids)

    print(f"  Running {model_id} / Condition {condition} "
          f"({done}/{total} done, {len(remaining)} remaining)")

    with open(output_path, "a") as f:
        for i, item in enumerate(remaining):
            # Get system prompt
            system_prompt = format_system_prompt(
                condition, item["domain"],
                c_subprompt=item.get("c_subprompt")
            )
            user_message = format_user_message(item)

            # Run trial
            t0 = time.time()
            raw_output = run_single_trial(llm, system_prompt, user_message,
                                          temperature=temperature)
            elapsed = time.time() - t0

            # Parse
            parsed_letter, is_valid = parse_response(raw_output)

            # Score
            correct = None
            if is_valid and parsed_letter is not None:
                correct = int(parsed_letter == item["answer"])

            # Record
            record = {
                "model_id": model_id,
                "domain": item["domain"],
                "condition": condition,
                "c_subprompt": item.get("c_subprompt") if condition == "C" else None,
                "item_id": item["item_id"],
                "question": item["question"][:100] + "...",  # truncated for space
                "answer_key": item["answer"],
                "raw_output": raw_output,
                "parsed_response": parsed_letter,
                "is_valid": is_valid,
                "correct": correct,
                "temperature": temperature,
                "elapsed_s": round(elapsed, 3),
                "timestamp": datetime.utcnow().isoformat(),
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            done += 1
            if done % 50 == 0 or done == total:
                print(f"    [{done}/{total}] "
                      f"last: {parsed_letter} ({'correct' if correct else 'wrong/missing'}) "
                      f"({elapsed:.2f}s)")

    print(f"  Condition {condition} complete. Output: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="BCB Pilot — Inference")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Model ID")
    parser.add_argument("--condition", required=True,
                        choices=["A", "B", "C", "D", "all"],
                        help="Condition (or 'all' to run A→B→C→D in order)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Temperature (default: {TEMPERATURE})")
    args = parser.parse_args()

    # Load items
    items = load_sampled_items()
    print(f"Loaded {len(items)} sampled items across "
          f"{len(set(it['domain'] for it in items))} domains.")

    # Load model
    llm = load_model(args.model)

    # Run conditions
    conditions = get_conditions_to_run(args.condition)
    for cond in conditions:
        run_condition(llm, args.model, cond, items, args.temperature)

    print("\nAll done.")


if __name__ == "__main__":
    main()
