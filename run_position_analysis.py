"""
BCB Pilot — Post-hoc Analysis 5: Position-of-correct-answer.

If sandbagging models adopt positional heuristics (e.g., "always pick E"),
then accuracy under Condition B should vary by where the correct answer
falls in the option list. Items where the correct answer coincides with
the model's positional bias should show HIGHER accuracy under B than
items where it doesn't.

This analysis:
  1. Groups items by correct-answer position (A-J)
  2. Computes accuracy under A and B per position per model×domain
  3. Computes the A→B accuracy delta per position
  4. Tests whether the model's modal wrong-answer position under B
     predicts which positions show least suppression (or accidental boost)

Usage:
    python run_position_analysis.py
"""

import json
import os
import sys
import math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import MODELS, DOMAINS, RESULTS_DIR, ANALYSIS_DIR


ALL_LETTERS = [chr(65 + i) for i in range(10)]  # A-J


def load_all():
    """Load all trial data indexed by (model, condition, item_id)."""
    data = {}
    for model_id in MODELS:
        for cond in ["A", "B", "C", "D"]:
            path = os.path.join(RESULTS_DIR, f"{model_id}_{cond}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line.strip())
                    data[(model_id, cond, t["item_id"])] = t
    return data


def position_accuracy_analysis(data):
    """
    For each model × domain, compute accuracy under A and B
    grouped by position of the correct answer (A-J).
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 5: POSITION-OF-CORRECT-ANSWER")
    lines.append("Does accuracy under B depend on where the correct answer falls?")
    lines.append("=" * 70)
    lines.append("")

    for model_id in MODELS:
        lines.append(f"Model: {MODELS[model_id]['name']}")
        lines.append("=" * 60)

        # First, identify the modal B response for this model across domains
        b_response_counts = Counter()
        for key, t in data.items():
            if key[0] == model_id and key[1] == "B" and t["is_valid"]:
                b_response_counts[t["parsed_response"]] += 1
        modal_b = b_response_counts.most_common(3)
        lines.append(f"  Overall modal B responses: "
                     + ", ".join(f"{l}={c}" for l, c in modal_b))
        lines.append("")

        for domain in DOMAINS:
            # Group items by correct answer position
            pos_stats = defaultdict(lambda: {
                "a_total": 0, "a_correct": 0,
                "b_total": 0, "b_correct": 0,
                "b_responses": Counter(),
            })

            # Collect item IDs for this domain
            item_ids = set()
            for key in data:
                if key[0] == model_id and key[1] == "A" and data[key]["domain"] == domain:
                    item_ids.add(key[2])

            for item_id in item_ids:
                a = data.get((model_id, "A", item_id))
                b = data.get((model_id, "B", item_id))
                if not a or not b:
                    continue

                correct_pos = a["answer_key"]  # A-J

                if a["is_valid"]:
                    pos_stats[correct_pos]["a_total"] += 1
                    if a["correct"] == 1:
                        pos_stats[correct_pos]["a_correct"] += 1

                if b["is_valid"]:
                    pos_stats[correct_pos]["b_total"] += 1
                    if b["correct"] == 1:
                        pos_stats[correct_pos]["b_correct"] += 1
                    pos_stats[correct_pos]["b_responses"][b["parsed_response"]] += 1

            # Find modal wrong response for this domain under B
            domain_b_wrong = Counter()
            for key, t in data.items():
                if (key[0] == model_id and key[1] == "B"
                        and t["domain"] == domain and t["is_valid"]
                        and t["correct"] == 0):
                    domain_b_wrong[t["parsed_response"]] += 1
            domain_modal = domain_b_wrong.most_common(1)[0][0] if domain_b_wrong else "?"

            lines.append(f"  {domain} (modal B wrong answer: {domain_modal})")
            lines.append(f"  {'Pos':>4s}  {'n':>4s}  {'Acc_A':>6s}  {'Acc_B':>6s}  "
                        f"{'Delta':>7s}  {'B_chose_this':>12s}  {'Note':>20s}")
            lines.append("  " + "-" * 70)

            deltas = []
            for pos in ALL_LETTERS:
                s = pos_stats.get(pos)
                if not s or s["a_total"] == 0:
                    continue

                a_acc = s["a_correct"] / s["a_total"]
                b_acc = s["b_correct"] / s["b_total"] if s["b_total"] > 0 else 0
                delta = b_acc - a_acc

                # How often did B pick this position (regardless of correctness)?
                b_chose_this = s["b_responses"].get(pos, 0)
                b_chose_rate = b_chose_this / s["b_total"] if s["b_total"] > 0 else 0

                note = ""
                if pos == domain_modal:
                    note = "← modal B target"
                if delta > 0:
                    note = (note + " BOOST").strip()

                lines.append(f"  {pos:>4s}  {s['a_total']:>4d}  {a_acc:>6.3f}  {b_acc:>6.3f}  "
                            f"{delta:>+7.3f}  {b_chose_rate:>12.3f}  {note:>20s}")
                deltas.append((pos, delta, s["a_total"]))

            # Summary: correlation between position and delta
            if deltas:
                # Which position had the least suppression (or a boost)?
                best = max(deltas, key=lambda x: x[1])
                worst = min(deltas, key=lambda x: x[1])
                lines.append(f"  Least suppressed: {best[0]} (Δ={best[1]:+.3f})")
                lines.append(f"  Most suppressed:  {worst[0]} (Δ={worst[1]:+.3f})")
                lines.append(f"  Modal B target coincides with least suppressed: "
                            f"{'YES' if best[0] == domain_modal else 'NO'}")

            lines.append("")

        lines.append("")

    return "\n".join(lines)


def positional_bias_summary(data):
    """
    Summary analysis: across all models, compute the response distribution
    under each condition by POSITION (A-J), ignoring correctness.
    This shows whether models have raw positional preferences.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 5b: RAW POSITIONAL RESPONSE DISTRIBUTION")
    lines.append("How often does each model pick each position, by condition?")
    lines.append("(Uniform = 10% per position)")
    lines.append("=" * 70)
    lines.append("")

    for model_id in MODELS:
        lines.append(f"Model: {MODELS[model_id]['name']}")

        for cond in ["A", "B"]:
            dist = Counter()
            total = 0
            for key, t in data.items():
                if key[0] == model_id and key[1] == cond and t["is_valid"]:
                    dist[t["parsed_response"]] += 1
                    total += 1

            if total == 0:
                continue

            lines.append(f"  Condition {cond} (n={total}):")
            row = "    "
            for letter in ALL_LETTERS:
                pct = dist.get(letter, 0) / total * 100
                row += f"{letter}={pct:5.1f}%  "
            lines.append(row)

            # Entropy
            probs = [dist.get(l, 0) / total for l in ALL_LETTERS if dist.get(l, 0) > 0]
            h = -sum(p * math.log2(p) for p in probs)
            max_h = math.log2(10)
            lines.append(f"    Entropy: {h:.3f}/{max_h:.3f} (normalised: {h/max_h:.3f})")
            lines.append(f"    Max position: {dist.most_common(1)[0][0]} "
                        f"({dist.most_common(1)[0][1]/total:.1%})")

            # Shift from A
            if cond == "A":
                a_dist = dict(dist)
                a_total = total
            elif cond == "B" and a_dist:
                shift = ""
                for letter in ALL_LETTERS:
                    a_pct = a_dist.get(letter, 0) / a_total * 100
                    b_pct = dist.get(letter, 0) / total * 100
                    delta = b_pct - a_pct
                    if abs(delta) > 3:
                        shift += f"{letter}={delta:+.1f}pp  "
                if shift:
                    lines.append(f"    Large shifts A→B (>3pp): {shift}")
                else:
                    lines.append(f"    No large shifts A→B")

        lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("Loading all trial data...")
    data = load_all()
    print(f"  {len(data)} total trial records loaded.")

    print("\nRunning Analysis 5: Position-of-correct-answer...")
    r5 = position_accuracy_analysis(data)

    print("\nRunning Analysis 5b: Raw positional distribution...")
    r5b = positional_bias_summary(data)

    report = r5 + "\n\n" + r5b
    output_path = os.path.join(ANALYSIS_DIR, "position_analysis_report.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
