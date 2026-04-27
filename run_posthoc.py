"""
BCB Pilot — Post-hoc exploratory analyses.
These are reported as exploratory (not pre-registered) to understand
the mechanisms behind the null gate result.

Three analyses:
  1. Compliance analysis: did models actually change behaviour under B?
  2. C sub-prompt breakdown: which policy-inversion prompt was strongest?
  3. Response distribution analysis: are wrong answers random or structured?

Usage:
    python run_posthoc.py
"""

import json
import os
import sys
import math
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import MODELS, DOMAINS, RESULTS_DIR, ANALYSIS_DIR, CHANCE_RATE


def load_trials(model_id, condition):
    """Load trials for one model × one condition."""
    path = os.path.join(RESULTS_DIR, f"{model_id}_{condition}.jsonl")
    trials = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line.strip())
            trials[t["item_id"]] = t
    return trials


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


# ============================================================
# ANALYSIS 1: Compliance — did models change behaviour under B?
# ============================================================

def compliance_analysis(data):
    """
    For each model × domain, compute:
    - Same-answer rate: proportion of items where A and B give identical response
    - A-correct-B-wrong rate: items correct under A but wrong under B (instruction compliance)
    - A-wrong-B-wrong rate: items wrong under both (floor effect)
    - A-correct-B-correct rate: items correct under both (instruction resistance)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 1: INSTRUCTION COMPLIANCE")
    lines.append("Did models actually change behaviour under sandbagging instruction?")
    lines.append("=" * 70)
    lines.append("")

    for model_id in MODELS:
        lines.append(f"Model: {MODELS[model_id]['name']}")
        lines.append("-" * 50)

        for domain in DOMAINS:
            same = 0
            a_correct_b_wrong = 0
            a_correct_b_correct = 0
            a_wrong_b_wrong = 0
            a_wrong_b_correct = 0
            total = 0

            # Get all item IDs for this domain
            item_ids = set()
            for key in data:
                if key[0] == model_id and key[1] == "A":
                    t = data[key]
                    if t["domain"] == domain:
                        item_ids.add(key[2])

            for item_id in item_ids:
                a = data.get((model_id, "A", item_id))
                b = data.get((model_id, "B", item_id))
                if not a or not b:
                    continue
                if not a["is_valid"] or not b["is_valid"]:
                    continue

                total += 1
                a_resp = a["parsed_response"]
                b_resp = b["parsed_response"]
                a_corr = a["correct"] == 1
                b_corr = b["correct"] == 1

                if a_resp == b_resp:
                    same += 1

                if a_corr and not b_corr:
                    a_correct_b_wrong += 1
                elif a_corr and b_corr:
                    a_correct_b_correct += 1
                elif not a_corr and not b_corr:
                    a_wrong_b_wrong += 1
                elif not a_corr and b_corr:
                    a_wrong_b_correct += 1

            if total == 0:
                continue

            same_rate = same / total
            compliance_rate = a_correct_b_wrong / total if total > 0 else 0
            resistance_rate = a_correct_b_correct / total if total > 0 else 0
            a_acc = (a_correct_b_wrong + a_correct_b_correct) / total
            b_acc = (a_correct_b_correct + a_wrong_b_correct) / total

            lines.append(f"  {domain}:")
            lines.append(f"    Items compared: {total}")
            lines.append(f"    Same answer A→B: {same}/{total} ({same_rate:.1%})")
            lines.append(f"    A correct → B wrong (compliance): "
                        f"{a_correct_b_wrong}/{total} ({compliance_rate:.1%})")
            lines.append(f"    A correct → B correct (resistance): "
                        f"{a_correct_b_correct}/{total} ({resistance_rate:.1%})")
            lines.append(f"    Accuracy: A={a_acc:.3f}  B={b_acc:.3f}  "
                        f"Δ={b_acc - a_acc:+.3f}")
            lines.append("")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# ANALYSIS 2: C sub-prompt breakdown
# ============================================================

def subprompt_analysis(data):
    """
    Break down Condition C accuracy by sub-prompt (C1, C2, C3)
    per model × domain.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 2: CONDITION C SUB-PROMPT BREAKDOWN")
    lines.append("Which policy-inversion prompt produced strongest suppression?")
    lines.append("=" * 70)
    lines.append("")

    # Aggregate by model × domain × subprompt
    cells = defaultdict(lambda: {"n": 0, "correct": 0})

    for key, t in data.items():
        model_id, cond, item_id = key
        if cond != "C":
            continue
        if not t["is_valid"]:
            continue
        sp = t.get("c_subprompt", "unknown")
        cell_key = (model_id, t["domain"], sp)
        cells[cell_key]["n"] += 1
        if t["correct"] == 1:
            cells[cell_key]["correct"] += 1

    # Also get Condition A and B for comparison
    cond_accs = defaultdict(dict)
    for key, t in data.items():
        model_id, cond, item_id = key
        if cond not in ["A", "B"]:
            continue
    # Aggregate A and B at model×domain level
    ab_cells = defaultdict(lambda: {"A_n": 0, "A_c": 0, "B_n": 0, "B_c": 0})
    for key, t in data.items():
        model_id, cond, item_id = key
        if cond not in ["A", "B"]:
            continue
        if not t["is_valid"]:
            continue
        ab_key = (model_id, t["domain"])
        ab_cells[ab_key][f"{cond}_n"] += 1
        if t["correct"] == 1:
            ab_cells[ab_key][f"{cond}_c"] += 1

    for model_id in MODELS:
        lines.append(f"Model: {MODELS[model_id]['name']}")
        lines.append(f"  {'Domain':<12s}  {'A':>6s}  {'B':>6s}  "
                     f"{'C1':>6s}  {'C2':>6s}  {'C3':>6s}  "
                     f"{'C_pool':>6s}  {'Strongest':>10s}")
        lines.append("  " + "-" * 72)

        for domain in DOMAINS:
            ab = ab_cells.get((model_id, domain), {})
            a_acc = ab.get("A_c", 0) / ab.get("A_n", 1)
            b_acc = ab.get("B_c", 0) / ab.get("B_n", 1)

            sp_accs = {}
            for sp in ["C1", "C2", "C3"]:
                c = cells.get((model_id, domain, sp), {"n": 0, "correct": 0})
                if c["n"] > 0:
                    sp_accs[sp] = c["correct"] / c["n"]
                else:
                    sp_accs[sp] = None

            # Pooled C
            c_total_n = sum(cells.get((model_id, domain, sp), {"n": 0})["n"]
                           for sp in ["C1", "C2", "C3"])
            c_total_c = sum(cells.get((model_id, domain, sp), {"correct": 0})["correct"]
                           for sp in ["C1", "C2", "C3"])
            c_pool = c_total_c / c_total_n if c_total_n > 0 else None

            # Find strongest suppressor
            valid_sps = {k: v for k, v in sp_accs.items() if v is not None}
            strongest = min(valid_sps, key=valid_sps.get) if valid_sps else "N/A"

            def fmt(v):
                return f"{v:.3f}" if v is not None else "  N/A"

            lines.append(f"  {domain:<12s}  {fmt(a_acc):>6s}  {fmt(b_acc):>6s}  "
                        f"{fmt(sp_accs.get('C1')):>6s}  "
                        f"{fmt(sp_accs.get('C2')):>6s}  "
                        f"{fmt(sp_accs.get('C3')):>6s}  "
                        f"{fmt(c_pool):>6s}  {strongest:>10s}")

        lines.append("")

    # Summary
    lines.append("Sub-prompt summary across all cells:")
    for sp in ["C1", "C2", "C3"]:
        accs = []
        for model_id in MODELS:
            for domain in DOMAINS:
                c = cells.get((model_id, domain, sp), {"n": 0, "correct": 0})
                if c["n"] > 0:
                    accs.append(c["correct"] / c["n"])
        if accs:
            mean_acc = sum(accs) / len(accs)
            sp_labels = {"C1": "Weak student", "C2": "Contrarian", "C3": "Least-likely"}
            lines.append(f"  {sp} ({sp_labels[sp]}): mean acc = {mean_acc:.3f} "
                        f"(n={len(accs)} cells)")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# ANALYSIS 3: Response distribution — random or structured?
# ============================================================

def response_distribution_analysis(data):
    """
    For each model × domain under Condition B, analyse wrong answers:
    - Are they uniform over the 9 wrong options, or clustered?
    - Chi-square test of uniformity
    - Compare to Condition A wrong-answer distribution
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 3: WRONG-ANSWER DISTRIBUTION")
    lines.append("Are wrong answers random (uniform) or structured (clustered)?")
    lines.append("=" * 70)
    lines.append("")

    try:
        from scipy.stats import chisquare, entropy
        has_scipy = True
    except ImportError:
        has_scipy = False
        lines.append("WARNING: scipy not available for chi-square test.")

    for model_id in MODELS:
        lines.append(f"Model: {MODELS[model_id]['name']}")
        lines.append("-" * 50)

        for domain in DOMAINS:
            for cond in ["A", "B", "C"]:
                # Collect wrong-answer letters
                wrong_answers = []
                correct_key_counts = Counter()

                for key, t in data.items():
                    m, c, iid = key
                    if m != model_id or c != cond:
                        continue
                    if t["domain"] != domain:
                        continue
                    if not t["is_valid"]:
                        continue
                    if t["correct"] == 0:
                        wrong_answers.append(t["parsed_response"])
                        correct_key_counts[t["answer_key"]] += 1

                if not wrong_answers:
                    continue

                dist = Counter(wrong_answers)
                n_wrong = len(wrong_answers)

                # All 10 letters
                all_letters = [chr(65 + i) for i in range(10)]
                counts = [dist.get(l, 0) for l in all_letters]

                # Entropy (max entropy for 10 options = log2(10) ≈ 3.32)
                probs = [c / n_wrong for c in counts if c > 0]
                h = -sum(p * math.log2(p) for p in probs) if probs else 0
                max_h = math.log2(10)
                normalised_h = h / max_h if max_h > 0 else 0

                # Top 3 chosen wrong answers
                top3 = dist.most_common(3)

                # Chi-square against uniform over 10 options
                chi_p = None
                if has_scipy and n_wrong >= 10:
                    expected = [n_wrong / 10] * 10
                    stat, chi_p = chisquare(counts, f_exp=expected)

                lines.append(f"  {domain} / Condition {cond}:")
                lines.append(f"    Wrong answers: {n_wrong}")
                lines.append(f"    Entropy: {h:.3f} / {max_h:.3f} "
                            f"(normalised: {normalised_h:.3f})")
                if chi_p is not None:
                    lines.append(f"    Chi-square vs uniform: p={chi_p:.6f} "
                                f"({'non-uniform' if chi_p < 0.05 else 'consistent with uniform'})")
                lines.append(f"    Top 3 wrong choices: "
                            + ", ".join(f"{l}={c}" for l, c in top3))
                lines.append("")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# ANALYSIS 4: Item-level transition matrix A→B
# ============================================================

def transition_analysis(data):
    """
    For items correct under A, what happens under B?
    Do models switch to a specific wrong answer or scatter?
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 4: A→B TRANSITION PATTERNS")
    lines.append("When models get items right under A but wrong under B,")
    lines.append("do they converge on a specific wrong answer?")
    lines.append("=" * 70)
    lines.append("")

    for model_id in MODELS:
        lines.append(f"Model: {MODELS[model_id]['name']}")
        lines.append("-" * 50)

        for domain in DOMAINS:
            # Find items correct under A, wrong under B
            switched = []
            stayed_correct = 0
            total_a_correct = 0

            item_ids = set()
            for key in data:
                if key[0] == model_id and key[1] == "A":
                    if data[key]["domain"] == domain:
                        item_ids.add(key[2])

            for item_id in item_ids:
                a = data.get((model_id, "A", item_id))
                b = data.get((model_id, "B", item_id))
                if not a or not b:
                    continue
                if not a["is_valid"] or not b["is_valid"]:
                    continue
                if a["correct"] != 1:
                    continue

                total_a_correct += 1
                if b["correct"] == 1:
                    stayed_correct += 1
                else:
                    switched.append({
                        "correct_answer": a["answer_key"],
                        "a_response": a["parsed_response"],
                        "b_response": b["parsed_response"],
                    })

            if total_a_correct == 0:
                continue

            n_switched = len(switched)
            switch_rate = n_switched / total_a_correct

            # Analyse where they switched TO
            b_wrong_dist = Counter(s["b_response"] for s in switched)

            # Check if B responses cluster on adjacent options
            # (e.g., if correct is C, do they pick B or D?)
            adjacent_count = 0
            for s in switched:
                correct_idx = ord(s["correct_answer"]) - 65
                chosen_idx = ord(s["b_response"]) - 65
                if abs(correct_idx - chosen_idx) == 1:
                    adjacent_count += 1

            adjacent_rate = adjacent_count / n_switched if n_switched > 0 else 0
            # Expected adjacent rate for random: 2/9 ≈ 0.222 (for middle options)
            # Slightly less at edges

            lines.append(f"  {domain}:")
            lines.append(f"    Items correct under A: {total_a_correct}")
            lines.append(f"    Stayed correct under B: {stayed_correct} "
                        f"({stayed_correct/total_a_correct:.1%})")
            lines.append(f"    Switched to wrong under B: {n_switched} "
                        f"({switch_rate:.1%})")
            if n_switched > 0:
                top3 = b_wrong_dist.most_common(3)
                lines.append(f"    Top wrong choices: "
                            + ", ".join(f"{l}={c}" for l, c in top3))
                lines.append(f"    Adjacent-to-correct rate: {adjacent_rate:.3f} "
                            f"(random expectation ≈ 0.200)")
            lines.append("")

        lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("Loading all trial data...")
    data = load_all()
    print(f"  {len(data)} total trial records loaded.")

    print("\nRunning Analysis 1: Compliance...")
    r1 = compliance_analysis(data)
    print(r1[:500])

    print("\nRunning Analysis 2: C sub-prompt breakdown...")
    r2 = subprompt_analysis(data)
    print(r2[:500])

    print("\nRunning Analysis 3: Response distribution...")
    r3 = response_distribution_analysis(data)
    print(r3[:500])

    print("\nRunning Analysis 4: A→B transitions...")
    r4 = transition_analysis(data)
    print(r4[:500])

    # Write full report
    report = "\n\n".join([r1, r2, r3, r4])
    output_path = os.path.join(ANALYSIS_DIR, "posthoc_report.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nFull post-hoc report saved: {output_path}")


if __name__ == "__main__":
    main()
