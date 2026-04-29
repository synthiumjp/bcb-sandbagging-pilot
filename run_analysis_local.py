"""
run_analysis_local.py

Reads the 12 JSONL result files and runs the full pre-registered analysis.
Run from D:\bcb_pilot with the venv activated.

Usage:
    python run_analysis_local.py

Expects files at: results/positional_bias/{model}_{condition}.jsonl
Expects permutation file at: data/permutation_assignments.csv
"""

import json
import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join("results", "positional_bias")
PERM_FILE = os.path.join("data", "permutation_assignments.csv")
OUTPUT_DIR = os.path.join("results", "positional_bias", "analysis")

OPTION_LETTERS = list("ABCDEFGHIJ")
MODELS = ["llama3-8b", "qwen2.5-7b", "phi3.5-mini"]
CONDITIONS = ["A_original", "A_perm", "B_original", "B_perm"]

# Pre-registered thresholds
ENTROPY_THRESHOLD = 0.90
MANIP_ALPHA = 0.01
MANIP_MIN_DROP = 5.0
SL_THRESHOLD = 0.50
GAP_THRESHOLD = 30.0
SC_MIN_RATE = 0.25
SL_MAX_FOR_CONTENT = 0.20
OTHER_UNCLEAR = 0.50
MISSING_FLAG = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def response_vector(responses):
    counts = Counter(responses.dropna())
    vec = np.array([counts.get(l, 0) for l in OPTION_LETTERS], dtype=float)
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def norm_entropy(vec):
    p = vec[vec > 0]
    return -np.sum(p * np.log(p)) / np.log(10)


def top_k(vec, k=2):
    return [OPTION_LETTERS[i] for i in np.argsort(vec)[::-1][:k]]


def mcnemar(acc_a, acc_b):
    bc = np.sum((acc_a == 0) & (acc_b == 1))
    cb = np.sum((acc_a == 1) & (acc_b == 0))
    n = bc + cb
    if n == 0:
        return 0.0, 1.0
    chi2 = (bc - cb) ** 2 / n
    p = stats.chi2.sf(chi2, df=1)
    return chi2, p


def classify_items(b_orig_resp, b_perm_resp, shifts):
    results = []
    for i in range(len(b_orig_resp)):
        ro = b_orig_resp.iloc[i]
        rp = b_perm_resp.iloc[i]
        k = int(shifts.iloc[i])

        if pd.isna(ro) or pd.isna(rp):
            results.append("other")
            continue

        orig_idx = OPTION_LETTERS.index(ro)
        content_pos = OPTION_LETTERS[(orig_idx + k) % 10]

        if rp == ro:
            results.append("same_letter")
        elif rp == content_pos:
            results.append("shifted_content")
        else:
            results.append("other")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load permutation assignments
    perm_df = pd.read_csv(PERM_FILE)
    print(f"Loaded {len(perm_df)} permutation assignments")

    # Load all data
    data = {}
    for model in MODELS:
        data[model] = {}
        for cond in CONDITIONS:
            path = os.path.join(RESULTS_DIR, f"{model}_{cond}.jsonl")
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping")
                continue
            df = load_jsonl(path)
            data[model][cond] = df
            print(f"  Loaded {model}/{cond}: {len(df)} trials")

    all_results = {}

    for model in MODELS:
        if not all(c in data[model] for c in CONDITIONS):
            print(f"\n{'='*70}")
            print(f"SKIPPING {model} (missing conditions)")
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print(f"{'='*70}")

        a_orig = data[model]["A_original"].sort_values("item_id").reset_index(drop=True)
        a_perm = data[model]["A_perm"].sort_values("item_id").reset_index(drop=True)
        b_orig = data[model]["B_original"].sort_values("item_id").reset_index(drop=True)
        b_perm = data[model]["B_perm"].sort_values("item_id").reset_index(drop=True)

        model_results = {"model": model}

        # =============================================================
        # GATE 1: H2 Entropy replication
        # =============================================================
        vec_a = response_vector(a_orig["parsed_response"])
        vec_b = response_vector(b_orig["parsed_response"])
        ent_a = norm_entropy(vec_a)
        ent_b = norm_entropy(vec_b)
        gate1 = ent_b < ENTROPY_THRESHOLD

        print(f"\n--- GATE 1: H2 Entropy Replication ---")
        print(f"  Entropy A-original: {ent_a:.4f}")
        print(f"  Entropy B-original: {ent_b:.4f}")
        print(f"  Threshold: < {ENTROPY_THRESHOLD}")
        print(f"  Result: {'PASSED' if gate1 else 'FAILED'}")

        print(f"  B-original distribution:")
        for i, l in enumerate(OPTION_LETTERS):
            print(f"    {l}: {vec_b[i]*100:5.1f}%")

        model_results["gate1"] = {"entropy_a": ent_a, "entropy_b": ent_b, "passed": gate1}

        # =============================================================
        # GATE 2: Manipulation check
        # =============================================================
        acc_a_vals = a_orig["correct"].fillna(0).astype(int).values
        acc_b_vals = b_orig["correct"].fillna(0).astype(int).values
        mean_a = acc_a_vals.mean()
        mean_b = acc_b_vals.mean()
        drop = (mean_a - mean_b) * 100
        chi2_m, p_m = mcnemar(acc_a_vals, acc_b_vals)
        gate2 = p_m < MANIP_ALPHA and drop >= MANIP_MIN_DROP

        print(f"\n--- GATE 2: Manipulation Check ---")
        print(f"  Accuracy A-original: {mean_a:.4f} ({mean_a*100:.1f}%)")
        print(f"  Accuracy B-original: {mean_b:.4f} ({mean_b*100:.1f}%)")
        print(f"  Drop: {drop:.1f} pp")
        print(f"  McNemar chi2={chi2_m:.1f}, p={p_m:.6f}")
        print(f"  Result: {'PASSED' if gate2 else 'FAILED'}")

        model_results["gate2"] = {
            "acc_a": mean_a, "acc_b": mean_b,
            "drop_pp": drop, "mcnemar_p": p_m, "passed": gate2
        }

        # =============================================================
        # Missing rates
        # =============================================================
        print(f"\n--- Missing Rates ---")
        for cond in CONDITIONS:
            df = data[model][cond]
            miss = df["parsed_response"].isna().sum()
            rate = miss / len(df)
            flag = " *** FLAG" if rate > MISSING_FLAG else ""
            print(f"  {cond}: {miss}/{len(df)} ({rate:.1%}){flag}")

        # =============================================================
        # H1: Primary diagnostic (if both gates pass)
        # =============================================================
        if gate1 and gate2:
            print(f"\n--- GATE 3: H1 Primary Diagnostic ---")

            perm_lookup = perm_df.set_index("item_id")["shift_k"]
            shifts = b_orig["item_id"].map(perm_lookup).reset_index(drop=True)

            classifications = classify_items(
                b_orig["parsed_response"].reset_index(drop=True),
                b_perm["parsed_response"].reset_index(drop=True),
                shifts
            )

            cc = Counter(classifications)
            n = len(classifications)
            sl = cc["same_letter"] / n
            sc = cc["shifted_content"] / n
            ot = cc["other"] / n
            gap = (sl - sc) * 100

            print(f"\n  Per-item classification (n={n}):")
            print(f"    Same-letter:      {cc['same_letter']:4d} ({sl:.1%})")
            print(f"    Shifted-content:  {cc['shifted_content']:4d} ({sc:.1%})")
            print(f"    Other:            {cc['other']:4d} ({ot:.1%})")
            print(f"    Gap (SL - SC):    {gap:.1f} pp")

            vec_bp = response_vector(b_perm["parsed_response"])
            t2_orig = top_k(vec_b)
            t2_perm = top_k(vec_bp)
            overlap = set(t2_orig) & set(t2_perm)

            print(f"\n  Top-2 modal B-original: {t2_orig}")
            print(f"  Top-2 modal B-perm:     {t2_perm}")
            print(f"  Overlap: {overlap if overlap else 'NONE'}")

            print(f"\n  B-perm distribution:")
            for i, l in enumerate(OPTION_LETTERS):
                print(f"    {l}: {vec_bp[i]*100:5.1f}%")

            # Pearson r
            r = np.corrcoef(vec_b, vec_bp)[0, 1]
            print(f"\n  S1: Pearson r = {r:.4f}")

            # Decision
            pt1 = sl > SL_THRESHOLD
            pt2 = gap >= GAP_THRESHOLD
            pt3 = len(overlap) > 0
            ct1 = sc > sl
            ct2 = sc > SC_MIN_RATE
            ct3 = sl < SL_MAX_FOR_CONTENT

            print(f"\n  --- DECISION RULE ---")
            print(f"  Position-tracking:")
            print(f"    SL > 50%:      {sl:.1%} -> {'PASS' if pt1 else 'FAIL'}")
            print(f"    Gap >= 30 pp:  {gap:.1f} -> {'PASS' if pt2 else 'FAIL'}")
            print(f"    Top-2 overlap: {overlap} -> {'PASS' if pt3 else 'FAIL'}")

            if pt1 and pt2 and pt3:
                outcome = "POSITION-TRACKING CONFIRMED"
            else:
                print(f"  Content-tracking:")
                print(f"    SC > SL:   {sc:.1%} > {sl:.1%} -> {'PASS' if ct1 else 'FAIL'}")
                print(f"    SC > 25%:  {sc:.1%} -> {'PASS' if ct2 else 'FAIL'}")
                print(f"    SL < 20%:  {sl:.1%} -> {'PASS' if ct3 else 'FAIL'}")

                if ct1 and ct2 and ct3:
                    outcome = "CONTENT-TRACKING SUPPORTED"
                elif ot > OTHER_UNCLEAR:
                    outcome = "MECHANISM UNCLEAR"
                else:
                    outcome = "AMBIGUOUS"

            print(f"\n  >>> {outcome} <<<")

            model_results["h1"] = {
                "same_letter": sl, "shifted_content": sc, "other": ot,
                "gap_pp": gap, "top2_orig": t2_orig, "top2_perm": t2_perm,
                "pearson_r": r, "outcome": outcome
            }

            # S4: accuracy by correct-answer position
            print(f"\n  --- S4: Accuracy by correct-answer position (B-perm) ---")
            for l in OPTION_LETTERS:
                mask = b_perm["answer_key"] == l
                n_pos = mask.sum()
                if n_pos > 0:
                    acc_pos = b_perm.loc[mask, "correct"].fillna(0).mean()
                    print(f"    Correct at {l}: n={n_pos:3d}, acc={acc_pos:.3f} ({acc_pos*100:.1f}%)")

            # Save item-level classifications
            class_df = pd.DataFrame({
                "item_id": b_orig["item_id"].values,
                "domain": b_orig["domain"].values,
                "b_orig_response": b_orig["parsed_response"].values,
                "b_perm_response": b_perm["parsed_response"].values,
                "shift_k": shifts.values,
                "classification": classifications,
            })
            class_path = os.path.join(OUTPUT_DIR, f"{model}_item_classifications.csv")
            class_df.to_csv(class_path, index=False)
            print(f"\n  Item classifications saved: {class_path}")

        else:
            reasons = []
            if not gate1:
                reasons.append("Gate 1 failed")
            if not gate2:
                reasons.append("Gate 2 failed")
            print(f"\n  H1 SKIPPED: {', '.join(reasons)}")
            model_results["h1"] = {"outcome": "skipped", "reasons": reasons}

        # =============================================================
        # H3: Option-order sensitivity (all models)
        # =============================================================
        acc_ao = a_orig["correct"].fillna(0).astype(int).values
        acc_ap = a_perm["correct"].fillna(0).astype(int).values
        diff_h3 = (acc_ao.mean() - acc_ap.mean()) * 100
        _, p_h3 = mcnemar(acc_ao, acc_ap)

        print(f"\n--- H3: Option-order sensitivity ---")
        print(f"  Acc A-original: {acc_ao.mean():.4f} ({acc_ao.mean()*100:.1f}%)")
        print(f"  Acc A-perm:     {acc_ap.mean():.4f} ({acc_ap.mean()*100:.1f}%)")
        print(f"  Difference:     {diff_h3:.1f} pp")
        print(f"  McNemar p={p_h3:.6f} (Bonferroni threshold: 0.0167)")

        model_results["h3"] = {
            "acc_orig": float(acc_ao.mean()), "acc_perm": float(acc_ap.mean()),
            "diff_pp": diff_h3, "p": p_h3
        }

        # =============================================================
        # H4: Model-specific predictions
        # =============================================================
        if model == "phi3.5-mini":
            vec_bp_phi = response_vector(b_perm["parsed_response"])
            modal_phi = OPTION_LETTERS[np.argmax(vec_bp_phi)]
            print(f"\n--- H4a: Phi J-bias persistence ---")
            print(f"  Modal B-perm position: {modal_phi}")
            print(f"  Prediction: J")
            print(f"  Result: {'CONFIRMED' if modal_phi == 'J' else 'NOT CONFIRMED (modal=' + modal_phi + ')'}")
            model_results["h4a"] = {"modal": modal_phi, "confirmed": modal_phi == "J"}

        if model == "qwen2.5-7b":
            vec_ap_q = response_vector(a_perm["parsed_response"])
            vec_bp_q = response_vector(b_perm["parsed_response"])
            obs_q = np.array([vec_ap_q * len(a_perm), vec_bp_q * len(b_perm)])
            try:
                chi2_q, p_q, _, _ = stats.chi2_contingency(obs_q)
            except ValueError:
                chi2_q, p_q = 0, 1
            print(f"\n--- H4b: Qwen non-compliance ---")
            print(f"  Chi-square (A-perm vs B-perm): {chi2_q:.1f}, p={p_q:.6f}")
            print(f"  Prediction: non-significant (p > 0.01)")
            print(f"  Result: {'CONFIRMED' if p_q > 0.01 else 'NOT CONFIRMED'}")
            model_results["h4b"] = {"chi2": chi2_q, "p": p_q, "confirmed": p_q > 0.01}

        all_results[model] = model_results

    # Save all results
    results_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"All results saved: {results_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
