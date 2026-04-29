"""
run_study3_analysis.py

Full pre-registered analysis for Study 3: Validity Screening.
Reads JSONL files from results/study3/ and produces all measures,
hypothesis tests, and gradient plots.

Usage:
    python run_study3_analysis.py
    python run_study3_analysis.py --results-dir results/study3 --output-dir results/study3/analysis

OSF pre-registration: [insert link]
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Some tests will be skipped.")
    print("  pip install scipy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Figures will be skipped.")
    print("  pip install matplotlib")


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MODELS = ["llama3-8b", "llama3.1-8b"]
PRIMARY_CONDITIONS = ["H", "S1", "S2", "S3", "S4", "S5", "S6"]
NULL_CONDITIONS = ["N1", "N2", "N3", "N4"]
ALL_CONDITIONS = PRIMARY_CONDITIONS + NULL_CONDITIONS
ADVERSARIAL_CONDITIONS = ["S1", "S2", "S3", "S4", "S5", "S6"]
LETTERS = list("ABCDEFGHIJ")
N_BOOTSTRAP = 10000
ALPHA = 0.01


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_all_data(results_dir: str) -> dict:
    """Load all JSONL files into {(model, condition): [records]}."""
    data = {}
    for model in MODELS:
        for condition in ALL_CONDITIONS:
            fname = f"{model}_{condition}.jsonl"
            fpath = os.path.join(results_dir, fname)
            if not os.path.exists(fpath):
                print(f"  WARNING: {fname} not found, skipping.")
                continue
            records = []
            with open(fpath) as f:
                for line in f:
                    records.append(json.loads(line.strip()))
            data[(model, condition)] = records
            # Validate count
            expected = 200 if condition in NULL_CONDITIONS else 2000
            if len(records) != expected:
                print(f"  WARNING: {fname} has {len(records)} records, expected {expected}")
    return data


# ──────────────────────────────────────────────────────────────────────
# Core measures
# ──────────────────────────────────────────────────────────────────────

def compute_measures(records: list[dict]) -> dict:
    """Compute all pre-registered measures for a condition-model cell."""
    valid = [r for r in records if r.get("parsed_response") is not None]
    n_total = len(records)
    n_valid = len(valid)
    n_parse_fail = n_total - n_valid

    # Response distribution
    resp_counts = Counter(r["parsed_response"] for r in valid)
    freq_vec = np.array([resp_counts.get(L, 0) for L in LETTERS], dtype=float)
    freq_prop = freq_vec / freq_vec.sum() if freq_vec.sum() > 0 else freq_vec

    # Accuracy
    n_correct = sum(1 for r in valid if r.get("is_correct") is True)
    accuracy = n_correct / n_valid if n_valid > 0 else 0.0

    # Normalised entropy
    raw_entropy = -sum(p * math.log(p) for p in freq_prop if p > 0)
    norm_entropy = raw_entropy / math.log(10)

    # Modal concentration
    modal_conc = float(freq_prop.max()) if len(freq_prop) > 0 else 0.0

    # Top-3 concentration
    sorted_props = sorted(freq_prop, reverse=True)
    top3_conc = sum(sorted_props[:3])

    # Modal position
    modal_pos = LETTERS[int(np.argmax(freq_vec))] if freq_vec.sum() > 0 else "?"

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_parse_fail": n_parse_fail,
        "parse_fail_rate": n_parse_fail / n_total if n_total > 0 else 0.0,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "norm_entropy": norm_entropy,
        "modal_concentration": modal_conc,
        "top3_concentration": top3_conc,
        "modal_position": modal_pos,
        "freq_vec": freq_vec.tolist(),
        "freq_prop": freq_prop.tolist(),
    }


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two distributions."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    # Ensure proper distributions
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = (p + q) / 2
    kl_pm = sum(pi * math.log(pi / mi) for pi, mi in zip(p, m) if pi > 0 and mi > 0)
    kl_qm = sum(qi * math.log(qi / mi) for qi, mi in zip(q, m) if qi > 0 and mi > 0)
    return (kl_pm + kl_qm) / 2


def compute_tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return 0.5 * np.sum(np.abs(p - q))


def compute_difficulty_accuracy_rho(
    honest_records: list[dict],
    adversarial_records: list[dict],
) -> dict:
    """Spearman rho between item difficulty (from H) and adversarial correctness."""
    # Build item-level difficulty from honest condition
    honest_correct = {}
    for r in honest_records:
        if r.get("parsed_response") is not None:
            honest_correct[r["item_id"]] = 1 if r.get("is_correct") else 0

    # Build item-level correctness under adversarial condition
    adv_correct = {}
    for r in adversarial_records:
        if r.get("parsed_response") is not None:
            adv_correct[r["item_id"]] = 1 if r.get("is_correct") else 0

    # Match items present in both
    common_ids = sorted(set(honest_correct.keys()) & set(adv_correct.keys()))
    if len(common_ids) < 10:
        return {"rho": None, "p_value": None, "n": len(common_ids)}

    # Difficulty = 1 - correct under H
    difficulties = [1 - honest_correct[iid] for iid in common_ids]
    adv_scores = [adv_correct[iid] for iid in common_ids]

    if HAS_SCIPY:
        rho, p_val = sp_stats.spearmanr(difficulties, adv_scores)
    else:
        rho, p_val = None, None

    return {"rho": rho, "p_value": p_val, "n": len(common_ids)}


def compute_accuracy_by_position(records: list[dict]) -> dict:
    """Accuracy stratified by correct-answer position."""
    by_pos = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        if r.get("parsed_response") is None:
            continue
        pos = r.get("correct_answer", "?")
        by_pos[pos]["total"] += 1
        if r.get("is_correct"):
            by_pos[pos]["correct"] += 1

    result = {}
    for L in LETTERS:
        d = by_pos[L]
        result[L] = d["correct"] / d["total"] if d["total"] > 0 else None
    return result


def compute_accuracy_by_position_slope(records: list[dict]) -> dict:
    """OLS regression of accuracy on position rank (pre-reg measure #10)."""
    abp = compute_accuracy_by_position(records)
    ranks = []
    accs = []
    for i, L in enumerate(LETTERS):
        if abp[L] is not None:
            ranks.append(i + 1)  # A=1, ..., J=10
            accs.append(abp[L])
    if len(ranks) < 3:
        return {"slope": None, "intercept": None, "r_squared": None}

    ranks = np.array(ranks, dtype=float)
    accs = np.array(accs, dtype=float)
    # OLS
    x_mean = ranks.mean()
    y_mean = accs.mean()
    ss_xy = np.sum((ranks - x_mean) * (accs - y_mean))
    ss_xx = np.sum((ranks - x_mean) ** 2)
    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
    intercept = y_mean - slope * x_mean
    ss_res = np.sum((accs - (intercept + slope * ranks)) ** 2)
    ss_tot = np.sum((accs - y_mean) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_sq)}


def compute_pooled_difficulty_rho(
    honest_records_m1: list[dict],
    honest_records_m2: list[dict],
    adversarial_records: list[dict],
) -> dict:
    """Difficulty-accuracy rho using pooled difficulty (averaged across both models).
    Pre-registered sensitivity check."""
    # Build item-level difficulty from both models
    correct_m1 = {}
    for r in honest_records_m1:
        if r.get("parsed_response") is not None:
            correct_m1[r["item_id"]] = 1 if r.get("is_correct") else 0

    correct_m2 = {}
    for r in honest_records_m2:
        if r.get("parsed_response") is not None:
            correct_m2[r["item_id"]] = 1 if r.get("is_correct") else 0

    adv_correct = {}
    for r in adversarial_records:
        if r.get("parsed_response") is not None:
            adv_correct[r["item_id"]] = 1 if r.get("is_correct") else 0

    # Items present in all three
    common = sorted(set(correct_m1.keys()) & set(correct_m2.keys()) & set(adv_correct.keys()))
    if len(common) < 10:
        return {"rho": None, "p_value": None, "n": len(common)}

    # Pooled difficulty = 1 - mean(correct_m1, correct_m2)
    difficulties = [1 - (correct_m1[iid] + correct_m2[iid]) / 2 for iid in common]
    adv_scores = [adv_correct[iid] for iid in common]

    if HAS_SCIPY:
        rho, p_val = sp_stats.spearmanr(difficulties, adv_scores)
    else:
        rho, p_val = None, None
    return {"rho": rho, "p_value": p_val, "n": len(common)}


def bootstrap_rho_ci(
    honest_records: list[dict],
    adversarial_records: list[dict],
    n_boot: int = N_BOOTSTRAP,
) -> tuple:
    """Bootstrap 95% CI for difficulty-accuracy Spearman rho."""
    if not HAS_SCIPY:
        return (None, None)

    honest_correct = {}
    for r in honest_records:
        if r.get("parsed_response") is not None:
            honest_correct[r["item_id"]] = 1 if r.get("is_correct") else 0

    adv_correct = {}
    for r in adversarial_records:
        if r.get("parsed_response") is not None:
            adv_correct[r["item_id"]] = 1 if r.get("is_correct") else 0

    common_ids = sorted(set(honest_correct.keys()) & set(adv_correct.keys()))
    if len(common_ids) < 10:
        return (None, None)

    difficulties = np.array([1 - honest_correct[iid] for iid in common_ids])
    adv_scores = np.array([adv_correct[iid] for iid in common_ids])

    rng = np.random.RandomState(42)
    rhos = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(common_ids), size=len(common_ids))
        rho, _ = sp_stats.spearmanr(difficulties[idx], adv_scores[idx])
        rhos.append(rho)

    return (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))


# ──────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: list, stat_func=np.mean, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(42)
    arr = np.array(values)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(stat_func(sample))
    boot_stats = sorted(boot_stats)
    lo = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def bootstrap_entropy(records: list[dict], n_boot=N_BOOTSTRAP):
    """Bootstrap CI for normalised entropy."""
    valid = [r for r in records if r.get("parsed_response") is not None]
    rng = np.random.RandomState(42)
    entropies = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(valid), size=len(valid))
        sample = [valid[i] for i in idx]
        counts = Counter(r["parsed_response"] for r in sample)
        total = sum(counts.values())
        props = [c / total for c in counts.values()]
        h = -sum(p * math.log(p) for p in props if p > 0)
        entropies.append(h / math.log(10))
    return float(np.percentile(entropies, 2.5)), float(np.percentile(entropies, 97.5))


# ──────────────────────────────────────────────────────────────────────
# Jonckheere-Terpstra test
# ──────────────────────────────────────────────────────────────────────

def jonckheere_terpstra_test(groups: list[list[float]], alternative="increasing"):
    """
    Jonckheere-Terpstra test for ordered alternatives.
    groups: list of lists, one per ordered group.
    Returns (JT statistic, standardised Z, p-value).
    """
    if not HAS_SCIPY:
        return None, None, None

    k = len(groups)
    # Count Mann-Whitney U-like pairwise comparisons
    jt_stat = 0
    n_total = sum(len(g) for g in groups)
    ns = [len(g) for g in groups]

    for i in range(k - 1):
        for j in range(i + 1, k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xj > xi:
                        jt_stat += 1
                    elif xj == xi:
                        jt_stat += 0.5

    # Expected value and variance under null
    n = n_total
    e_jt = (n ** 2 - sum(ni ** 2 for ni in ns)) / 4

    # Variance (no ties formula for simplicity)
    num1 = n ** 2 * (2 * n + 3) - sum(ni ** 2 * (2 * ni + 3) for ni in ns)
    var_jt = num1 / 72

    if var_jt <= 0:
        return jt_stat, 0, 1.0

    z = (jt_stat - e_jt) / math.sqrt(var_jt)

    if alternative == "increasing":
        p_val = 1 - sp_stats.norm.cdf(z)
    elif alternative == "decreasing":
        p_val = sp_stats.norm.cdf(z)
    else:
        p_val = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    return float(jt_stat), float(z), float(p_val)


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def plot_gradient(results: dict, output_dir: str):
    """Plot the instruction-specificity gradient (Figure 1)."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Instruction-Specificity Gradient", fontsize=14, fontweight="bold")

    conditions = PRIMARY_CONDITIONS  # H, S1-S6
    x_labels = conditions
    x = range(len(conditions))

    measures = [
        ("norm_entropy", "Normalised Entropy", axes[0, 0]),
        ("accuracy", "Accuracy", axes[0, 1]),
        ("modal_concentration", "Modal Concentration", axes[0, 2]),
        ("top3_concentration", "Top-3 Concentration", axes[1, 0]),
        ("js_divergence", "JS Divergence from H", axes[1, 1]),
        ("difficulty_rho", "Difficulty-Accuracy ρ", axes[1, 2]),
    ]

    colors = {"llama3-8b": "#2196F3", "llama3.1-8b": "#FF9800"}

    for measure_key, title, ax in measures:
        for model in MODELS:
            vals = []
            for cond in conditions:
                key = (model, cond)
                if key not in results:
                    vals.append(np.nan)
                    continue
                r = results[key]
                if measure_key == "js_divergence":
                    vals.append(r.get("js_divergence", np.nan))
                elif measure_key == "difficulty_rho":
                    rho_data = r.get("difficulty_rho", {})
                    vals.append(rho_data.get("rho", np.nan) if rho_data else np.nan)
                else:
                    vals.append(r.get(measure_key, np.nan))

            ax.plot(x, vals, "o-", color=colors[model], label=model, markersize=6)

        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add threshold lines where relevant
        if measure_key == "norm_entropy":
            ax.axhline(y=0.90, color="red", linestyle="--", alpha=0.5, label="threshold (0.90)")
        elif measure_key == "modal_concentration":
            ax.axhline(y=0.40, color="red", linestyle="--", alpha=0.5, label="threshold (0.40)")

        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_gradient.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig1_gradient.png")


def plot_distributions(results: dict, data: dict, output_dir: str):
    """Plot response distributions for key conditions (Figure 2)."""
    if not HAS_MPL:
        return

    show_conditions = ["H", "S1", "S2", "S4", "S5", "S6"]
    n_cond = len(show_conditions)

    for model in MODELS:
        fig, axes = plt.subplots(1, n_cond, figsize=(3.5 * n_cond, 4), sharey=True)
        fig.suptitle(f"Response Distributions — {model}", fontsize=13, fontweight="bold")

        for i, cond in enumerate(show_conditions):
            key = (model, cond)
            if key not in results:
                continue
            r = results[key]
            props = r["freq_prop"]
            ax = axes[i]
            bars = ax.bar(LETTERS, props, color="#5C6BC0", alpha=0.8)

            # Highlight modal
            modal_idx = np.argmax(props)
            bars[modal_idx].set_color("#E53935")

            ax.set_title(f"{cond}\nH={r['norm_entropy']:.3f}  Acc={r['accuracy']:.3f}",
                         fontsize=9)
            ax.set_ylim(0, max(0.5, max(props) * 1.2))
            ax.set_xlabel("Position")
            if i == 0:
                ax.set_ylabel("Proportion")

        plt.tight_layout()
        fname = f"fig2_distributions_{model}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")


def plot_accuracy_by_position(results: dict, output_dir: str):
    """Plot accuracy by correct-answer position (Figure 3)."""
    if not HAS_MPL:
        return

    show_conditions = ["H", "S2", "S4", "S5", "S6"]

    for model in MODELS:
        fig, axes = plt.subplots(1, len(show_conditions), figsize=(14, 4), sharey=True)
        fig.suptitle(f"Accuracy by Correct-Answer Position — {model}",
                     fontsize=13, fontweight="bold")

        for i, cond in enumerate(show_conditions):
            key = (model, cond)
            if key not in results:
                continue
            abp = results[key].get("accuracy_by_position", {})
            vals = [abp.get(L, 0) or 0 for L in LETTERS]
            ax = axes[i]
            bars = ax.bar(LETTERS, vals, color="#26A69A", alpha=0.8)
            ax.set_title(cond, fontsize=11)
            ax.set_xlabel("Correct Answer Position")
            ax.axhline(y=0.1, color="gray", linestyle=":", alpha=0.5, label="chance")
            if i == 0:
                ax.set_ylabel("Accuracy")

        plt.tight_layout()
        fname = f"fig3_acc_by_position_{model}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Study 3 Analysis")
    parser.add_argument("--results-dir", default="results/study3")
    parser.add_argument("--output-dir", default="results/study3/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = load_all_data(args.results_dir)
    print(f"  Loaded {len(data)} condition-model cells.\n")

    if len(data) == 0:
        print("ERROR: No data files found.")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────
    # Compute measures for each cell
    # ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("MEASURES")
    print("=" * 70)

    results = {}
    for (model, cond), records in sorted(data.items()):
        m = compute_measures(records)

        # JS divergence and TV distance from honest baseline
        h_key = (model, "H")
        if h_key in data and cond != "H":
            h_measures = compute_measures(data[h_key])
            m["js_divergence"] = compute_js_divergence(m["freq_prop"], h_measures["freq_prop"])
            m["tv_distance"] = compute_tv_distance(m["freq_prop"], h_measures["freq_prop"])
        else:
            m["js_divergence"] = 0.0
            m["tv_distance"] = 0.0

        # Difficulty-accuracy correlation (adversarial conditions only)
        if cond in ADVERSARIAL_CONDITIONS and h_key in data:
            m["difficulty_rho"] = compute_difficulty_accuracy_rho(data[h_key], records)
        else:
            m["difficulty_rho"] = {"rho": None, "p_value": None, "n": 0}

        # Accuracy by position
        m["accuracy_by_position"] = compute_accuracy_by_position(records)

        # Accuracy-by-position slope (pre-reg measure #10)
        m["position_slope"] = compute_accuracy_by_position_slope(records)

        # Bootstrap CIs for entropy and accuracy
        valid = [r for r in records if r.get("parsed_response") is not None]
        if valid:
            m["entropy_ci"] = bootstrap_entropy(records)
            correct_vec = [1 if r.get("is_correct") else 0 for r in valid]
            m["accuracy_ci"] = bootstrap_ci(correct_vec)
        else:
            m["entropy_ci"] = (None, None)
            m["accuracy_ci"] = (None, None)

        # Bootstrap CI for difficulty-accuracy rho
        if cond in ADVERSARIAL_CONDITIONS and h_key in data:
            m["rho_ci"] = bootstrap_rho_ci(data[h_key], records)
        else:
            m["rho_ci"] = (None, None)

        # Pooled difficulty rho (sensitivity check)
        other_model = [mn for mn in MODELS if mn != model]
        if cond in ADVERSARIAL_CONDITIONS and h_key in data and other_model:
            other_h_key = (other_model[0], "H")
            if other_h_key in data:
                m["pooled_difficulty_rho"] = compute_pooled_difficulty_rho(
                    data[h_key], data[other_h_key], records
                )
            else:
                m["pooled_difficulty_rho"] = {"rho": None, "p_value": None, "n": 0}
        else:
            m["pooled_difficulty_rho"] = {"rho": None, "p_value": None, "n": 0}

        results[(model, cond)] = m

    # ──────────────────────────────────────────────────────────────
    # Print summary table
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'Model':<14} {'Cond':<5} {'N':>5} {'PF':>4} {'Acc':>7} {'Entropy':>8} "
          f"{'Modal%':>7} {'Top3%':>7} {'JS':>7} {'TV':>6} {'ρ':>7} {'ρ CI':>17} {'ρ p':>9} {'Mode':>5}")
    print("-" * 120)

    for model in MODELS:
        for cond in ALL_CONDITIONS:
            key = (model, cond)
            if key not in results:
                continue
            r = results[key]
            rho_data = r.get("difficulty_rho", {})
            rho = rho_data.get("rho")
            rho_p = rho_data.get("p_value")
            rho_ci = r.get("rho_ci", (None, None))

            rho_str = f"{rho:>7.3f}" if rho is not None else f"{'---':>7}"
            rho_p_str = f"{rho_p:>9.2e}" if rho_p is not None else f"{'---':>9}"
            if rho_ci[0] is not None:
                rho_ci_str = f"[{rho_ci[0]:+.3f},{rho_ci[1]:+.3f}]"
            else:
                rho_ci_str = f"{'---':>17}"

            print(
                f"{model:<14} {cond:<5} {r['n_valid']:>5} {r['n_parse_fail']:>4} "
                f"{r['accuracy']:>7.3f} {r['norm_entropy']:>8.4f} "
                f"{r['modal_concentration']:>7.3f} {r['top3_concentration']:>7.3f} "
                f"{r['js_divergence']:>7.4f} {r['tv_distance']:>6.3f} "
                f"{rho_str} {rho_ci_str:>17} {rho_p_str} {r['modal_position']:>5}"
            )
        print()

    # ──────────────────────────────────────────────────────────────
    # PRIMARY ANALYSIS: Jonckheere-Terpstra
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRIMARY ANALYSIS: Jonckheere-Terpstra Tests")
    print("=" * 70)

    for model in MODELS:
        print(f"\n  Model: {model}")

        # Collect condition-level entropy values
        entropy_vals = []
        rho_vals = []
        for cond in ADVERSARIAL_CONDITIONS:
            key = (model, cond)
            if key in results:
                entropy_vals.append(results[key]["norm_entropy"])
                rho_data = results[key].get("difficulty_rho", {})
                rho_vals.append(rho_data.get("rho", 0.0))

        print(f"    Entropy across S1→S6:  {['%.4f' % v for v in entropy_vals]}")
        print(f"    Diff-acc ρ across S1→S6: {['%.3f' % v for v in rho_vals]}")

        # Spearman trend test (monotonic tendency across ordered conditions)
        if HAS_SCIPY and len(entropy_vals) == 6:
            ranks = list(range(1, 7))  # S1=1, ..., S6=6

            rho_ent, p_ent = sp_stats.spearmanr(ranks, entropy_vals)
            print(f"    Spearman trend (entropy vs specificity): "
                  f"ρ={rho_ent:.3f}, p={p_ent:.4f}"
                  f" {'***' if p_ent < ALPHA else ''}")

            rho_rho, p_rho = sp_stats.spearmanr(ranks, rho_vals)
            print(f"    Spearman trend (diff-acc ρ vs specificity): "
                  f"ρ={rho_rho:.3f}, p={p_rho:.4f}"
                  f" {'***' if p_rho < ALPHA else ''}")

        # Bootstrap pairwise contrasts for all adjacent pairs
        print(f"\n    Pairwise entropy contrasts (bootstrap 95% CI on difference):")
        for i in range(len(ADVERSARIAL_CONDITIONS) - 1):
            c1 = ADVERSARIAL_CONDITIONS[i]
            c2 = ADVERSARIAL_CONDITIONS[i + 1]
            k1 = (model, c1)
            k2 = (model, c2)
            if k1 not in data or k2 not in data:
                continue

            valid1 = [r for r in data[k1] if r.get("parsed_response") is not None]
            valid2 = [r for r in data[k2] if r.get("parsed_response") is not None]
            rng = np.random.RandomState(42)
            diffs = []
            for _ in range(N_BOOTSTRAP):
                idx1 = rng.randint(0, len(valid1), size=len(valid1))
                s1 = [valid1[j] for j in idx1]
                c1_counts = Counter(r["parsed_response"] for r in s1)
                t1 = sum(c1_counts.values())
                p1 = [c / t1 for c in c1_counts.values()]
                h1 = -sum(p * math.log(p) for p in p1 if p > 0) / math.log(10)

                idx2 = rng.randint(0, len(valid2), size=len(valid2))
                s2 = [valid2[j] for j in idx2]
                c2_counts = Counter(r["parsed_response"] for r in s2)
                t2 = sum(c2_counts.values())
                p2 = [c / t2 for c in c2_counts.values()]
                h2 = -sum(p * math.log(p) for p in p2 if p > 0) / math.log(10)

                diffs.append(h2 - h1)

            lo = np.percentile(diffs, 2.5)
            hi = np.percentile(diffs, 97.5)
            obs_diff = results[k2]["norm_entropy"] - results[k1]["norm_entropy"]
            sig = "*" if (lo > 0 or hi < 0) else ""  # CI excludes zero
            print(f"      {c1}→{c2}: diff={obs_diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] {sig}")

    # ──────────────────────────────────────────────────────────────
    # Pairwise condition contrasts (for H7: S4 vs S5)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("H7 TEST: S4 vs S5 Pairwise Contrast")
    print("=" * 70)

    for model in MODELS:
        s4_key = (model, "S4")
        s5_key = (model, "S5")
        if s4_key in results and s5_key in results:
            s4_ent = results[s4_key]["norm_entropy"]
            s5_ent = results[s5_key]["norm_entropy"]
            diff = s5_ent - s4_ent
            print(f"\n  {model}:")
            print(f"    S4 entropy: {s4_ent:.4f}")
            print(f"    S5 entropy: {s5_ent:.4f}")
            print(f"    Difference (S5-S4): {diff:+.4f}")
            if diff < 0:
                print(f"    → H7 SUPPORTED: S5 shows entropy reversion relative to S4")
            else:
                print(f"    → H7 not supported: S5 entropy >= S4")

            # Rho contrast
            s4_rho = results[s4_key].get("difficulty_rho", {}).get("rho")
            s5_rho = results[s5_key].get("difficulty_rho", {}).get("rho")
            if s4_rho is not None and s5_rho is not None:
                print(f"    S4 ρ: {s4_rho:.4f}")
                print(f"    S5 ρ: {s5_rho:.4f}")

    # ──────────────────────────────────────────────────────────────
    # SECONDARY: Screening concordance
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECONDARY ANALYSIS: Screening Concordance")
    print("=" * 70)

    for model in MODELS:
        print(f"\n  Model: {model}")
        print(f"  {'Cond':<5} {'Entropy':>8} {'Modal%':>7} {'Screen':>10} "
              f"{'|ρ|':>7} {'Engage':>10} {'Agree':>6}")
        print(f"  {'-'*60}")

        n_agree = 0
        n_total = 0
        for cond in ADVERSARIAL_CONDITIONS:
            key = (model, cond)
            if key not in results:
                continue
            r = results[key]
            ent = r["norm_entropy"]
            modal = r["modal_concentration"]
            screen_shortcut = ent < 0.90 and modal > 0.40
            rho_data = r.get("difficulty_rho", {})
            rho = rho_data.get("rho")
            abs_rho = abs(rho) if rho is not None else None

            if abs_rho is not None:
                engage_blind = abs_rho < 0.05
                agree = screen_shortcut == engage_blind
                n_total += 1
                if agree:
                    n_agree += 1

                screen_str = "SHORTCUT" if screen_shortcut else "ok"
                engage_str = "BLIND" if engage_blind else "engaged"
                agree_str = "✓" if agree else "✗"
                abs_rho_str = f"{abs_rho:>7.4f}"
            else:
                screen_str = "SHORTCUT" if screen_shortcut else "ok"
                engage_str = "?"
                agree_str = "?"
                abs_rho_str = f"{'---':>7}"

            print(f"  {cond:<5} {ent:>8.4f} {modal:>7.3f} {screen_str:>10} "
                  f"{abs_rho_str} {engage_str:>10} {agree_str:>6}")

        if n_total > 0:
            print(f"\n  Agreement: {n_agree}/{n_total} ({n_agree/n_total*100:.0f}%)")
            n_disagree = n_total - n_agree
            if n_disagree > 0:
                print(f"  Discordant cases: {n_disagree} — distributional collapse "
                      f"with content engagement (shortcut mechanism coexists with "
                      f"difficulty sensitivity)")

    # ──────────────────────────────────────────────────────────────
    # NULL BASELINE ANALYSIS
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NULL BASELINE ANALYSIS")
    print("=" * 70)

    for model in MODELS:
        print(f"\n  Model: {model}")
        for cond in NULL_CONDITIONS:
            key = (model, cond)
            if key not in results:
                continue
            r = results[key]
            fp = r["norm_entropy"] < 0.90
            print(f"    {cond}: entropy={r['norm_entropy']:.4f}  "
                  f"modal={r['modal_concentration']:.3f} ({r['modal_position']})  "
                  f"false_positive={'YES' if fp else 'no'}")

    # ──────────────────────────────────────────────────────────────
    # EXPLORATORY: Domain breakdown
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPLORATORY: Domain-Level Accuracy")
    print("=" * 70)

    for model in MODELS:
        print(f"\n  Model: {model}")
        print(f"  {'Cond':<5}", end="")
        domains = ["economics", "law", "physics", "psychology"]
        for d in domains:
            print(f" {d:>12}", end="")
        print()
        print(f"  {'-'*55}")

        for cond in PRIMARY_CONDITIONS:
            key = (model, cond)
            if key not in data:
                continue
            records = data[key]
            by_domain = defaultdict(list)
            for r in records:
                if r.get("parsed_response") is not None:
                    by_domain[r["domain"]].append(r)
            print(f"  {cond:<5}", end="")
            for d in domains:
                items = by_domain.get(d, [])
                if items:
                    acc = sum(1 for r in items if r.get("is_correct")) / len(items)
                    print(f" {acc:>12.3f}", end="")
                else:
                    print(f" {'':>12}", end="")
            print()

    # ──────────────────────────────────────────────────────────────
    # PRE-REG: Domain-level full gradient measures
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPLORATORY: Domain-Level Entropy & Difficulty-Accuracy ρ")
    print("=" * 70)

    domains = ["economics", "law", "physics", "psychology"]
    for model in MODELS:
        print(f"\n  Model: {model} — Entropy by domain")
        print(f"  {'Cond':<5}", end="")
        for d in domains:
            print(f" {d:>12}", end="")
        print()
        print(f"  {'-'*55}")
        for cond in PRIMARY_CONDITIONS:
            key = (model, cond)
            if key not in data:
                continue
            records = data[key]
            by_domain = defaultdict(list)
            for r in records:
                if r.get("parsed_response") is not None:
                    by_domain[r["domain"]].append(r)
            print(f"  {cond:<5}", end="")
            for d in domains:
                items = by_domain.get(d, [])
                if items:
                    counts = Counter(r["parsed_response"] for r in items)
                    total_c = sum(counts.values())
                    props = [c / total_c for c in counts.values()]
                    h = -sum(p * math.log(p) for p in props if p > 0) / math.log(10)
                    print(f" {h:>12.4f}", end="")
                else:
                    print(f" {'':>12}", end="")
            print()

    # ──────────────────────────────────────────────────────────────
    # PRE-REG: Pooled difficulty sensitivity check
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SENSITIVITY CHECK: Pooled Difficulty vs Within-Model Difficulty")
    print("=" * 70)

    for model in MODELS:
        print(f"\n  Model: {model}")
        print(f"  {'Cond':<5} {'Within-ρ':>10} {'Pooled-ρ':>10} {'Diff':>8}")
        print(f"  {'-'*40}")
        for cond in ADVERSARIAL_CONDITIONS:
            key = (model, cond)
            if key not in results:
                continue
            within = results[key].get("difficulty_rho", {}).get("rho")
            pooled = results[key].get("pooled_difficulty_rho", {}).get("rho")
            if within is not None and pooled is not None:
                diff = pooled - within
                print(f"  {cond:<5} {within:>10.4f} {pooled:>10.4f} {diff:>+8.4f}")
            else:
                print(f"  {cond:<5} {'---':>10} {'---':>10} {'---':>8}")

    # ──────────────────────────────────────────────────────────────
    # PRE-REG: Threshold sensitivity analysis
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SENSITIVITY: Concordance Across Threshold Bands")
    print("=" * 70)

    entropy_thresholds = [0.85, 0.875, 0.90, 0.925, 0.95]
    modal_thresholds = [0.35, 0.40, 0.45, 0.50]
    rho_thresholds = [0.03, 0.05, 0.07]

    for model in MODELS:
        print(f"\n  Model: {model}")
        print(f"  {'Ent_thr':>8} {'Mod_thr':>8} {'ρ_thr':>6} {'Agree':>6} {'N':>4}")
        print(f"  {'-'*40}")
        for ent_thr in entropy_thresholds:
            for mod_thr in modal_thresholds:
                for rho_thr in rho_thresholds:
                    n_agree = 0
                    n_total = 0
                    for cond in ADVERSARIAL_CONDITIONS:
                        key = (model, cond)
                        if key not in results:
                            continue
                        r = results[key]
                        screen = r["norm_entropy"] < ent_thr and r["modal_concentration"] > mod_thr
                        rho_data = r.get("difficulty_rho", {})
                        rho = rho_data.get("rho")
                        if rho is not None:
                            blind = abs(rho) < rho_thr
                            if screen == blind:
                                n_agree += 1
                            n_total += 1
                    if n_total > 0 and ent_thr == 0.90:  # Only print primary modal/rho combos at default entropy
                        pct = n_agree / n_total * 100
                        print(f"  {ent_thr:>8.3f} {mod_thr:>8.2f} {rho_thr:>6.2f} "
                              f"{n_agree}/{n_total}={pct:>3.0f}%")

    # ──────────────────────────────────────────────────────────────
    # PRE-REG: Accuracy-by-position slope
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRE-REG: Accuracy-by-Position Slope")
    print("=" * 70)

    for model in MODELS:
        print(f"\n  Model: {model}")
        print(f"  {'Cond':<5} {'Slope':>8} {'R²':>6}")
        print(f"  {'-'*25}")
        for cond in PRIMARY_CONDITIONS:
            key = (model, cond)
            if key not in results:
                continue
            ps = results[key].get("position_slope", {})
            slope = ps.get("slope")
            r_sq = ps.get("r_squared")
            if slope is not None:
                print(f"  {cond:<5} {slope:>+8.4f} {r_sq:>6.3f}")

    # ──────────────────────────────────────────────────────────────
    # PRE-REG: N4 vs N1-N3 chi-square test
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRE-REG: N4 vs N1-N3 Chi-Square Test of Homogeneity")
    print("=" * 70)

    if HAS_SCIPY:
        for model in MODELS:
            print(f"\n  Model: {model}")
            # Compare N2 and N4 (both have content/options; N1/N3 are degenerate single-letter)
            n2_key = (model, "N2")
            n4_key = (model, "N4")
            if n2_key in results and n4_key in results:
                n2_freq = np.array(results[n2_key]["freq_vec"])
                n4_freq = np.array(results[n4_key]["freq_vec"])
                # Chi-square test of homogeneity
                contingency = np.array([n2_freq, n4_freq])
                # Remove zero columns
                nonzero = contingency.sum(axis=0) > 0
                contingency = contingency[:, nonzero]
                chi2, p, dof, expected = sp_stats.chi2_contingency(contingency)
                print(f"    N2 vs N4: χ²={chi2:.2f}, df={dof}, p={p:.4f}")
                print(f"    N2 modal: {results[n2_key]['modal_position']} "
                      f"({results[n2_key]['modal_concentration']:.3f})")
                print(f"    N4 modal: {results[n4_key]['modal_position']} "
                      f"({results[n4_key]['modal_concentration']:.3f})")

            # Also note N1/N3 are degenerate (single letter), not testable via chi-square
            n1_key = (model, "N1")
            n3_key = (model, "N3")
            if n1_key in results and n3_key in results:
                print(f"    N1 modal: {results[n1_key]['modal_position']} "
                      f"(100% — degenerate, chi-square not applicable)")
                print(f"    N3 modal: {results[n3_key]['modal_position']} "
                      f"(100% — degenerate, chi-square not applicable)")

    # ──────────────────────────────────────────────────────────────
    # PRE-REG: S3 calibration check
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRE-REG: S3 Calibration Check (target: 10%, tolerance: 5-20%)")
    print("=" * 70)

    for model in MODELS:
        key = (model, "S3")
        if key not in results:
            continue
        r = results[key]
        acc = r["accuracy"]
        in_band = 0.05 <= acc <= 0.20
        print(f"\n  {model}: S3 accuracy = {acc:.3f} ({acc*100:.1f}%)")
        if in_band:
            print(f"    → WITHIN tolerance band (5-20%). Calibration success.")
        else:
            print(f"    → OUTSIDE tolerance band. Calibration failure.")
            print(f"    All distributional and difficulty-accuracy measures still reported.")

    # ──────────────────────────────────────────────────────────────
    # POST-HOC: Attractor migration table
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("POST-HOC: Attractor Migration Across Conditions")
    print("=" * 70)

    print(f"\n  {'Cond':<5}", end="")
    for model in MODELS:
        print(f" {model:>20}", end="")
    print()
    print(f"  {'-'*50}")
    for cond in ALL_CONDITIONS:
        print(f"  {cond:<5}", end="")
        for model in MODELS:
            key = (model, cond)
            if key in results:
                r = results[key]
                modal = r["modal_position"]
                conc = r["modal_concentration"]
                print(f" {modal} ({conc:.1%}){'':<8}", end="")
            else:
                print(f" {'---':>20}", end="")
        print()

    # ──────────────────────────────────────────────────────────────
    # POST-HOC: S5 attractor = N3 default position
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("POST-HOC: S5 Attractor vs Content-Absent Default (N3)")
    print("=" * 70)

    for model in MODELS:
        s5_key = (model, "S5")
        n3_key = (model, "N3")
        if s5_key in results and n3_key in results:
            s5_modal = results[s5_key]["modal_position"]
            n3_modal = results[n3_key]["modal_position"]
            match = s5_modal == n3_modal
            print(f"\n  {model}:")
            print(f"    S5 modal position: {s5_modal} "
                  f"({results[s5_key]['modal_concentration']:.1%})")
            print(f"    N3 modal position: {n3_modal} "
                  f"({results[n3_key]['modal_concentration']:.1%})")
            if match:
                print(f"    → MATCH: S5 reverts to same position as content-absent null.")
                print(f"      Interpretation: two-step instruction causes model to ignore")
                print(f"      question content, reverting to its no-question positional prior.")
            else:
                print(f"    → NO MATCH: S5 attractor differs from N3 default.")

    # ──────────────────────────────────────────────────────────────
    # POST-HOC: S5 rho sign analysis
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("POST-HOC: S5 Difficulty-Accuracy ρ Analysis")
    print("=" * 70)

    for model in MODELS:
        key = (model, "S5")
        if key not in results:
            continue
        rho_data = results[key].get("difficulty_rho", {})
        rho = rho_data.get("rho")
        p_val = rho_data.get("p_value")
        rho_ci = results[key].get("rho_ci", (None, None))
        if rho is not None:
            print(f"\n  {model}: S5 ρ = {rho:.4f}, p = {p_val:.4f}")
            if rho_ci[0] is not None:
                print(f"    Bootstrap 95% CI: [{rho_ci[0]:+.4f}, {rho_ci[1]:+.4f}]")
            ci_includes_zero = rho_ci[0] is not None and rho_ci[0] <= 0 <= rho_ci[1]
            if ci_includes_zero or (p_val is not None and p_val > ALPHA):
                print(f"    → Content-blind: ρ not significantly different from zero.")
            elif rho > 0:
                print(f"    → Slight positive ρ: consistent with positional coincidence")
                print(f"      (harder items may have correct answers at different positions).")

    # ──────────────────────────────────────────────────────────────
    # TERTIARY: Mixed-effects logistic regression
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TERTIARY: Mixed-Effects Logistic Regression (position × condition)")
    print("=" * 70)

    try:
        import statsmodels.formula.api as smf
        import pandas as pd
        HAS_SM = True
    except ImportError:
        HAS_SM = False
        print("  WARNING: statsmodels/pandas not found. Install with:")
        print("    pip install statsmodels pandas")
        print("  Skipping mixed-effects model.")

    if HAS_SM:
        for model in MODELS:
            print(f"\n  Model: {model}")

            # Build dataframe from primary conditions
            rows = []
            for cond in PRIMARY_CONDITIONS:
                key = (model, cond)
                if key not in data:
                    continue
                for r in data[key]:
                    if r.get("parsed_response") is None:
                        continue
                    correct_pos = r.get("correct_answer", "?")
                    if correct_pos in LETTERS:
                        pos_rank = LETTERS.index(correct_pos) + 1
                    else:
                        continue
                    rows.append({
                        "correct": 1 if r.get("is_correct") else 0,
                        "position": pos_rank,
                        "condition": cond,
                        "item_id": r["item_id"],
                    })

            df = pd.DataFrame(rows)
            df["condition"] = pd.Categorical(
                df["condition"],
                categories=PRIMARY_CONDITIONS,
                ordered=True
            )

            print(f"    N observations: {len(df)}")
            print(f"    N items: {df['item_id'].nunique()}")

            try:
                # Fit mixed-effects logistic regression
                # correct ~ position * condition + (1 | item_id)
                model_fit = smf.mixedlm(
                    "correct ~ position * condition",
                    data=df,
                    groups=df["item_id"],
                ).fit(reml=False)

                # Print key results
                print(f"\n    Fixed effects (position × condition interactions):")
                params = model_fit.params
                pvals = model_fit.pvalues
                for name in sorted(params.index):
                    if "position" in name.lower() and ":" in name:
                        sig = "***" if pvals[name] < ALPHA else ""
                        print(f"      {name}: β={params[name]:+.4f}, "
                              f"p={pvals[name]:.4f} {sig}")

                # Position main effect
                if "position" in params.index:
                    print(f"\n    Position main effect: β={params['position']:+.4f}, "
                          f"p={pvals['position']:.4f}")

                # Save full summary
                summary_path = os.path.join(args.output_dir, f"mixed_model_{model}.txt")
                with open(summary_path, "w") as f:
                    f.write(str(model_fit.summary()))
                print(f"    Full summary saved to {summary_path}")

            except Exception as e:
                print(f"    Mixed model failed: {e}")
                print(f"    This may require more memory or a simpler specification.")

    # ──────────────────────────────────────────────────────────────
    # Figures
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIGURES")
    print("=" * 70)

    plot_gradient(results, args.output_dir)
    plot_distributions(results, data, args.output_dir)
    plot_accuracy_by_position(results, args.output_dir)

    # ──────────────────────────────────────────────────────────────
    # Save full results as JSON
    # ──────────────────────────────────────────────────────────────
    json_results = {}
    for (model, cond), r in results.items():
        key_str = f"{model}__{cond}"
        # Convert numpy types for JSON serialisation
        clean = {}
        for k, v in r.items():
            if isinstance(v, np.floating):
                clean[k] = float(v)
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        json_results[key_str] = clean

    json_path = os.path.join(args.output_dir, "study3_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\n  Saved full results to {json_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
