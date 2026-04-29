"""
generate_figures.py

Generates publication-quality figures for the positional-bias paper.
Run from D:\bcb_pilot with venv activated.

Usage:
    python generate_figures.py

Requires: matplotlib, numpy, pandas
Outputs: figures/ directory with PNG and PDF versions
"""

import json
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join("results", "positional_bias")
PERM_FILE = os.path.join("data", "permutation_assignments.csv")
FIG_DIR = "figures"
OPTION_LETTERS = list("ABCDEFGHIJ")

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'honest': '#4878CF',
    'sandbag_orig': '#D65F5F',
    'sandbag_perm': '#B07AA1',
    'chance': '#888888',
    'accent': '#D4A03C',
}


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


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load data
    a_orig = load_jsonl(os.path.join(RESULTS_DIR, "llama3-8b_A_original.jsonl"))
    b_orig = load_jsonl(os.path.join(RESULTS_DIR, "llama3-8b_B_original.jsonl"))
    b_perm = load_jsonl(os.path.join(RESULTS_DIR, "llama3-8b_B_perm.jsonl"))
    perm_df = pd.read_csv(PERM_FILE)

    vec_a = response_vector(a_orig["parsed_response"]) * 100
    vec_b = response_vector(b_orig["parsed_response"]) * 100
    vec_bp = response_vector(b_perm["parsed_response"]) * 100

    # ==================================================================
    # Figure 1: Response-position distributions (3 conditions)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(OPTION_LETTERS))
    width = 0.25

    bars_a = ax.bar(x - width, vec_a, width, label='A-original (honest)',
                    color=COLORS['honest'], edgecolor='white', linewidth=0.5)
    bars_b = ax.bar(x, vec_b, width, label='B-original (sandbagging)',
                    color=COLORS['sandbag_orig'], edgecolor='white', linewidth=0.5)
    bars_bp = ax.bar(x + width, vec_bp, width, label='B-perm (sandbagging, permuted)',
                     color=COLORS['sandbag_perm'], edgecolor='white', linewidth=0.5)

    ax.axhline(y=10, color=COLORS['chance'], linestyle='--', linewidth=1,
               label='Uniform (10%)', alpha=0.7)

    # Annotate peak
    peak_idx = np.argmax(vec_b)
    ax.annotate(f'{vec_b[peak_idx]:.1f}%',
                xy=(peak_idx, vec_b[peak_idx]),
                xytext=(peak_idx + 0.8, vec_b[peak_idx] + 3),
                fontsize=10, fontweight='bold', color=COLORS['sandbag_orig'],
                arrowprops=dict(arrowstyle='->', color=COLORS['sandbag_orig'], lw=1.2))

    ax.set_xlabel('Response position')
    ax.set_ylabel('Response frequency (%)')
    ax.set_title('Llama-3-8B: Response-position distributions under three conditions')
    ax.set_xticks(x)
    ax.set_xticklabels(OPTION_LETTERS)
    ax.set_ylim(0, 60)
    ax.legend(loc='upper right', framealpha=0.9)

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig1_distributions.{fmt}'))
    plt.close(fig)
    print("Figure 1: Response-position distributions saved.")

    # ==================================================================
    # Figure 2: Accuracy by correct-answer position (B-perm)
    # ==================================================================
    b_perm_s = b_perm.sort_values("item_id").reset_index(drop=True)

    positions = []
    accuracies = []
    ci_low = []
    ci_high = []
    ns = []

    for letter in OPTION_LETTERS:
        mask = b_perm_s["answer_key"] == letter
        vals = b_perm_s.loc[mask, "correct"].fillna(0).values
        n = len(vals)
        if n > 0:
            acc = vals.mean() * 100
            se = np.sqrt(acc/100 * (1 - acc/100) / n) * 100
            positions.append(letter)
            accuracies.append(acc)
            ci_low.append(max(0, acc - 1.96 * se))
            ci_high.append(min(100, acc + 1.96 * se))
            ns.append(n)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(positions))
    bar_colors = [COLORS['sandbag_perm']] * len(positions)
    # Highlight E
    e_idx = positions.index('E')
    bar_colors[e_idx] = COLORS['sandbag_orig']

    errors = [np.array(accuracies) - np.array(ci_low),
              np.array(ci_high) - np.array(accuracies)]

    bars = ax.bar(x, accuracies, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.errorbar(x, accuracies, yerr=errors, fmt='none', ecolor='#333333',
                capsize=3, capthick=1, linewidth=1)

    ax.axhline(y=10, color=COLORS['chance'], linestyle='--', linewidth=1,
               label='Chance (10%)', alpha=0.7)
    ax.axhline(y=38.0, color=COLORS['honest'], linestyle=':', linewidth=1,
               label='Honest baseline (38.0%)', alpha=0.7)

    # Annotate E spike
    ax.annotate(f'+34.1 pp\n(above honest)',
                xy=(e_idx, accuracies[e_idx]),
                xytext=(e_idx + 1.5, accuracies[e_idx] - 5),
                fontsize=9, color=COLORS['sandbag_orig'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['sandbag_orig'], lw=1.2))

    # Annotate A drop
    a_idx = positions.index('A')
    ax.annotate(f'-33.7 pp',
                xy=(a_idx, accuracies[a_idx]),
                xytext=(a_idx + 1.2, accuracies[a_idx] + 12),
                fontsize=9, color='#333333',
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1))

    ax.set_xlabel('Position of correct answer')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Llama-3-8B: Accuracy by correct-answer position under B-perm')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylim(0, 85)
    ax.legend(loc='upper right', framealpha=0.9)

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig2_accuracy_by_position.{fmt}'))
    plt.close(fig)
    print("Figure 2: Accuracy by correct-answer position saved.")

    # ==================================================================
    # Figure 3: JS divergence comparison
    # ==================================================================
    from scipy.spatial.distance import jensenshannon

    vec_a_raw = response_vector(a_orig["parsed_response"])
    vec_b_raw = response_vector(b_orig["parsed_response"])
    vec_bp_raw = response_vector(b_perm["parsed_response"])

    js_ab = jensenshannon(vec_a_raw, vec_b_raw)
    js_bp = jensenshannon(vec_b_raw, vec_bp_raw)

    fig, ax = plt.subplots(figsize=(5, 4))

    bars = ax.bar(
        [0, 1],
        [js_ab, js_bp],
        color=[COLORS['honest'], COLORS['sandbag_perm']],
        edgecolor='white',
        linewidth=0.5,
        width=0.5,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        'Honest vs\nSandbagging\n(instruction effect)',
        'B-original vs\nB-perm\n(content rotation)',
    ], fontsize=10)
    ax.set_ylabel('Jensen-Shannon divergence')
    ax.set_title('Distributional shift: instruction vs content rotation')

    # Annotate values
    for i, (bar, val) in enumerate(zip(bars, [js_ab, js_bp])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Annotate ratio
    ax.annotate(f'{js_ab/js_bp:.0f}x',
                xy=(0.5, (js_ab + js_bp) / 2),
                fontsize=14, fontweight='bold', color='#333333',
                ha='center', va='center')

    ax.set_ylim(0, 0.5)

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig3_js_divergence.{fmt}'))
    plt.close(fig)
    print("Figure 3: JS divergence comparison saved.")

    # ==================================================================
    # Figure 4: Per-item classification breakdown
    # ==================================================================
    b_orig_s = b_orig.sort_values("item_id").reset_index(drop=True)
    b_perm_s = b_perm.sort_values("item_id").reset_index(drop=True)
    perm_lookup = perm_df.set_index("item_id")["shift_k"].to_dict()

    classes = []
    for idx in range(len(b_orig_s)):
        ro = b_orig_s.iloc[idx]["parsed_response"]
        rp = b_perm_s.iloc[idx]["parsed_response"]
        iid = b_orig_s.iloc[idx]["item_id"]
        k = perm_lookup.get(iid, 0)
        if pd.isna(ro) or pd.isna(rp):
            classes.append("Other")
            continue
        oi = OPTION_LETTERS.index(ro)
        cp = OPTION_LETTERS[(oi + k) % 10]
        if rp == ro:
            classes.append("Same-letter\n(position)")
        elif rp == cp:
            classes.append("Shifted-content\n(content)")
        else:
            classes.append("Other")

    cc = Counter(classes)
    labels = ["Same-letter\n(position)", "Shifted-content\n(content)", "Other"]
    counts = [cc.get(l, 0) for l in labels]
    pcts = [c / sum(counts) * 100 for c in counts]
    colors_pie = [COLORS['sandbag_orig'], COLORS['sandbag_perm'], COLORS['chance']]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(range(len(labels)), pcts, color=colors_pie, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Percentage of items')
    ax.set_title('Llama-3-8B: Per-item mechanism classification (T = 0)')
    ax.set_xlim(0, 55)

    for i, (bar, pct, count) in enumerate(zip(bars, pcts, counts)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}% (n={count})', va='center', fontsize=10)

    ax.axvline(x=10, color=COLORS['chance'], linestyle='--', linewidth=1, alpha=0.5)
    ax.text(10.5, -0.4, 'chance\n(10%)', fontsize=8, color=COLORS['chance'], alpha=0.7)

    ax.invert_yaxis()

    for fmt in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig4_classification.{fmt}'))
    plt.close(fig)
    print("Figure 4: Per-item classification saved.")

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
