"""
eval_screen: Distributional validity screening for LLM evaluation responses.

Detects content-blind positional shortcuts in multiple-choice LLM evaluation data.
Implements the two-stage screening architecture from Cacioli (2026a,b,c):
  Stage A: Screen distributional structure and content engagement.
  Stage B: Interpret substantive performance only if Stage A passes.

Reference:
  Cacioli, J.-P. (2026). Instruction complexity induces positional collapse
  in adversarial LLM evaluation. arXiv preprint. OSF: osf.io/7p64.
"""

import math
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class ScreeningReport:
    """Results of distributional validity screening for a single condition."""

    n_total: int = 0
    n_valid: int = 0
    n_parse_fail: int = 0
    parse_fail_rate: float = 0.0

    # Distributional indices
    norm_entropy: float = 0.0
    modal_concentration: float = 0.0
    top3_concentration: float = 0.0
    modal_position: str = "?"
    freq_vec: list = field(default_factory=list)

    # Content engagement
    difficulty_rho: Optional[float] = None
    difficulty_rho_p: Optional[float] = None
    difficulty_rho_n: int = 0

    # Distributional distance from baseline
    js_divergence: Optional[float] = None
    tv_distance: Optional[float] = None

    # Accuracy
    accuracy: float = 0.0

    # Screening verdicts
    distributional_flag: bool = False  # True if entropy < threshold AND modal > threshold
    content_blind_flag: bool = False   # True if |rho| < equivalence band
    regime: str = "unknown"            # content-engaged / shortcut-with-engagement / collapsed

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Distributional Validity Screen",
            f"  N items: {self.n_total} ({self.n_parse_fail} parse failures)",
            f"  Accuracy: {self.accuracy:.3f}",
            f"",
            f"  Distributional indices:",
            f"    Normalised entropy: {self.norm_entropy:.4f}",
            f"    Modal concentration: {self.modal_concentration:.3f} (position {self.modal_position})",
            f"    Top-3 concentration: {self.top3_concentration:.3f}",
        ]
        if self.js_divergence is not None:
            lines.append(f"    JS divergence from baseline: {self.js_divergence:.4f}")
            lines.append(f"    TV distance from baseline: {self.tv_distance:.4f}")

        lines.append(f"")
        lines.append(f"  Content engagement:")
        if self.difficulty_rho is not None:
            lines.append(f"    Difficulty-accuracy rho: {self.difficulty_rho:.4f} (p={self.difficulty_rho_p:.2e}, n={self.difficulty_rho_n})")
        else:
            lines.append(f"    Difficulty-accuracy rho: not computed (no baseline provided)")

        lines.append(f"")
        lines.append(f"  Screening verdict:")
        lines.append(f"    Distributional flag: {'SHORTCUT' if self.distributional_flag else 'ok'}")
        lines.append(f"    Content engagement: {'BLIND' if self.content_blind_flag else 'engaged' if self.difficulty_rho is not None else 'unknown'}")
        lines.append(f"    Regime: {self.regime}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to dictionary."""
        return {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_parse_fail": self.n_parse_fail,
            "parse_fail_rate": self.parse_fail_rate,
            "accuracy": self.accuracy,
            "norm_entropy": self.norm_entropy,
            "modal_concentration": self.modal_concentration,
            "top3_concentration": self.top3_concentration,
            "modal_position": self.modal_position,
            "difficulty_rho": self.difficulty_rho,
            "difficulty_rho_p": self.difficulty_rho_p,
            "difficulty_rho_n": self.difficulty_rho_n,
            "js_divergence": self.js_divergence,
            "tv_distance": self.tv_distance,
            "distributional_flag": self.distributional_flag,
            "content_blind_flag": self.content_blind_flag,
            "regime": self.regime,
        }


LETTERS = list("ABCDEFGHIJ")


def _normalised_entropy(freq_vec: list[int]) -> float:
    """Normalised Shannon entropy over a frequency vector. Returns H/log(K)."""
    total = sum(freq_vec)
    if total == 0:
        return 0.0
    k = len(freq_vec)
    if k <= 1:
        return 0.0
    props = [f / total for f in freq_vec]
    raw_h = -sum(p * math.log(p) for p in props if p > 0)
    return raw_h / math.log(k)


def _js_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence between two distributions."""
    p_sum = sum(p)
    q_sum = sum(q)
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p = [x / p_sum for x in p]
    q = [x / q_sum for x in q]
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    kl_pm = sum(pi * math.log(pi / mi) for pi, mi in zip(p, m) if pi > 0 and mi > 0)
    kl_qm = sum(qi * math.log(qi / mi) for qi, mi in zip(q, m) if qi > 0 and mi > 0)
    return (kl_pm + kl_qm) / 2


def _tv_distance(p: list[float], q: list[float]) -> float:
    """Total variation distance between two distributions."""
    p_sum = sum(p)
    q_sum = sum(q)
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p = [x / p_sum for x in p]
    q = [x / q_sum for x in q]
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))


def screen_responses(
    responses: list[dict],
    baseline: Optional[list[dict]] = None,
    options: Optional[list[str]] = None,
    response_key: str = "parsed_response",
    correct_key: str = "correct_answer",
    correctness_key: str = "is_correct",
    item_id_key: str = "item_id",
    entropy_threshold: float = 0.90,
    modal_threshold: float = 0.40,
    rho_equivalence: float = 0.05,
) -> ScreeningReport:
    """
    Screen a set of LLM evaluation responses for distributional validity.

    Parameters
    ----------
    responses : list of dict
        Each dict must contain at minimum the response_key field.
        For content-engagement analysis, also needs correct_key or correctness_key and item_id_key.
    baseline : list of dict, optional
        Honest-condition responses for difficulty estimation. Same format as responses.
        Required for difficulty-accuracy correlation (content-engagement criterion).
    options : list of str, optional
        The option labels (default: ["A", "B", ..., "J"]).
    response_key : str
        Key for the model's parsed response letter.
    correct_key : str
        Key for the correct answer letter.
    correctness_key : str
        Key for binary correctness (True/False or 1/0).
    item_id_key : str
        Key for unique item identifier.
    entropy_threshold : float
        Entropy below this flags distributional collapse (default 0.90).
    modal_threshold : float
        Modal concentration above this flags distributional collapse (default 0.40).
    rho_equivalence : float
        Absolute rho below this is treated as content-blind (default 0.05).

    Returns
    -------
    ScreeningReport
        Contains all indices, flags, and regime classification.
    """
    if options is None:
        options = LETTERS

    report = ScreeningReport()

    # Separate valid and parse-failed responses
    valid = [r for r in responses if r.get(response_key) is not None]
    report.n_total = len(responses)
    report.n_valid = len(valid)
    report.n_parse_fail = report.n_total - report.n_valid
    report.parse_fail_rate = report.n_parse_fail / report.n_total if report.n_total > 0 else 0.0

    if report.n_valid == 0:
        report.regime = "no_data"
        return report

    # Response distribution
    resp_counts = Counter(r[response_key] for r in valid)
    freq_vec = [resp_counts.get(opt, 0) for opt in options]
    total = sum(freq_vec)
    freq_prop = [f / total for f in freq_vec] if total > 0 else freq_vec

    report.freq_vec = freq_vec

    # Normalised entropy
    report.norm_entropy = _normalised_entropy(freq_vec)

    # Modal concentration
    max_freq = max(freq_prop) if freq_prop else 0.0
    report.modal_concentration = max_freq
    modal_idx = freq_prop.index(max_freq) if freq_prop else 0
    report.modal_position = options[modal_idx] if modal_idx < len(options) else "?"

    # Top-3 concentration
    sorted_props = sorted(freq_prop, reverse=True)
    report.top3_concentration = sum(sorted_props[:3])

    # Accuracy
    n_correct = sum(1 for r in valid if r.get(correctness_key) is True or r.get(correctness_key) == 1)
    report.accuracy = n_correct / report.n_valid

    # JS divergence and TV distance from baseline
    if baseline is not None:
        baseline_valid = [r for r in baseline if r.get(response_key) is not None]
        if baseline_valid:
            baseline_counts = Counter(r[response_key] for r in baseline_valid)
            baseline_freq = [baseline_counts.get(opt, 0) for opt in options]
            baseline_total = sum(baseline_freq)
            baseline_prop = [f / baseline_total for f in baseline_freq] if baseline_total > 0 else baseline_freq
            report.js_divergence = _js_divergence(freq_prop, baseline_prop)
            report.tv_distance = _tv_distance(freq_prop, baseline_prop)

    # Difficulty-accuracy correlation
    if baseline is not None and HAS_SCIPY:
        # Build item-level difficulty from baseline
        baseline_correct = {}
        for r in baseline:
            if r.get(response_key) is not None:
                iid = r.get(item_id_key)
                if iid is not None:
                    baseline_correct[iid] = 1 if (r.get(correctness_key) is True or r.get(correctness_key) == 1) else 0

        # Build item-level correctness under this condition
        adv_correct = {}
        for r in valid:
            iid = r.get(item_id_key)
            if iid is not None:
                adv_correct[iid] = 1 if (r.get(correctness_key) is True or r.get(correctness_key) == 1) else 0

        # Match items
        common_ids = sorted(set(baseline_correct.keys()) & set(adv_correct.keys()))
        if len(common_ids) >= 10:
            difficulties = [1 - baseline_correct[iid] for iid in common_ids]
            adv_scores = [adv_correct[iid] for iid in common_ids]
            rho, p_val = sp_stats.spearmanr(difficulties, adv_scores)
            report.difficulty_rho = float(rho)
            report.difficulty_rho_p = float(p_val)
            report.difficulty_rho_n = len(common_ids)

    # Screening verdicts
    report.distributional_flag = (
        report.norm_entropy < entropy_threshold and
        report.modal_concentration > modal_threshold
    )

    if report.difficulty_rho is not None:
        report.content_blind_flag = abs(report.difficulty_rho) < rho_equivalence

    # Regime classification
    if report.distributional_flag and report.content_blind_flag:
        report.regime = "collapsed"
    elif report.distributional_flag and not report.content_blind_flag:
        report.regime = "shortcut-with-engagement"
    elif not report.distributional_flag and not report.content_blind_flag:
        report.regime = "content-engaged"
    elif not report.distributional_flag and report.content_blind_flag:
        report.regime = "uniform-blind"  # rare: uniform distribution but no content sensitivity
    elif report.difficulty_rho is None:
        if report.distributional_flag:
            report.regime = "shortcut (content unknown)"
        else:
            report.regime = "ok (content unknown)"
    else:
        report.regime = "unknown"

    return report


def screen_from_jsonl(
    responses_path: str,
    baseline_path: Optional[str] = None,
    **kwargs,
) -> ScreeningReport:
    """
    Screen responses from JSONL files.

    Parameters
    ----------
    responses_path : str
        Path to JSONL file with model responses.
    baseline_path : str, optional
        Path to JSONL file with honest-baseline responses.

    Returns
    -------
    ScreeningReport
    """
    with open(responses_path) as f:
        responses = [json.loads(line.strip()) for line in f if line.strip()]

    baseline = None
    if baseline_path is not None:
        with open(baseline_path) as f:
            baseline = [json.loads(line.strip()) for line in f if line.strip()]

    return screen_responses(responses, baseline=baseline, **kwargs)
